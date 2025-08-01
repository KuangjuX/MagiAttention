import os

import torch
import torch.distributed as dist

# --- Imports (assuming they are correct) ---
from zigzag_flex_flash_attn import zigzag_flex_flash_attn_varlen_func
from magi_attention.common.ranges import AttnRanges
from magi_attention.functional.dist_attn import dist_attn_func
from magi_attention.dist_attn_runtime_mgr import init_dist_attn_runtime_mgr
from magi_attention.config import DistAttnConfig
from magi_attention.common.enum import AttnMaskType
from magi_attention.benchmarking import do_bench

# --- FLOPs calculator (assuming it's available) ---
try:
    from tools.calculate_flops import calculate_attn_flops
except ImportError:
    from magi_attention.meta._calc_dispatch_meta import _calc_self_attn_areas
    def calculate_attn_flops(
        q_ranges: AttnRanges, k_ranges: AttnRanges, attn_mask_type: list[AttnMaskType],
        total_seqlen_q: int, num_heads_q: int, head_dim: int,
    ) -> dict[str, float]:
        attn_area = _calc_self_attn_areas(
            q_ranges, k_ranges, attn_mask_type,
            num_chunks=1, chunk_size=total_seqlen_q,
        ).area
        flops_fwd = 4 * attn_area * num_heads_q * head_dim
        flops_bwd = flops_fwd * 2.5
        flops_1f1b = flops_fwd + flops_bwd
        return {"fwd": flops_fwd, "bwd": flops_bwd, "1f1b": flops_1f1b}


def calculate_global_causal_varlen_flops(
    world_size: int,
    batch_size: int,
    seqlen: int,
    num_heads: int,
    head_dim: int,
    forward_only: bool,
) -> float:
    """
    Calculates the total theoretical FLOPs for a distributed causal varlen attention task.

    This function models the entire problem as a single, large causal attention
    computation spread across all GPUs. The resulting FLOPs value is the
    theoretical total for the entire system, and is applicable to any algorithm
    (like Magi or Zigzag) that correctly solves this problem.

    Args:
        world_size: The number of GPUs in the process group.
        batch_size: The number of sequences on EACH GPU.
        seqlen: The length of each sequence.
        num_heads: The number of attention heads.
        head_dim: The dimension of each attention head.
        forward_only: If True, calculates forward pass FLOPs. Otherwise, fwd + bwd.

    Returns:
        The total theoretical FLOPs for the distributed computation.
    """
    # 1. Define the sequence layout for a SINGLE rank.
    # This assumes uniform sequence lengths for simplicity, matching the benchmark.
    # ... (steps 1 and 2, building the global lists, are all correct and remain unchanged) ...
    local_cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32)
    global_q_ranges_list = []
    global_k_ranges_list = []
    total_tokens = 0
    for rank in range(world_size):
        rank_offset = rank * batch_size * seqlen
        for i in range(batch_size):
            seq_start = rank_offset + local_cu_seqlens[i].item()
            seq_end = rank_offset + local_cu_seqlens[i+1].item()
            global_q_ranges_list.append([seq_start, seq_end])
            global_k_ranges_list.append([0, seq_end]) # Causality is defined here
        total_tokens += batch_size * seqlen

    global_q_ranges = AttnRanges.from_ranges(global_q_ranges_list)
    global_k_ranges = AttnRanges.from_ranges(global_k_ranges_list)
    global_batch_size = world_size * batch_size

    # 3. Use the precise calculator with the CORRECT mask type.
    # We tell the calculator that each sub-problem is a full, dense computation.
    flops_dict = calculate_attn_flops(
        q_ranges=global_q_ranges,
        k_ranges=global_k_ranges,
        # *** THE FIX IS HERE ***
        attn_mask_type=[AttnMaskType.FULL] * global_batch_size,
        total_seqlen_q=total_tokens,
        num_heads_q=num_heads,
        head_dim=head_dim,
    )

    # 4. Return the final FLOPs count (unchanged).
    total_flops = flops_dict["fwd"] if forward_only else flops_dict["1f1b"]
    return total_flops



def full_attention_to_varlen_attention(batch_size: int, seqlen: int):
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen,
        dtype=torch.int32, device=torch.cuda.current_device(),
    )
    return cu_seqlens, cu_seqlens

### MODIFICATION START 1 ###
# Modify the function to return the global metadata we need.
# def create_magi_runtime_with_mgr_causal_varlen(
#     cu_seqlens: torch.Tensor,
#     cp_group: dist.ProcessGroup,
#     chunk_size: int = 1024,
# ) -> Tuple[DistAttnRuntimeMgr, AttnRanges, AttnRanges, int]:
#     """
#     Creates a MagiAttention runtime manager and also returns the global
#     attention metadata needed for FLOPs calculation.
#     """
#     # 1. Calculate global information
#     total_seqlen = cu_seqlens[-1].item()
#     num_seqs = len(cu_seqlens) - 1

#     # 2. Construct GLOBAL q_ranges and k_ranges
#     q_ranges_list = [[cu_seqlens[i].item(), cu_seqlens[i+1].item()] for i in range(num_seqs)]
#     global_q_ranges = AttnRanges.from_ranges(q_ranges_list)

#     k_ranges_list = [[0, cu_seqlens[i+1].item()] for i in range(num_seqs)]
#     global_k_ranges = AttnRanges.from_ranges(k_ranges_list)

#     # 3. Call the core initialization function
#     dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
#         q_ranges=global_q_ranges,
#         k_ranges=global_k_ranges,
#         attn_mask_type=AttnMaskType.CAUSAL,
#         total_seqlen_q=total_seqlen,
#         total_seqlen_k=total_seqlen,
#         chunk_size=chunk_size,
#         cp_group=cp_group,
#         is_same_source=True,
#         is_q_permutable=True,
#         is_k_permutable=True,
#         dist_attn_config=DistAttnConfig(),
#     )

#     # 4. Return the manager AND the global metadata
#     return dist_attn_runtime_mgr, global_q_ranges, global_k_ranges, total_seqlen
# ### MODIFICATION END 1 ###


# def benchmark_attention(
#     attn_type: str,
#     local_total_tokens: int,
#     batch_size: int,
#     num_heads: int,
#     num_kv_heads: int,
#     head_dim: int,
#     num_iter: int,
#     forward_only: bool,
#     cp_groups: List[dist.ProcessGroup],
# ):
#     dtype = torch.bfloat16
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     device = torch.device(f"cuda:{rank}")
#     torch.cuda.set_device(device)

#     # seqlen = local_total_tokens // batch_size
#     # cu_seqlens, _ = full_attention_to_varlen_attention(batch_size, seqlen)

#     seqlen = local_total_tokens // batch_size
    
#     # --- 正确的全局元数据构建 ---

#     # 1. 定义本地的序列布局 (这和之前一样)
#     local_cu_seqlens, _ = full_attention_to_varlen_attention(batch_size, seqlen)
#     num_local_seqs = len(local_cu_seqlens) - 1

#     # 2. 构建跨所有 rank 的全局 q_ranges 和 k_ranges
#     global_q_ranges_list = []
#     global_k_ranges_list = []

#     for r in range(world_size):
#         # 计算 rank 'r' 的 token 偏移量
#         rank_offset = r * local_total_tokens
#         for i in range(num_local_seqs):
#             # 计算每个序列在全局范围内的起始和结束位置
#             global_seq_start = rank_offset + local_cu_seqlens[i].item()
#             global_seq_end = rank_offset + local_cu_seqlens[i+1].item()
            
#             # 全局 Q 的范围就是序列自身的位置
#             global_q_ranges_list.append([global_seq_start, global_seq_end])
            
#             # 对于因果注意力，全局 K 的范围是从最开始 (token 0) 
#             # 一直到当前序列的结束位置
#             global_k_ranges_list.append([0, global_seq_end])

#     global_q_ranges = AttnRanges.from_ranges(global_q_ranges_list)
#     global_k_ranges = AttnRanges.from_ranges(global_k_ranges_list)
    
#     # 3. 计算真正的全局总序列长度
#     global_total_seqlen = world_size * local_total_tokens
#     global_num_seqs = world_size * num_local_seqs

#     # if rank == 0:
#     #     print(f"global_total_seqlen: {global_total_seqlen}")
#     #     print(f"global_num_seqs: {global_num_seqs}")


#     q = torch.randn(local_total_tokens, num_heads, head_dim, device=device, dtype=dtype, requires_grad=not forward_only)
#     k = torch.randn(local_total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=not forward_only)
#     v = torch.randn(local_total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=not forward_only)
    
#     if attn_type == "magi":
#         # runtime_mgr, global_q_ranges, global_k_ranges, total_seqlen = create_magi_runtime_with_mgr_causal_varlen(
#         #     cu_seqlens, cp_groups[0]
#         # )
#         # runtime = runtime_mgr.dist_attn_runtime

#         runtime_mgr = init_dist_attn_runtime_mgr(
#             q_ranges=global_q_ranges,
#             k_ranges=global_k_ranges,
#             attn_mask_type=AttnMaskType.CAUSAL,
#             total_seqlen_q=global_total_seqlen,
#             total_seqlen_k=global_total_seqlen,
#             chunk_size=1024,
#             cp_group=cp_groups[0],
#             is_same_source=True,
#             is_q_permutable=True,
#             is_k_permutable=True,
#             dist_attn_config=DistAttnConfig(),
#         )
#         runtime = runtime_mgr.dist_attn_runtime
#     elif attn_type == "zigzag":
#         ranges_tensor = torch.stack([local_cu_seqlens[:-1], local_cu_seqlens[1:]], dim=1).to(device)
#         q_ranges_tensor = ranges_tensor
#         k_ranges_tensor = ranges_tensor
#         max_seqlen_q = max_seqlen_k = seqlen
#         sm_margin = 0
#     else:
#         raise ValueError(f"Unknown attention type: {attn_type}")

#     # --- Warmup ---
#     # (Warmup code is correct and unchanged)
#     for _ in range(10):
#         if attn_type == "magi":
#             _, _ = dist_attn_func(q, k, v, runtime)
#         else:
#             _ = zigzag_flex_flash_attn_varlen_func(
#                 q, k, v, max_seqlen_q, max_seqlen_k, sm_margin,
#                 ranges_tensor, q_ranges_tensor, k_ranges_tensor,
#                 process_group=cp_groups[0], dgrad_process_group=cp_groups[1]
#             )

#     # 确保所有 GPU 同步开始
#     torch.cuda.synchronize()
#     dist.barrier()

#     # --- 使用 CUDA events 进行计时 ---
#     # 创建 CUDA events
#     start_event = torch.cuda.Event(enable_timing=True)
#     start_event.record()

#     for i in range(num_iter):
#         if attn_type == "magi":
#             _, _ = dist_attn_func(q, k, v, runtime)
#         else:
#             _ = zigzag_flex_flash_attn_varlen_func(
#                 q, k, v, max_seqlen_q, max_seqlen_k, sm_margin,
#                 ranges_tensor, q_ranges_tensor, k_ranges_tensor,
#                 process_group=cp_groups[0], dgrad_process_group=cp_groups[1]
#             )

#     end_event = torch.cuda.Event(enable_timing=True)
#     end_event.record()
#     torch.cuda.synchronize()
#     dist.barrier()
#     total_time = start_event.elapsed_time(end_event) / 1000.0

#     # 计算性能指标
#     if attn_type == "magi":
#         flops_dict = calculate_attn_flops(
#             q_ranges=global_q_ranges,
#             k_ranges=global_k_ranges,
#             attn_mask_type=[AttnMaskType.CAUSAL] * global_num_seqs,
#             total_seqlen_q=global_total_seqlen,
#             num_heads_q=num_heads,
#             head_dim=head_dim,
#         )
#         total_flops = flops_dict["fwd"] if forward_only else flops_dict["1f1b"]
#         tflops = total_flops / total_time / 1e12
#     else:
#         total_flops = calculate_global_causal_varlen_flops(
#             world_size, batch_size, seqlen, num_heads, head_dim, forward_only
#         ) 
#         tflops = total_flops / total_time / 1e12

#     sec_per_iter = total_time / num_iter

#     if rank == 0:
#         print(f"| {attn_type:<10} | {'Causal Varlen':<14} | {world_size:<4} | {local_total_tokens:<12} | {sec_per_iter:<10.4f} | {tflops:<10.2f} |")

def dist_do_bench(fn, warmup=25, rep=100, grad_to_none=None):
    """
    A simplified, distributed-safe version of triton.testing.do_bench.
    It uses a fixed number of repetitions (`rep`) across all ranks.
    """
    # 1. Warm-up: All ranks run the same number of warmup iterations.
    for _ in range(warmup):
        fn()
    
    # Ensure all ranks have finished warmup before starting the timer.
    torch.cuda.synchronize()
    dist.barrier()

    # 2. Benchmark: All ranks run the same number of benchmark iterations.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(rep):
        fn()
    end_event.record()

    # 3. Synchronize and get timing
    torch.cuda.synchronize()
    # No need for a barrier here because end_event.record() is placed after
    # the last fn() call, which itself contains a barrier.
    # The final synchronize() ensures the event is captured.

    # Calculate total time for `rep` iterations
    total_time_ms = start_event.elapsed_time(end_event)
    
    # Return the average time per iteration in ms
    return total_time_ms / rep


def calculate_zigzag_causal_forward_flops(
    ranges_tensor: torch.Tensor,
    q_ranges_tensor: torch.Tensor,
    k_ranges_tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
    world_size: int,
) -> float:
    """
    精确计算 zigzag_flex_flash_attn_varlen_forward 的前向传播 FLOPs。
    这个函数的逻辑必须完美匹配实际执行代码的循环和条件。
    """
    
    # 辅助函数，计算一个 attention block 的 FLOPs
    # M = query tokens, N = key/value tokens, D = head_dim
    def get_attn_flops(m: int, n: int, d: int) -> float:
        # 忽略 softmax，只计算两个主要矩阵乘法
        # Q @ K.T  -> 2 * M * N * D
        # P @ V    -> 2 * M * N * D
        return 4.0 * m * n * d

    # 辅助函数，从 ranges_tensor 计算总 token 数
    def count_tokens(ranges: torch.Tensor) -> int:
        if ranges.numel() == 0:
            return 0
        return torch.sum(ranges[:, 1] - ranges[:, 0]).item()

    # 获取不同部分的 token 数量
    # 注意：这里假设所有 rank 的 local_tokens 数量是相同的，
    # 并且 q_ranges/k_ranges 也是对称的。
    # 在实际 benchmark 中，每个 rank 的 token 数可能略有不同，
    # 但为了计算总 FLOPs，使用一个 rank 的情况乘以 world_size 是合理的。
    q_full_len = count_tokens(ranges_tensor)
    k_full_len = count_tokens(ranges_tensor)
    q_half_len = count_tokens(q_ranges_tensor)
    k_half_len = count_tokens(k_ranges_tensor)

    # 计算单个 rank 的总 FLOPs
    rank_flops = 0.0

    # 模拟 zigzag 的 N 步循环
    for step in range(world_size):
        # 这里的 rank 是一个代表性的 rank，我们只计算一个 rank 的工作量
        # 假设 rank = 0
        rank = 0 

        # Step 0: 本地计算
        if step == 0:
            # 本地是因果注意力，计算量约为全注意力的一半
            # S = Q @ K.T 是下三角，所以是 M*N*D
            # P @ V 仍然是 M*N*D
            # 总计 3 * M * N * D，或者简单地用全计算的一半来近似
            # 为了更精确，我们知道因果掩码的面积是 N^2/2
            # 所以 FLOPs 是 2 * (N^2/2) * D + 2 * (N^2/2) * D = 2 * N^2 * D
            # 这里用全计算的一半来近似，即 0.5 * get_attn_flops
            # 注意：FlashAttention 的实现可能更优化，但这是一个很好的理论近似
            block_flops = 0.5 * get_attn_flops(q_full_len, k_full_len, head_dim)
            rank_flops += block_flops
        
        # 在 zigzag 算法中，rank 的判断是相对于 step 的
        # 我们需要模拟一个通用 rank 的行为
        # 为了计算总 FLOPs，我们可以计算所有 rank 的 FLOPs 然后求和
        # 或者，利用对称性，计算一个 rank 的 FLOPs 然后乘以 world_size
        # 让我们用更精确的前者
        pass # 下面的循环会处理所有 rank

    # 让我们换一种更精确的计算方式：计算整个系统的总 FLOPs
    total_system_flops = 0.0
    for step in range(world_size):
        for rank in range(world_size):
            block_flops = 0.0
            if step == 0:
                # 只有本地计算，每个 rank 都做
                # 因果注意力，FLOPs 约为全注意力的 1/2
                block_flops = 0.5 * get_attn_flops(q_full_len, k_full_len, head_dim)
                # 注意：这里有个错误，step=0 的计算只发生一次，不应该在 rank 循环里
                # 我们需要重新组织这个逻辑
                pass

    # --- 正确的逻辑 ---
    total_system_flops = 0.0
    
    # Step 0: 所有 rank 都执行本地因果计算
    local_causal_flops = 0.5 * get_attn_flops(q_full_len, k_full_len, head_dim)
    total_system_flops += world_size * local_causal_flops

    # Step 1 to world_size - 1:
    for step in range(1, world_size):
        # 在这一步，有 `step` 个 rank 执行 "else" 分支 (rank < step)
        # 有 `world_size - step` 个 rank 执行 "elif" 分支 (rank >= step)
        
        # "else" 分支 (rank < step): 计算 Q_half vs K_full
        else_branch_flops = get_attn_flops(q_half_len, k_full_len, head_dim)
        total_system_flops += step * else_branch_flops

        # "elif" 分支 (rank >= step): 计算 Q_full vs K_half
        elif_branch_flops = get_attn_flops(q_full_len, k_half_len, head_dim)
        total_system_flops += (world_size - step) * elif_branch_flops

    # 乘以注意力头数
    total_system_flops *= num_heads
    
    return total_system_flops



def benchmark_attention(
    attn_type: str,
    local_total_tokens: int,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_iter: int, # num_iter 现在可以被 do_bench 的 rep 参数替代
    forward_only: bool,
    cp_groups: list[dist.ProcessGroup],
):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    seqlen = local_total_tokens // batch_size
    
    # --- 全局元数据构建 (这部分代码保持不变) ---
    local_cu_seqlens, _ = full_attention_to_varlen_attention(batch_size, seqlen)
    num_local_seqs = len(local_cu_seqlens) - 1
    global_q_ranges_list = []
    global_k_ranges_list = []
    for r in range(world_size):
        rank_offset = r * local_total_tokens
        for i in range(num_local_seqs):
            global_seq_start = rank_offset + local_cu_seqlens[i].item()
            global_seq_end = rank_offset + local_cu_seqlens[i+1].item()
            global_q_ranges_list.append([global_seq_start, global_seq_end])
            global_k_ranges_list.append([0, global_seq_end])
    global_q_ranges = AttnRanges.from_ranges(global_q_ranges_list)
    global_k_ranges = AttnRanges.from_ranges(global_k_ranges_list)
    global_total_seqlen = world_size * local_total_tokens
    global_num_seqs = world_size * num_local_seqs

    # --- 输入张量 (保持不变) ---
    q = torch.randn(local_total_tokens, num_heads, head_dim, device=device, dtype=dtype, requires_grad=not forward_only)
    k = torch.randn(local_total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=not forward_only)
    v = torch.randn(local_total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=not forward_only)
    
    # --- 运行时和参数准备 (保持不变) ---
    runtime = None
    if attn_type == "magi":
        runtime_mgr = init_dist_attn_runtime_mgr(
            q_ranges=global_q_ranges,
            k_ranges=global_k_ranges,
            attn_mask_type=AttnMaskType.CAUSAL,
            total_seqlen_q=global_total_seqlen,
            total_seqlen_k=global_total_seqlen,
            chunk_size=1024,
            cp_group=cp_groups[0],
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            dist_attn_config=DistAttnConfig(),
        )
        runtime = runtime_mgr.dist_attn_runtime
    elif attn_type == "zigzag":
        ranges_tensor = torch.stack([local_cu_seqlens[:-1], local_cu_seqlens[1:]], dim=1).to(device)
        q_ranges_tensor = ranges_tensor
        k_ranges_tensor = ranges_tensor
        max_seqlen_q = max_seqlen_k = seqlen
        sm_margin = 0
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

    # --- 定义要被 do_bench 测量的函数 ---
    # 这是关键的封装步骤
    def get_bench_fn():
        if attn_type == "magi":
            def magi_fn():
                # 在函数内部调用，并加上 barrier
                _, _ = dist_attn_func(q, k, v, runtime)
                dist.barrier() # 必须！确保所有 rank 都完成了
            return magi_fn
        else: # zigzag
            def zigzag_fn():
                # 在函数内部调用，并加上 barrier
                _ = zigzag_flex_flash_attn_varlen_func(
                    q, k, v, max_seqlen_q, max_seqlen_k, sm_margin,
                    ranges_tensor, q_ranges_tensor, k_ranges_tensor,
                    process_group=cp_groups[0], dgrad_process_group=cp_groups[1]
                )
                dist.barrier() # 必须！确保所有 rank 都完成了
            return zigzag_fn

    bench_fn = get_bench_fn()

    # --- 使用 do_bench 进行基准测试 ---
    # warmup 和 rep 的单位是 ms，可以根据需要调整
    # 我们使用 return_mode='median' 来获得更稳定的结果
    # `rep` 控制了测量的总时间，可以替代原来的 `num_iter`
    
    # 确保所有 rank 都准备好了再开始 benchmark
    dist.barrier()
    
    # 调用 do_bench
    # 注意：do_bench 内部有自己的 warmup，所以外部的 warmup 循环可以移除
    sec_per_iter = dist_do_bench(
        bench_fn, 
        warmup=10, 
        rep=num_iter
    ) / 1000.0
    


    # total_flops = calculate_global_causal_varlen_flops(
    #     world_size, batch_size, seqlen, num_heads, head_dim, forward_only
    # )

    total_flops = calculate_global_causal_varlen_flops(
        world_size, batch_size, seqlen, num_heads, head_dim, forward_only
    )
   
    # TFLOPs = (Total FLOPs / sec_per_iter) / 1e12
    tflops = (total_flops /  world_size) / sec_per_iter / 1e12

    if rank == 0:
        print(f"| {attn_type:<10} | {'Causal Varlen':<14} | {world_size:<4} | {local_total_tokens:<12} | {sec_per_iter:<10.4f} | {tflops:<10.2f} |")

    # 清理，防止内存泄漏
    del q, k, v, runtime
    torch.cuda.empty_cache()
    dist.barrier()


if __name__ == "__main__":
    # --- Distributed setup ---
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{LOCAL_RANK}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cp_group_kv = dist.new_group(ranks=list(range(world_size)))
    cp_group_dkv = dist.new_group(ranks=list(range(world_size)))
    cp_groups = [cp_group_kv, cp_group_dkv]

    # --- Benchmark parameters ---
    local_total_tokens = [8192, 16384, 32768, 65536, 131072]
    batch_size = 4
    num_heads = 32
    num_kv_heads = 32 
    head_dim = 128
    num_iter = 50
    forward_only = True

    if rank == 0:
        print(f"--- Comparing CAUSAL VARLEN Attention on {world_size} GPUs ---")
        print(f"Local Tokens: {local_total_tokens}, Total Tokens: {local_total_tokens * world_size}")
        print("-------------------------------------------------------------------------------------")
        print("| Type       | Mode           | CP   | Local Tokens | Time(s)    | TFLOPs     |")
        print("|------------|----------------|------|--------------|------------|------------|")

    for local_total_token in local_total_tokens:
        benchmark_attention("magi", local_total_token, batch_size, num_heads, num_kv_heads, head_dim, num_iter, forward_only, cp_groups)
        benchmark_attention("zigzag", local_total_token, batch_size, num_heads, num_kv_heads, head_dim, num_iter, forward_only, cp_groups)

    if rank == 0:
        print("-------------------------------------------------------------------------------------")

    dist.destroy_process_group()