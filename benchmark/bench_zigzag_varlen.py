import os
import torch
import torch.distributed as dist
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime
import math

from magi_attention.common.ranges import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta._calc_dispatch_meta import _calc_self_attn_areas
from magi_attention.functional.dist_attn import dist_attn_func
from magi_attention.dist_attn_runtime_mgr import init_dist_attn_runtime_mgr
from magi_attention.config import DistAttnConfig

from zigzag_flex_flash_attn import zigzag_flex_flash_attn_varlen_func
from ring_flash_attn import zigzag_ring_flash_attn_varlen_func

def calculate_attn_flops(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: List[AttnMaskType],
    total_seqlen_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> Dict[str, float]:
    """计算注意力机制的理论 FLOPs
    
    对于因果注意力，每个位置 i 只能看到位置 0 到 i，所以：
    - 位置 0 看到 1 个 token
    - 位置 1 看到 2 个 tokens
    - 位置 2 看到 3 个 tokens
    ...
    - 位置 n-1 看到 n 个 tokens
    总计算量是等差数列求和：n*(n+1)/2
    """
    # 计算每个序列的实际注意力计算量
    total_tokens = 0
    total_attn_elements = 0
    
    for i in range(len(q_ranges)):
        q_start, q_end = q_ranges[i].start, q_ranges[i].end
        k_start, k_end = k_ranges[i].start, k_ranges[i].end
        seq_len = q_end - q_start
        
        if attn_mask_type[i] == AttnMaskType.CAUSAL:
            # 因果注意力：等差数列求和 n*(n+1)/2
            attn_elements = (seq_len * (seq_len + 1)) // 2
        else:
            # 全量注意力：n*n
            attn_elements = seq_len * seq_len
            
        total_tokens += seq_len
        total_attn_elements += attn_elements
    
    # 计算 GQA 的实际 FLOPs
    # 1. QK 乘法：每个 token 位置都需要计算 head_dim 维度的点积
    qk_flops = 2 * total_attn_elements * head_dim  # 乘加各算一次
    
    # 2. Softmax: exp + sum + div，每个注意力分数都需要
    softmax_flops = 3 * total_attn_elements
    
    # 3. PV 乘法：每个注意力分数都要乘以对应的 V 向量
    pv_flops = 2 * total_attn_elements * head_dim
    
    # 考虑 GQA：每个 KV head 被 num_heads_q/num_heads_kv 个 Q head 使用
    heads_ratio = num_heads_q / num_heads_kv
    flops_per_kv_head = qk_flops + softmax_flops + pv_flops
    
    # 总 FLOPs = 每个 KV head 的 FLOPs * KV head 数量 * 每个 KV head 服务的 Q head 数量
    flops_fwd = flops_per_kv_head * num_heads_kv
    
    # 反向传播约为前向传播的 2 倍
    # - QK 反向：计算 Q 和 K 的梯度
    # - Softmax 反向：计算 softmax 的梯度
    # - PV 反向：计算 P 和 V 的梯度
    flops_bwd = flops_fwd * 2
    flops_1f1b = flops_fwd + flops_bwd
    
    return {
        "fwd": flops_fwd,
        "bwd": flops_bwd,
        "1f1b": flops_1f1b,
        "details": {
            "total_tokens": total_tokens,
            "total_attn_elements": total_attn_elements,
            "qk_flops": qk_flops * num_heads_kv,
            "softmax_flops": softmax_flops * num_heads_kv,
            "pv_flops": pv_flops * num_heads_kv,
            "flops_per_kv_head": flops_per_kv_head
        }
    }

def calculate_global_causal_varlen_flops(
    world_size: int,
    batch_size: int,
    seqlen: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    forward_only: bool,
) -> Dict[str, float]:
    """计算分布式因果可变长度注意力的理论 FLOPs"""
    # 构建全局序列布局
    local_cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32)
    global_q_ranges_list = []
    global_k_ranges_list = []
    
    for rank in range(world_size):
        rank_offset = rank * batch_size * seqlen
        for i in range(batch_size):
            seq_start = rank_offset + local_cu_seqlens[i].item()
            seq_end = rank_offset + local_cu_seqlens[i+1].item()
            global_q_ranges_list.append([seq_start, seq_end])
            # 因果掩码：只能看到当前及之前的 tokens
            global_k_ranges_list.append([0, seq_end])

    global_q_ranges = AttnRanges.from_ranges(global_q_ranges_list)
    global_k_ranges = AttnRanges.from_ranges(global_k_ranges_list)
    global_batch_size = world_size * batch_size

    # 使用因果掩码计算 FLOPs
    flops_dict = calculate_attn_flops(
        q_ranges=global_q_ranges,
        k_ranges=global_k_ranges,
        attn_mask_type=[AttnMaskType.CAUSAL] * global_batch_size,
        total_seqlen_q=seqlen * batch_size * world_size,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )
    
    # if rank == 0:
    #     details = flops_dict["details"]
    #     print(f"\nComputation Analysis:")
    #     print(f"Total tokens: {details['total_tokens']}")
    #     print(f"Total attention elements: {details['total_attn_elements']}")
    #     print(f"FLOPs per KV head: {details['flops_per_kv_head']/1e9:.2f} GFLOPs")
    #     print(f"\nFLOPs Breakdown (per iteration):")
    #     print(f"QK multiply: {details['qk_flops']/1e12:.2f} TFLOPs")
    #     print(f"Softmax: {details['softmax_flops']/1e12:.2f} TFLOPs")
    #     print(f"PV multiply: {details['pv_flops']/1e12:.2f} TFLOPs")
    #     print(f"Total forward: {flops_dict['fwd']/1e12:.2f} TFLOPs")
    #     if not forward_only:
    #         print(f"Total backward: {flops_dict['bwd']/1e12:.2f} TFLOPs")
    
    return flops_dict

class BenchmarkMetrics:
    """性能指标收集器"""
    def __init__(self):
        self.times = []  # 每次迭代的时间
        self.start_memory = 0  # 开始时的显存使用
        self.peak_memory = 0   # 峰值显存使用
        
    def update_memory(self):
        current_memory = torch.cuda.memory_allocated()
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def add_time(self, time_ms):
        self.times.append(time_ms)
        
    def get_stats(self) -> Dict[str, float]:
        times = np.array(self.times)
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "memory_used": (self.peak_memory - self.start_memory) / 1024**2,  # MB
            "peak_memory": self.peak_memory / 1024**2  # MB
        }

def full_attention_to_varlen_attention(batch_size: int, seqlen: int):
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen,
        dtype=torch.int32, device=torch.cuda.current_device(),
    )
    return cu_seqlens, cu_seqlens

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
    num_iter: int,
    forward_only: bool,
    cp_groups: List[dist.ProcessGroup],
):
    """运行基准测试
    
    Args:
        attn_type: "magi", "zigzag_flex", 或 "zigzag_ring"
        local_total_tokens: 每个 GPU 的 token 数量
        batch_size: 批次大小
        num_heads: 注意力头数
        num_kv_heads: KV 头数
        head_dim: 注意力头维度
        num_iter: 迭代次数
        forward_only: 是否只测试前向传播
        cp_groups: 通信组列表
    """
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 计算序列长度和创建输入张量
    seqlen = local_total_tokens // batch_size
    local_cu_seqlens, _ = full_attention_to_varlen_attention(batch_size, seqlen)
    
    # --- 构建全局元数据 ---
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
    
    # 创建标准格式的输入张量
    q = torch.randn(
        batch_size, seqlen, num_heads, head_dim,
        device=device, dtype=dtype,
        requires_grad=not forward_only
    )
    k = torch.randn(
        batch_size, seqlen, num_kv_heads, head_dim,
        device=device, dtype=dtype,
        requires_grad=not forward_only
    )
    v = torch.randn(
        batch_size, seqlen, num_kv_heads, head_dim,
        device=device, dtype=dtype,
        requires_grad=not forward_only
    )
    
    # 转换为 varlen 格式
    q = q.reshape(-1, num_heads, head_dim)
    k = k.reshape(-1, num_kv_heads, head_dim)
    v = v.reshape(-1, num_kv_heads, head_dim)

    # 初始化性能指标收集器
    metrics = BenchmarkMetrics()
    metrics.start_memory = torch.cuda.memory_allocated()

    # 准备运行时配置
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
    elif attn_type == "zigzag_flex":
        ranges_tensor = torch.stack([local_cu_seqlens[:-1], local_cu_seqlens[1:]], dim=1).to(device)
        q_ranges_tensor = k_ranges_tensor = ranges_tensor
        max_seqlen_q = max_seqlen_k = seqlen
        sm_margin = 0
    elif attn_type == "zigzag_ring":
        softmax_scale = 1.0 / math.sqrt(head_dim)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

    # 预热
    warmup_iters = 10
    for _ in range(warmup_iters):
        if attn_type == "magi":
            _, _ = dist_attn_func(q, k, v, runtime)
        elif attn_type == "zigzag_flex":
            _ = zigzag_flex_flash_attn_varlen_func(
                q, k, v, max_seqlen_q, max_seqlen_k, sm_margin,
                ranges_tensor, q_ranges_tensor, k_ranges_tensor,
                process_group=cp_groups[0], dgrad_process_group=cp_groups[1]
            )
        else:  # zigzag_ring
            _ = zigzag_ring_flash_attn_varlen_func(
                q, k, v, local_cu_seqlens, seqlen,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(-1, -1),
                group=cp_groups[0]
            )

    # 确保预热完成
    torch.cuda.synchronize()
    dist.barrier()

    # 主要测试循环
    for i in range(num_iter):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        if attn_type == "magi":
            _, _ = dist_attn_func(q, k, v, runtime)
        elif attn_type == "zigzag_flex":
            _ = zigzag_flex_flash_attn_varlen_func(
                q, k, v, max_seqlen_q, max_seqlen_k, sm_margin,
                ranges_tensor, q_ranges_tensor, k_ranges_tensor,
                process_group=cp_groups[0], dgrad_process_group=cp_groups[1]
            )
        else:  # zigzag_ring
            _ = zigzag_ring_flash_attn_varlen_func(
                q, k, v, local_cu_seqlens, seqlen,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(-1, -1),
                group=cp_groups[0]
            )
            
        end_event.record()
        torch.cuda.synchronize()
        metrics.add_time(start_event.elapsed_time(end_event))
        metrics.update_memory()

    dist.barrier()
    
    # 计算性能指标
    stats = metrics.get_stats()
    flops_dict = calculate_global_causal_varlen_flops(
        world_size, batch_size, seqlen,
        num_heads, num_kv_heads, head_dim, forward_only
    )
    
    total_flops = flops_dict["fwd"] if forward_only else flops_dict["1f1b"]
    tflops = total_flops / (stats["mean_time"] / 1000.0) / 1e12  # 转换为 TFLOPs

    if rank == 0:
        print(
            f"| {attn_type:<12} | {'Causal Varlen':<14} | {world_size:<4} | "
            f"{local_total_tokens:<12} | {stats['mean_time']/1000:<10.4f} | "
            f"{tflops:<10.2f} | {stats['memory_used']:<8.1f} | "
            f"{stats['std_time']/1000:<8.4f} |"
        )

def main():
    # 设置基准测试参数
    local_total_tokens_list = [8192, 16384, 32768, 65536]
    batch_size = 4
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_iter = 50
    forward_only = True

    # 初始化分布式环境
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        rank = dist.get_rank()
    except KeyError:
        # 单进程模式
        local_rank = 0
        world_size = 1
        rank = 0
        # 创建一个假的进程组
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            world_size=1,
            rank=0
        )

    # 设置设备
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 创建通信组
    cp_groups = [dist.group.WORLD, dist.group.WORLD]

    if rank == 0:
        print(f"\n=== Benchmark Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"PyTorch: {torch.__version__}")
        print(f"\nConfiguration:")
        print(f"- Batch Size: {batch_size}")
        print(f"- Num Heads: {num_heads}")
        print(f"- Head Dim: {head_dim}")
        print(f"- Forward Only: {forward_only}")
        print(f"\n--- Running {'Multi-GPU' if world_size > 1 else 'Single-GPU'} Benchmark ---")
        print(f"World Size: {world_size}")
        print(f"Local Tokens: {local_total_tokens_list}")
        print("-" * 100)
        print("| Type         | Mode           | CP   | Local Tokens | Time(s)    | TFLOPs     | Mem(MB)  | Std(s)  |")
        print("|-------------|----------------|------|--------------|------------|------------|----------|----------|")

    try:
        for local_total_tokens in local_total_tokens_list:
            for attn_type in ["magi", "zigzag_flex", "zigzag_ring"]:
                benchmark_attention(
                    attn_type, local_total_tokens, batch_size,
                    num_heads, num_kv_heads, head_dim,
                    num_iter, forward_only, cp_groups
                )
                if rank == 0:
                    print("|-------------|----------------|------|--------------|------------|------------|----------|----------|")

        if rank == 0:
            print(f"\nBenchmark Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    finally:
        # 清理分布式环境
        dist.destroy_process_group()

if __name__ == "__main__":
    main()