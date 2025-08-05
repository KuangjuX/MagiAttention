import os
import torch
import torch.distributed as dist

# --- 1. Import Implementations ---
# Import the single-GPU reference from Magi/FlashAttention
from magi_attention.functional import flex_flash_attn_func
from zigzag_flex_flash_attn import zigzag_flex_flash_attn_varlen_forward

def _extract_zigzag_data(
    global_tensor: torch.Tensor, 
    global_cu_seqlens: torch.Tensor, 
    rank: int, 
    world_size: int
) -> torch.Tensor:
    """
    实现 Zigzag Attention 所需的特定“头-尾”数据分布。
    对于全局批次中的每个序列，它被分成 (2 * world_size) 个块。
    Rank `r` 获取块 `r` (头部) 和块 `-(r+1)` (尾部)。
    """
    local_values = []
    cu_seqlens_list = global_cu_seqlens.cpu().tolist()

    for i in range(len(cu_seqlens_list) - 1):
        start, end = cu_seqlens_list[i], cu_seqlens_list[i + 1]
        sequence_tensor = global_tensor[start:end]
        
        if sequence_tensor.shape[0] % (world_size * 2) != 0:
            raise ValueError(
                f"Sequence length {sequence_tensor.shape[0]} must be divisible by "
                f"2 * world_size ({2 * world_size}) for zigzag distribution."
            )
            
        chunks = sequence_tensor.chunk(world_size * 2, dim=0)
        
        head_chunk = chunks[rank]
        tail_chunk = chunks[-rank - 1]
        
        local_values.append(torch.cat([head_chunk, tail_chunk]))
        
    return torch.cat(local_values, dim=0).contiguous()


def analyze_and_print_differences(tensor_a, tensor_b, name, rtol=1e-2, atol=1e-2):
    """比较两个张量并打印详细分析。"""
    print(f"\n--- Analyzing differences for: {name} ---")
    tensor_b = tensor_b.to(tensor_a.device, dtype=tensor_a.dtype)
    
    are_close = torch.allclose(tensor_a, tensor_b, rtol=rtol, atol=atol)
    print(f"Are tensors close (allclose): {are_close}")
    
    abs_diff = torch.abs(tensor_a - tensor_b)
    max_abs_diff = torch.max(abs_diff)
    
    rel_diff = abs_diff / (torch.abs(tensor_b) + 1e-6)
    max_rel_diff = torch.max(rel_diff)

    print(f"  Max absolute difference: {max_abs_diff.item():.6f}")
    print(f"  Max relative difference: {max_rel_diff.item():.6f}")

    if not are_close:
        max_idx = torch.argmax(abs_diff)
        print(f"  Location of max diff: {torch.unravel_index(max_idx, tensor_a.shape)}")
        print(f"  Value in Your Tensor at max diff: {tensor_a.flatten()[max_idx].item():.6f}")
        print(f"  Value in Reference Tensor at max diff: {tensor_b.flatten()[max_idx].item():.6f}")
    
    print("-" * (len(name) + 30))


# =============================================================================
# 3. 核心验证函数
# =============================================================================

def check_correctness(
    num_sequences: int,
    seqlen: int,
    num_heads: int,
    head_dim: int,
    process_group: dist.ProcessGroup,
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    """
    使用正确的“头-尾”数据分布，对照单卡参考实现来验证分布式 Zigzag Attention。
    """
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Zigzag 要求序列长度能被 2 * world_size 整除
    if seqlen % (2 * world_size) != 0:
        original_seqlen = seqlen
        seqlen = (seqlen // (2 * world_size)) * (2 * world_size)
        if seqlen == 0:
            raise ValueError(f"seqlen {original_seqlen} is too small for world_size {world_size}.")
        if rank == 0:
            print(f"INFO: Adjusting seqlen from {original_seqlen} to {seqlen} to be divisible by {2 * world_size}.")

    total_tokens = num_sequences * seqlen

    if rank == 0:
        print("-" * 80)
        print(f"Verifying for num_sequences={num_sequences}, seqlen={seqlen}, total_tokens={total_tokens}, world_size={world_size}")
        print("-" * 80)

    # --- A. 创建全局数据并在 Rank 0 上运行参考实现 ---
    global_q, global_k, global_v, out_ref = None, None, None, None
    if rank == 0:
        print("Step 1: Creating global data and running single-GPU reference on Rank 0...")
        torch.manual_seed(42)
        global_q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
        global_k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
        global_v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

        cu_seqlens_global = torch.arange(0, total_tokens + 1, step=seqlen, dtype=torch.int32, device=device)
        global_ranges_tensor = torch.stack([cu_seqlens_global[:-1], cu_seqlens_global[1:]], dim=1)
        attn_type_map_global = torch.ones(num_sequences, dtype=torch.int32, device=device)

        out_ref, _ = flex_flash_attn_func(
            q=global_q, k=global_k, v=global_v,
            q_ranges=global_ranges_tensor, k_ranges=global_ranges_tensor,
            max_seqlen_q=seqlen, max_seqlen_k=seqlen,
            attn_type_map=attn_type_map_global
        )
        print("Reference calculation complete.")

    # --- B. 使用 Zigzag 方式分发数据 ---
    if rank == 0: print("Step 2: Distributing data to all ranks using Zigzag layout...")
    
    # 1. 广播全局数据到所有 rank
    global_q_bcast = torch.empty(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    global_k_bcast = torch.empty(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    global_v_bcast = torch.empty(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    
    dist.broadcast(global_q if rank == 0 else global_q_bcast, src=0)
    dist.broadcast(global_k if rank == 0 else global_k_bcast, src=0)
    dist.broadcast(global_v if rank == 0 else global_v_bcast, src=0)
    if rank != 0: # Rank 0 已经有数据了
        global_q, global_k, global_v = global_q_bcast, global_k_bcast, global_v_bcast

    # 2. 每个 rank 从广播后的全局张量中提取自己的本地数据
    global_cu_seqlens_tensor = torch.arange(0, total_tokens + 1, step=seqlen, dtype=torch.int32, device=device)
    
    local_q = _extract_zigzag_data(global_q, global_cu_seqlens_tensor, rank, world_size)
    local_k = _extract_zigzag_data(global_k, global_cu_seqlens_tensor, rank, world_size)
    local_v = _extract_zigzag_data(global_v, global_cu_seqlens_tensor, rank, world_size)
    
    dist.barrier()
    if rank == 0: print("Data distribution complete.")

    # --- C. 运行分布式 Zigzag 实现 ---
    if rank == 0: print("Step 3: Running distributed Zigzag implementation...")

    local_seqlen = seqlen // world_size
    local_total_tokens = num_sequences * local_seqlen
    local_cu_seqlens = torch.arange(0, local_total_tokens + 1, step=local_seqlen, dtype=torch.int32, device=device)

    # 创建 zigzag 函数所需的 ranges 张量 (对半切分)
    ranges, q_ranges, k_ranges = [], [], []
    for start, end in zip(local_cu_seqlens[:-1], local_cu_seqlens[1:]):
        ranges.append([start.item(), end.item()])
        half = start + (end - start) // 2
        q_ranges.append([half.item(), end.item()])
        k_ranges.append([start.item(), half.item()])
    
    ranges_tensor = torch.tensor(ranges, dtype=torch.int32, device=device)
    q_ranges_tensor = torch.tensor(q_ranges, dtype=torch.int32, device=device)
    k_ranges_tensor = torch.tensor(k_ranges, dtype=torch.int32, device=device)

    local_out_zigzag, _ = zigzag_flex_flash_attn_varlen_forward(
        process_group=process_group,
        q=local_q, k=local_k, v=local_v,
        max_seqlen_q=local_seqlen, max_seqlen_k=local_seqlen,
        sm_margin=0,
        ranges_tensor=ranges_tensor,
        q_ranges_tensor=q_ranges_tensor,
        k_ranges_tensor=k_ranges_tensor,
    )
    
    dist.barrier()
    if rank == 0: print("Zigzag execution complete.")

    # --- D. 本地比较结果 ---
    if rank == 0: print("Step 4: Extracting local reference and comparing results on each rank...")

    # 1. 广播参考输出
    out_ref_bcast = torch.empty(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    dist.broadcast(out_ref if rank == 0 else out_ref_bcast, src=0)
    if rank != 0: out_ref = out_ref_bcast

    # 2. 每个 rank 提取自己的参考输出部分
    local_out_ref = _extract_zigzag_data(out_ref, global_cu_seqlens_tensor, rank, world_size)

    # 3. 在每个 rank 上进行比较
    are_outputs_close = torch.allclose(local_out_zigzag, local_out_ref, rtol=rtol, atol=atol)
    
    if not are_outputs_close:
        print(f"\n!!!!!!!!!! [Rank {rank}] MISMATCH FOUND !!!!!!!!!!")
        analyze_and_print_differences(local_out_zigzag, local_out_ref, f"Output on Rank {rank}", rtol, atol)

    # 4. 将结果汇总到 rank 0
    result_tensor = torch.tensor([1.0 if are_outputs_close else 0.0], device=device)
    dist.all_reduce(result_tensor, op=dist.ReduceOp.MIN)

    if rank == 0:
        print("\n" + "="*40 + " FINAL VERIFICATION " + "="*40)
        if result_tensor.item() == 1.0:
            print("✅ SUCCESS: All ranks reported matching outputs.")
        else:
            print("❌ FAILURE: At least one rank reported a mismatch.")
        print("="*102)



def main():
    num_sequences = 8
    seqlen = 4096
    num_heads = 32
    head_dim = 128

    # 初始化分布式环境
    if "RANK" not in os.environ:
        print("This script must be run with torchrun.")
        exit(1)
        
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"Starting test with {world_size} GPUs.")
        print("\n" + "*"*20 + " TEST PARAMETERS " + "*"*20)
        print(f"  Number of Sequences: {num_sequences}")
        print(f"  Sequence Length: {seqlen}")
        print(f"  Number of Heads: {num_heads}")
        print(f"  Head Dimension: {head_dim}")
        print("*"*59 + "\n")

    try:
        check_correctness(
            num_sequences=num_sequences,
            seqlen=seqlen,
            num_heads=num_heads,
            head_dim=head_dim,
            process_group=dist.group.WORLD,
            rtol=1e-2,
            atol=1e-2,
        )
    except Exception as e:
        print(f"Rank {rank} caught an exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            print("\nTest finished and distributed environment cleaned up.")

if __name__ == "__main__":
    main()






