# import os
# import sys
# import torch
# import torch.distributed as dist
# from typing import List

# # --- 导入你的实现 ---
# # 假设你的 zigzag 函数在 zigzag_flex_flash_attn.py 文件中
# from zigzag_flex_flash_attn import zigzag_flex_flash_attn_varlen_func

# # --- 导入参考实现和待测库 ---
# # 导入 Magi 的新 API
# from magi_attention.api import magi_attn_varlen_key, calc_attn
# # 导入 FlashAttn 作为黄金标准
# # from flash_attn.flash_attn_interface import flash_attn_varlen_func
# from magi_attention.functional import flex_flash_attn_func

# # --- 导入辅助函数 ---
# # 假设这些函数在 utils.py 文件中，或者你可以直接把它们粘贴到这个文件里
# # from utils import analyze_and_print_differences, full_attention_to_varlen_attention

# # --- 为了让脚本自包含，我们在这里定义这些辅助函数 ---
# def analyze_and_print_differences(tensor_a, tensor_b, name, rtol=1e-2, atol=1e-2):
#     """比较两个张量并打印分析结果"""
#     print(f"\n--- Analyzing differences for: {name} ---")
#     are_close = torch.allclose(tensor_a, tensor_b, rtol=rtol, atol=atol)
#     print(f"Are tensors close (allclose): {are_close}")
#     if not are_close:
#         abs_diff = torch.abs(tensor_a - tensor_b)
#         max_abs_diff = torch.max(abs_diff)
#         max_rel_diff = torch.max(abs_diff / torch.abs(tensor_b + 1e-6)) # 避免除以零
#         print(f"  Max absolute difference: {max_abs_diff.item():.6f}")
#         print(f"  Max relative difference: {max_rel_diff.item():.6f}")
        
#         # 找到最大差异的位置
#         max_idx = torch.argmax(abs_diff)
#         print(f"  Location of max diff: {torch.unravel_index(max_idx, tensor_a.shape)}")
#         print(f"  Value in Tensor A at max diff: {tensor_a.flatten()[max_idx].item():.6f}")
#         print(f"  Value in Tensor B at max diff: {tensor_b.flatten()[max_idx].item():.6f}")
#     print("-" * (len(name) + 30))

# def full_attention_to_varlen_attention(batch_size, seqlen):
#     """为 varlen attention 创建 cu_seqlens"""
#     cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32)
#     max_seqlen = seqlen
#     return cu_seqlens, max_seqlen

# def compute_pad_size(seqlen, cp_size, chunk_size):
#     """计算 magi 需要的 pad_size"""
#     multiple = chunk_size * cp_size
#     if seqlen % multiple == 0:
#         return 0
#     return multiple - (seqlen % multiple)

# # --- 核心验证逻辑 ---
# def verify_algorithms(
#     local_total_tokens: int,
#     batch_size: int,
#     num_heads: int,
#     head_dim: int,
#     cp_groups: List[dist.ProcessGroup],
#     rtol: float = 1e-2,
#     atol: float = 1e-2,
# ):
#     dtype = torch.bfloat16
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     device = torch.device(f"cuda:{rank}")
#     torch.cuda.set_device(device)

#     if rank == 0:
#         print("-" * 80)
#         print(f"Verifying for local_total_tokens={local_total_tokens}, world_size={world_size}")

#     # --- 1. 计算 Padding 并创建 Padded 全局数据 ---
#     torch.manual_seed(42)
#     torch.cuda.manual_seed_all(42)
    
#     global_total_tokens = local_total_tokens * world_size
    
#     # 计算 Magi 要求的总长度
#     chunk_size = 1536
#     multiple = world_size * chunk_size
#     padded_global_total_tokens = ((global_total_tokens + multiple - 1) // multiple) * multiple
#     padding_amount = padded_global_total_tokens - global_total_tokens
    
#     if rank == 0:
#         print(f"Original global tokens: {global_total_tokens}")
#         print(f"Magi requires multiple of: {multiple}")
#         print(f"Padded global tokens: {padded_global_total_tokens}")
#         print(f"Padding amount: {padding_amount}")

#     # 创建 Padded 全局张量
#     if rank == 0:
#         # 创建原始数据
#         original_q = torch.randn(global_total_tokens, num_heads, head_dim, device=device, dtype=dtype)
#         original_k = torch.randn(global_total_tokens, num_heads, head_dim, device=device, dtype=dtype)
#         original_v = torch.randn(global_total_tokens, num_heads, head_dim, device=device, dtype=dtype)
        
#         # 创建 Padded 容器并复制数据
#         global_q_padded = torch.zeros(padded_global_total_tokens, num_heads, head_dim, device=device, dtype=dtype)
#         global_k_padded = torch.zeros(padded_global_total_tokens, num_heads, head_dim, device=device, dtype=dtype)
#         global_v_padded = torch.zeros(padded_global_total_tokens, num_heads, head_dim, device=device, dtype=dtype)

#         global_q_padded[:global_total_tokens] = original_q
#         global_k_padded[:global_total_tokens] = original_k
#         global_v_padded[:global_total_tokens] = original_v
#     else:
#         # 在其他 Rank 上创建同样大小的空 Tensor
#         global_q_padded = torch.empty(padded_global_total_tokens, num_heads, head_dim, device=device, dtype=dtype)
#         global_k_padded = torch.empty(padded_global_total_tokens, num_heads, head_dim, device=device, dtype=dtype)
#         global_v_padded = torch.empty(padded_global_total_tokens, num_heads, head_dim, device=device, dtype=dtype)

#     # 广播 Padded 数据
#     dist.broadcast(global_q_padded, src=0)
#     dist.broadcast(global_k_padded, src=0)
#     dist.broadcast(global_v_padded, src=0)

#     # 创建 cu_seqlens
#     seqlen = local_total_tokens // batch_size
#     # FlashAttn 使用原始的 cu_seqlens
#     ref_cu_seqlens = torch.arange(0, global_total_tokens + 1, step=seqlen, dtype=torch.int32, device=device)
#     # Magi 和 Zigzag 使用 Padded cu_seqlens
#     magi_cu_seqlens = ref_cu_seqlens.clone()
#     magi_cu_seqlens[-1] += padding_amount # 将所有 padding 加到最后一个序列上

#     # --- 2. 运行 FlashAttn (黄金标准) ---
#     if rank == 0: print("Running FlashAttn (Reference)...")
#     # FlashAttn 在原始数据上运行，因为它不需要 padding
#     out_ref = flash_attn_varlen_func(
#         global_q_padded[:global_total_tokens], 
#         global_k_padded[:global_total_tokens], 
#         global_v_padded[:global_total_tokens], 
#         ref_cu_seqlens, ref_cu_seqlens, seqlen, seqlen, 0.0, causal=True
#     )
#     dist.barrier()
#     if rank == 0: print("FlashAttn execution complete.")

#     # --- 3. 准备本地数据 (从 Padded 全局数据中切分) ---
#     local_q_padded = global_q_padded.chunk(world_size, dim=0)[rank].contiguous()
#     local_k_padded = global_k_padded.chunk(world_size, dim=0)[rank].contiguous()
#     local_v_padded = global_v_padded.chunk(world_size, dim=0)[rank].contiguous()

#     # --- 4. 运行 Magi ---
#     if rank == 0: print("Running Magi...")
#     # 因为数据已经 padded，所以 pad_size 参数现在为 0
#     # 我们传入 padded cu_seqlens
#     _, dist_attn_runtime_key = magi_attn_varlen_key(
#         x=local_q_padded,
#         cu_seqlens_q=magi_cu_seqlens,
#         cu_seqlens_k=magi_cu_seqlens,
#         head_dim=head_dim,
#         pad_size=0, # 数据已手动 pad，这里传 0
#         cp_group=cp_groups[0],
#         causal=True,
#     )
    
#     out_magi_padded, _ = calc_attn(local_q_padded, local_k_padded, local_v_padded, dist_attn_runtime_key)
    
#     # 裁剪 Magi 的输出，只保留有效部分
#     local_out_magi = out_magi_padded[:local_q_padded.shape[0] - (padding_amount // world_size if rank == world_size - 1 else 0)]
    
#     dist.barrier()
#     if rank == 0: print("Magi execution complete.")

#     # --- 5. 运行 Zigzag ---
#     if rank == 0: print("Running Zigzag...")
#     # Zigzag 也应该在 padded 数据上运行，以保持公平比较
#     # 但为了简单起见，我们先让它在原始数据上运行，如果它能处理非均匀分片的话
#     local_q = global_q_padded[:global_total_tokens].chunk(world_size, dim=0)[rank].contiguous()
#     local_k = global_k_padded[:global_total_tokens].chunk(world_size, dim=0)[rank].contiguous()
#     local_v = global_v_padded[:global_total_tokens].chunk(world_size, dim=0)[rank].contiguous()

#     local_cu_seqlens, _ = full_attention_to_varlen_attention(batch_size, seqlen)
#     ranges_tensor = torch.stack([local_cu_seqlens[:-1], local_cu_seqlens[1:]], dim=1).to(device)
#     out_zigzag = zigzag_flex_flash_attn_varlen_func(
#         local_q, local_k, local_v, seqlen, seqlen, 0,
#         ranges_tensor, ranges_tensor, ranges_tensor,
#         process_group=cp_groups[0], dgrad_process_group=cp_groups[1]
#     )
#     dist.barrier()
#     if rank == 0: print("Zigzag execution complete.")

#     # --- 6. 收集结果并比较 ---
#     all_outputs = {}
#     # for name, tensor in [("magi", local_out_magi), ("zigzag", out_zigzag)]:
#     for name, tensor in [("zigzag", out_zigzag)]:
#         # 使用 all_gather_object 来处理可能不同大小的张量
#         gathered_tensors_on_diff_devices = [torch.empty(0, device="cpu", dtype=dtype) for _ in range(world_size)]
#         dist.all_gather_object(gathered_tensors_on_diff_devices, tensor.cpu()) # 将张量移到CPU再收集，避免跨GPU通信问题

#         if rank == 0:
#             # 在 rank 0 上，将所有收集到的张量移到当前设备 (cuda:0) 并拼接
#             tensors_on_current_device = [t.to(device) for t in gathered_tensors_on_diff_devices]
#             all_outputs[name] = torch.cat(tensors_on_current_device, dim=0)
    
#     dist.barrier()

#     if rank == 0:
#         print("\n" + "="*40 + " FINAL COMPARISON " + "="*40)
#         # 确保 out_ref 也在正确的设备上
#         out_ref = out_ref.to(device)
#         # analyze_and_print_differences(all_outputs["magi"], out_ref, "Magi vs FlashAttn (Reference)", rtol, atol)
#         analyze_and_print_differences(all_outputs["zigzag"], out_ref, "Zigzag vs FlashAttn (Reference)", rtol, atol)
#         print("="*98)




# # --- Main 执行入口 ---
# if __name__ == "__main__":
#     # 初始化分布式环境
#     dist.init_process_group("nccl")
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()

#     # 定义测试参数
#     local_total_tokens = 4096
#     batch_size = 4
#     num_heads = 8
#     head_dim = 64
    
#     # 创建你的算法需要的通信组
#     # 你的 zigzag 实现需要两个独立的通信组
#     all_ranks = list(range(world_size))
#     cp_group1 = dist.new_group(all_ranks)
#     cp_group2 = dist.new_group(all_ranks)
#     cp_groups = [cp_group1, cp_group2]

#     # 调用验证函数
#     verify_algorithms(
#         local_total_tokens=local_total_tokens,
#         batch_size=batch_size,
#         num_heads=num_heads,
#         head_dim=head_dim,
#         cp_groups=cp_groups,
#         rtol=1e-2,
#         atol=1e-2,
#     )

#     # 清理分布式环境
#     dist.destroy_process_group()

import os
import torch
import torch.distributed as dist

# --- 1. Import Implementations ---
# Import the single-GPU reference from Magi/FlashAttention
from magi_attention.functional import flex_flash_attn_func
from zigzag_flex_flash_attn import zigzag_flex_flash_attn_varlen_forward

# Import your distributed implementation from the file you just created
# --- 2. Helper Functions ---
def analyze_and_print_differences(tensor_a, tensor_b, name, rtol=1e-2, atol=1e-2):
    """Compares two tensors and prints a detailed analysis."""
    print(f"\n--- Analyzing differences for: {name} ---")
    # Ensure tensors are on the same device for comparison
    tensor_b = tensor_b.to(tensor_a.device)
    
    try:
        are_close = torch.allclose(tensor_a, tensor_b, rtol=rtol, atol=atol)
        print(f"Are tensors close (allclose): {are_close}")
        
        abs_diff = torch.abs(tensor_a - tensor_b)
        max_abs_diff = torch.max(abs_diff)
        
        # For relative difference, use the reference tensor as the denominator
        rel_diff = abs_diff / (torch.abs(tensor_b) + 1e-6)
        max_rel_diff = torch.max(rel_diff)

        print(f"  Max absolute difference: {max_abs_diff.item():.6f}")
        print(f"  Max relative difference: {max_rel_diff.item():.6f}")

        if not are_close:
            # Find and print location of max difference for debugging
            max_idx = torch.argmax(abs_diff)
            print(f"  Location of max diff: {torch.unravel_index(max_idx, tensor_a.shape)}")
            print(f"  Value in Your Tensor at max diff: {tensor_a.flatten()[max_idx].item():.6f}")
            print(f"  Value in Reference Tensor at max diff: {tensor_b.flatten()[max_idx].item():.6f}")

    except Exception as e:
        print(f"An error occurred during comparison: {e}")
        print(f"Your tensor shape: {tensor_a.shape}, dtype: {tensor_a.dtype}")
        print(f"Reference tensor shape: {tensor_b.shape}, dtype: {tensor_b.dtype}")
    
    print("-" * (len(name) + 30))

def check_correctness(
    total_tokens: int,
    num_sequences: int,
    num_heads: int,
    head_dim: int,
    process_group: dist.ProcessGroup,
    rtol: float = 1e-2,
    atol: float = 1e-2,

):
    """
    Verifies the distributed zigzag attention against a single-GPU reference.
    """
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Ensure parameters are divisible for simplicity
    assert total_tokens % num_sequences == 0, "total_tokens must be divisible by num_sequences"
    assert total_tokens % world_size == 0, "total_tokens must be divisible by world_size"
    assert num_sequences % world_size == 0, "num_sequences must be divisible by world_size"

    seqlen = total_tokens // num_sequences

    if rank == 0:
        print("-" * 80)
        print(f"Verifying for total_tokens={total_tokens}, num_sequences={num_sequences}, seqlen={seqlen}, world_size={world_size}")
        print("-" * 80)

    # --- A. Create Global Data and Run Single-GPU Reference (on Rank 0) ---
    global_q, global_k, global_v = None, None, None
    out_ref, lse_ref = None, None

    if rank == 0:
        print("Step 1: Creating global data and running single-GPU reference on Rank 0...")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        global_q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
        global_k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
        global_v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

        # Create ranges tensor for the entire global data
        cu_seqlens_global = torch.arange(0, total_tokens + 1, step=seqlen, dtype=torch.int32, device=device)
        global_ranges_tensor = torch.stack([cu_seqlens_global[:-1], cu_seqlens_global[1:]], dim=1)
        
        # For causal attention, attn_type_map is a tensor of 1s
        attn_type_map_global = torch.ones(num_sequences, dtype=torch.int32, device=device)

        # Run the reference implementation
        out_ref, lse_ref = flex_flash_attn_func(
            q=global_q,
            k=global_k,
            v=global_v,
            q_ranges=global_ranges_tensor,
            k_ranges=global_ranges_tensor,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            attn_type_map=attn_type_map_global
        )
        print("Reference calculation complete.")

    # --- B. Distribute Data to All Ranks ---
    if rank == 0: print("Step 2: Distributing data to all ranks...")

    local_total_tokens = total_tokens // world_size

    # Create local tensors to receive data
    local_q = torch.empty(local_total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    local_k = torch.empty(local_total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    local_v = torch.empty(local_total_tokens, num_heads, head_dim, device=device, dtype=dtype)

    # Scatter the data from rank 0
    dist.scatter(local_q, list(global_q.chunk(world_size, dim=0)) if rank == 0 else None, src=0)
    dist.scatter(local_k, list(global_k.chunk(world_size, dim=0)) if rank == 0 else None, src=0)
    dist.scatter(local_v, list(global_v.chunk(world_size, dim=0)) if rank == 0 else None, src=0)
    
    dist.barrier()
    if rank == 0: print("Data distribution complete.")

    # --- C. Run Distributed Zigzag Implementation ---
    if rank == 0: print("Step 3: Running distributed Zigzag implementation...")

    num_sequences_per_rank = num_sequences // world_size
    cu_seqlens_local = torch.arange(0, local_total_tokens + 1, step=seqlen, dtype=torch.int32, device=device)
    local_ranges_tensor = torch.stack([cu_seqlens_local[:-1], cu_seqlens_local[1:]], dim=1)

    # Your function expects q_ranges_tensor and k_ranges_tensor for the zigzag logic
    # Based on your implementation, these seem to be subsets of the local ranges
    # For a simple causal case, they are likely the same as the main ranges tensor
    # NOTE: Adjust this if your zigzag implementation requires more complex range slicing
    local_q_ranges_tensor = local_ranges_tensor 
    local_k_ranges_tensor = local_ranges_tensor

    local_out_zigzag, local_lse_zigzag = zigzag_flex_flash_attn_varlen_forward(
        process_group=process_group,
        q=local_q,
        k=local_k,
        v=local_v,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        sm_margin=0,
        ranges_tensor=local_ranges_tensor,
        q_ranges_tensor=local_q_ranges_tensor,
        k_ranges_tensor=local_k_ranges_tensor,
    )
    
    dist.barrier()
    if rank == 0: print("Zigzag execution complete.")

    # --- D. Gather Results and Compare ---
    if rank == 0: print("Step 4: Gathering results and comparing...")

    # Prepare tensors to receive the gathered data on all ranks
    full_out_zigzag = torch.empty_like(global_q) if rank == 0 else torch.empty(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    full_lse_zigzag = torch.empty_like(lse_ref) if rank == 0 else torch.empty(num_heads, total_tokens, device=device, dtype=torch.float32)

    # Gather all local outputs into a single tensor on all ranks
    dist.all_gather_into_tensor(full_out_zigzag, local_out_zigzag.contiguous())
    dist.all_gather_into_tensor(full_lse_zigzag, local_lse_zigzag.contiguous())

    if rank == 0:
        print("\n" + "="*40 + " FINAL COMPARISON (on Rank 0) " + "="*40)
        analyze_and_print_differences(full_out_zigzag, out_ref, "Output (Zigzag vs. Reference)", rtol, atol)
        analyze_and_print_differences(full_lse_zigzag, lse_ref, "LSE (Zigzag vs. Reference)", rtol, atol)
        print("="*104)

# --- 4. Main Execution Block ---
if __name__ == "__main__":

    # Initialize distributed environment
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create a single process group for the algorithm
    process_group = dist.new_group(ranks=list(range(world_size)))

    # --- Define Test Parameters ---
    # Use smaller values for quick tests, larger for more robust checks
    # Ensure total_tokens and num_sequences are divisible by world_size
    total_tokens = 4096 * world_size
    num_sequences = 16 * world_size
    num_heads = 8
    head_dim = 64

    # Call the verification function
    check_correctness(
        total_tokens=total_tokens,
        num_sequences=num_sequences,
        num_heads=num_heads,
        head_dim=head_dim,
        process_group=process_group,
        rtol=1e-2, # bfloat16 requires higher tolerance
        atol=1e-2,
    )

    # Clean up
    dist.destroy_process_group()









