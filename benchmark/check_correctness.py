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
        # Print first 128 elements of both tensors
        num_elements = min(128, tensor_a.numel())
        print(f"\nFirst {num_elements} elements of tensor A:")
        print(tensor_a.flatten()[:num_elements])
        print(f"\nFirst {num_elements} elements of tensor B:")
        print(tensor_b.flatten()[:num_elements])
        
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

    if rank == 0:
        print(f"local_ranges_tensor: {local_ranges_tensor}")

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
    # full_out_zigzag = torch.empty_like(global_q) if rank == 0 else torch.empty(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    # full_lse_zigzag = torch.empty_like(lse_ref) if rank == 0 else torch.empty(num_heads, total_tokens, device=device, dtype=torch.float32)

    # Gather all local outputs into a single tensor on all ranks
    # dist.all_gather_into_tensor(full_out_zigzag, local_out_zigzag.contiguous())
    # dist.all_gather_into_tensor(full_lse_zigzag, local_lse_zigzag.contiguous())
    if rank == 0:
        out = [torch.empty_like(local_out_zigzag) for _ in range(world_size)]
        lse = [torch.empty_like(local_lse_zigzag) for _ in range(world_size)]
    else:
        out = None
        lse = None

    dist.gather(local_out_zigzag.contiguous(), out, dst=0)
    dist.gather(local_lse_zigzag.contiguous(), lse, dst=0)
    dist.barrier()


    if rank == 0:
        full_out_zigzag = torch.cat(out, dim=0)
        full_lse_zigzag = torch.cat(lse, dim=0)
        print("\n" + "="*40 + " FINAL COMPARISON (on Rank 0) " + "="*40)
        analyze_and_print_differences(full_out_zigzag, out_ref, "Output (Zigzag vs. Reference)", rtol, atol)
        # analyze_and_print_differences(full_lse_zigzag, lse_ref, "LSE (Zigzag vs. Reference)", rtol, atol)
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
    total_tokens = 128 * world_size
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









