# Adapted from https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/zigzag_ring_flash_attn.py

from typing import Dict

import torch
from magi_attention.functional.flex_flash_attn import (
    _flex_flash_attn_backward,
    _flex_flash_attn_forward,
)

from utils import RingComm, update_out_and_lse

# from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward

# 0-full 1-causal
_ATTN_TYPE_TO_MASK: Dict[int, torch.Tensor] = {}


def _get_attn_type(
    attn_type: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    global _ATTN_TYPE_TO_MASK
    attn_type_map = _ATTN_TYPE_TO_MASK.get(attn_type, None)
    if attn_type_map is None or batch_size > attn_type_map.shape[0]:
        attn_type_map = torch.full(
            (batch_size,), fill_value=attn_type, dtype=torch.int32, device=device
        )
        _ATTN_TYPE_TO_MASK[attn_type] = attn_type_map
    return attn_type_map[:batch_size]


# pylint: disable=too-many-locals
def zigzag_flex_flash_attn_varlen_forward(
    process_group,  # 分布式进程组
    q: torch.Tensor,  # Query 张量
    k: torch.Tensor,  # Key 张量
    v: torch.Tensor,  # Value 张量
    max_seqlen_q: int,  # Q 的最大序列长度
    max_seqlen_k: int,  # K 的最大序列长度
    sm_margin: int,  # softmax 的边际值
    ranges_tensor: torch.Tensor,  # 完整的序列范围张量
    q_ranges_tensor: torch.Tensor,  # Q 的序列范围张量
    k_ranges_tensor: torch.Tensor,  # K 的序列范围张量
    *args,
    **kwargs,
):
    """
    ZigZag Flex Flash Attention 的前向传播函数，支持可变长度序列。
    
    实现了基于 Ring All-Reduce 的分布式注意力计算，采用 ZigZag 模式进行数据划分和计算。
    每个 GPU 负责部分序列的计算，通过环形通信交换数据完成完整的注意力计算。
    """
    # 确保 k, v 是连续的内存布局
    k, v = k.contiguous(), v.contiguous()
    # 创建用于通信的缓冲区
    k_buffer, v_buffer = torch.empty_like(k), torch.empty_like(v)
    # 初始化环形通信器
    comm = RingComm(process_group)

    # 初始化输出和 LSE (log-sum-exp，用于数值稳定性)
    out = None
    lse = None
    next_k, next_v = None, None

    # 计算 attention scale factor
    softmax_scale = q.shape[-1] ** (-0.5)
    softcap = 0.0  # softmax 上限值

    # 获取当前进程的 rank
    comm = RingComm(process_group)
    rank = comm.rank

    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_type: int,  # 注意力类型：0 表示部分计算，1 表示完整计算
        q_ranges_tensor: torch.Tensor,
        k_ranges_tensor: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ):  
        """
        执行单步的 Flash Attention 计算
        
        参数:
        - attn_type: 决定使用完整序列还是部分序列
        - q/k_ranges_tensor: 定义序列的范围
        - max_seqlen_q/k: 最大序列长度
        """
        # 根据 attn_type 获取注意力类型映射
        attn_type_map = _get_attn_type(
            attn_type, q_ranges_tensor.shape[0], q_ranges_tensor.device
        )
        
        # 调用底层的 Flash Attention 实现
        out, softmax_lse = _flex_flash_attn_forward(
            q, k, v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_type_map=attn_type_map,
            softmax_scale=softmax_scale,
            softcap=softcap,
            deterministic=False,
            sm_margin=sm_margin,
            return_dtype=None,
            # 如果 Q 长度大于等于 K 长度，禁用原子规约
            disable_fwd_atomic_reduction=max_seqlen_q >= max_seqlen_k,
        )
        return out, softmax_lse

    # 执行 world_size 步的环形通信和计算
    for step in range(comm.world_size):
        # 如果不是最后一步，执行通信
        if step + 1 != comm.world_size:
            # 与下一个 rank 交换 k, v
            next_k = comm.send_recv(k, k_buffer)
            next_v = comm.send_recv(v, v_buffer)
            comm.commit()

        # 第一步：计算本地完整的注意力
        if step == 0:
            block_out, block_lse = forward(
                q, k, v,
                attn_type=1,  # 完整计算
                q_ranges_tensor=ranges_tensor,  # 使用完整范围
                k_ranges_tensor=ranges_tensor,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            # 更新输出和 LSE
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            
        # 当前 rank 大于等于 step：计算前半部分 Q 与完整 K
        elif step <= comm.rank:
            block_out, block_lse = forward(
                q, k, v,
                attn_type=0,  # 部分计算
                q_ranges_tensor=ranges_tensor,  # 使用完整的 Q
                k_ranges_tensor=k_ranges_tensor,  # 使用前半部分 K
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k // 2,  # K 只用一半长度
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            
        # 当前 rank 小于 step：计算后半部分 Q 与完整 K
        else:
            block_out, block_lse = forward(
                q, k, v,
                attn_type=0,  # 部分计算
                q_ranges_tensor=q_ranges_tensor,  # 使用后半部分 Q
                k_ranges_tensor=ranges_tensor,  # 使用完整的 K
                max_seqlen_q=max_seqlen_q // 2,  # Q 只用一半长度
                max_seqlen_k=max_seqlen_k,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        # 如果不是最后一步，等待通信完成并更新 k, v
        if step + 1 != comm.world_size:
            comm.wait()  # 等待通信完成
            # 交换缓冲区
            k_buffer, v_buffer = k, v
            k, v = next_k, next_v
            
    # 转换输出格式
    out = out.to(q.dtype)  # 确保输出类型与输入一致
    lse = lse.squeeze(dim=-1).transpose(0, 1)  # 调整 LSE 的维度
    return out, lse


# pylint: disable=unsupported-assignment-operation,too-many-statements
def zigzag_flex_flash_attn_varlen_backward(
    process_group,
    dgrad_process_group,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_margin: int,
    ranges_tensor: torch.Tensor,
    q_ranges_tensor: torch.Tensor,
    k_ranges_tensor: torch.Tensor,
    *args,
    **kwargs,
):
    k, v = k.contiguous(), v.contiguous()
    k_buffer, v_buffer = torch.empty_like(k), torch.empty_like(v)
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(dgrad_process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    softmax_scale = q.shape[-1] ** (-0.5)
    softcap = 0.0

    def backward(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        attn_type: int,
        q_ranges_tensor: torch.Tensor,
        k_ranges_tensor: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ):
        attn_type_map = _get_attn_type(
            attn_type, q_ranges_tensor.shape[0], q_ranges_tensor.device
        )
        dq, dk, dv, _ = _flex_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            q_ranges=q_ranges_tensor,  # cu_seqlens_q
            k_ranges=k_ranges_tensor,  # cu_seqlens_k
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_type_map=attn_type_map,
            softmax_scale=softmax_scale,
            softcap=softcap,
            deterministic=False,  # deterministic,
            sm_margin=sm_margin,
        )

        return dq, dk, dv

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k, k_buffer)
            next_v = kv_comm.send_recv(v, v_buffer)
            kv_comm.commit()

        if step == 0:
            dq, dk, dv = backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                attn_type=1,
                q_ranges_tensor=ranges_tensor,  # use full q
                k_ranges_tensor=ranges_tensor,  # use full k
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            dq = dq.to(torch.float32)
            dk = dk.to(torch.float32)
            dv = dv.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                _dq, _dk, _dv = backward(
                    dout,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    attn_type=0,
                    q_ranges_tensor=ranges_tensor,  # use full q
                    k_ranges_tensor=k_ranges_tensor,  # use k of first half sequence
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k // 2,  # only half k
                )
            else:
                _dq, _dk, _dv = backward(
                    dout,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    attn_type=0,
                    q_ranges_tensor=q_ranges_tensor,
                    k_ranges_tensor=ranges_tensor,  # use full k
                    max_seqlen_q=max_seqlen_q // 2,  # only half q
                    max_seqlen_k=max_seqlen_k,
                )

            dq += _dq

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv
            dk += _dk
            dv += _dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k_buffer, v_buffer = k, v
            k, v = next_k, next_v

        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()

    dqkv = torch.concat(
        [dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)],
        dim=1,
    )

    return dqkv

class ZigZagFlexFlashAttnVarlenFunc(torch.autograd.Function):
    """
    封装了 ZigZag Flex Flash Attention 的 autograd.Function。
    
    这个类连接了前向和反向传播的逻辑，使得整个操作可以被 PyTorch 的自动微分引擎跟踪。
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        # --- 配置参数 ---
        max_seqlen_q: int,
        max_seqlen_k: int,
        sm_margin: int,
        ranges_tensor: torch.Tensor,
        q_ranges_tensor: torch.Tensor,
        k_ranges_tensor: torch.Tensor,
        # --- 分布式组 ---
        process_group: torch.distributed.ProcessGroup,
        dgrad_process_group: torch.distributed.ProcessGroup,
        # --- 其他 ---
        return_softmax: bool, # 用于兼容性，但在此实现中可能不使用
    ):
        """
        前向传播方法。
        
        调用核心的 forward 函数，并使用 ctx 保存反向传播所需的张量和参数。
        """
        # 调用您实现的核心前向传播逻辑
        out, softmax_lse = zigzag_flex_flash_attn_varlen_forward(
            process_group,
            q,
            k,
            v,
            max_seqlen_q,
            max_seqlen_k,
            sm_margin,
            ranges_tensor,
            q_ranges_tensor,
            k_ranges_tensor,
        )

        # 使用 ctx 保存反向传播所需的张量
        ctx.save_for_backward(
            q, k, v, out, softmax_lse, 
            ranges_tensor, q_ranges_tensor, k_ranges_tensor
        )
        
        # 使用 ctx 保存反向传播所需的非张量参数
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.sm_margin = sm_margin
        ctx.process_group = process_group
        ctx.dgrad_process_group = dgrad_process_group

        # forward 的返回值必须是张量。如果需要返回 lse，可以返回一个元组
        # 注意：如果 return_softmax 为 True，通常还会返回 softmax 矩阵，这里简化处理
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        """
        反向传播方法。
        
        从 ctx 中恢复保存的张量和参数，然后调用核心的 backward 函数。
        """
        # 从 ctx 恢复保存的张量
        (q, k, v, out, softmax_lse, 
         ranges_tensor, q_ranges_tensor, k_ranges_tensor) = ctx.saved_tensors

        # 调用您实现的核心反向传播逻辑
        # **重要**: 假设 backward 函数返回 (dq, dk, dv) 元组
        dq, dk, dv = zigzag_flex_flash_attn_varlen_backward(
            ctx.process_group,
            ctx.dgrad_process_group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.sm_margin,
            ranges_tensor,
            q_ranges_tensor,
            k_ranges_tensor,
        )

        # 返回的梯度必须与 forward 的输入参数一一对应
        # 对于不需要梯度的输入（如配置参数、进程组），返回 None
        return (
            dq,                      # q 的梯度
            dk,                      # k 的梯度
            dv,                      # v 的梯度
            None,                    # max_seqlen_q 的梯度
            None,                    # max_seqlen_k 的梯度
            None,                    # sm_margin 的梯度
            None,                    # ranges_tensor 的梯度
            None,                    # q_ranges_tensor 的梯度
            None,                    # k_ranges_tensor 的梯度
            None,                    # process_group 的梯度
            None,                    # dgrad_process_group 的梯度
            None,                    # return_softmax 的梯度
        )

# --- 用户友好的 apply 函数 ---

def zigzag_flex_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_margin: int,
    ranges_tensor: torch.Tensor,
    q_ranges_tensor: torch.Tensor,
    k_ranges_tensor: torch.Tensor,
    process_group: torch.distributed.ProcessGroup,
    dgrad_process_group: torch.distributed.ProcessGroup = None,
    return_softmax: bool = False,
):
    """
    对外暴露的接口函数，调用 autograd.Function.apply。
    
    参数:
    - q, k, v: 输入的 Query, Key, Value 张量。
    - max_seqlen_q, max_seqlen_k: Q 和 K 的最大序列长度。
    - sm_margin: softmax margin.
    - ranges_tensor, q_ranges_tensor, k_ranges_tensor: 定义序列范围的张量。
    - process_group: 用于 K/V 通信的进程组。
    - dgrad_process_group: 用于 dK/dV 梯度通信的进程组。如果为 None，则使用与 process_group 相同的组。
    - return_softmax: 是否返回 softmax 概率（当前实现中被忽略）。
    """
    if dgrad_process_group is None:
        dgrad_process_group = process_group

    return ZigZagFlexFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        max_seqlen_q,
        max_seqlen_k,
        sm_margin,
        ranges_tensor,
        q_ranges_tensor,
        k_ranges_tensor,
        process_group,
        dgrad_process_group,
        return_softmax,
    )


    
   