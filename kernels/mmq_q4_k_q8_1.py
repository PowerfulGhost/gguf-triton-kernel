"""
typedef struct
{
    union
    {
        struct
        {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];         // 4--bit quants
} block_q4_K;
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_q4_k_scales_mins(qblock_start_ptr, qblock_mask, subblk_idx):
    """
    qblock_start_ptr: (M, K1)
    qblock_mask: (M, K1)
    subblk_idx: (K2,)
    """

    ### Unpacking the following: ###
    #  0 EEAAAAAA
    #  1 FFBBBBBB
    #  2 GGCCCCCC
    #  3 HHDDDDDD

    #  4 eeaaaaaa
    #  5 ffbbbbbb
    #  6 ggcccccc
    #  7 hhdddddd

    #  8 eeeeEEEE
    #  9 ffffFFFF
    # 10 ggggGGGG
    # 11 hhhhHHHH

    idx = subblk_idx.to(tl.uint8)

    scale_idx_lo = ((idx & 4) << 1) | (idx & 3)
    scale_idx_hi = idx & 3
    scale_msk_lo = tl.full((8, ), 0x0F, tl.uint8)
    scale_msk_hi = 0x30 << ((idx >> 2) << 1)
    scale_sft_lo = tl.full((8, ), 0, tl.uint8)
    scale_sft_hi = (idx >> 2) << 1

    min_idx_lo = idx + 4
    min_idx_hi = (idx & 3) + 4
    min_msk_lo = 0x0F << ((idx >> 2) << 2)
    min_msk_hi = 0x30 << ((idx >> 2) << 1)
    min_sft_lo = (idx >> 2) << 2
    min_sft_hi = (idx >> 2) << 1

    ptr = (qblock_start_ptr + 4).expand_dims(2).to(tl.pointer_type(tl.uint8))
    mask = qblock_mask.expand_dims(2)
    scale_bytes_lo = tl.load(ptr + scale_idx_lo, mask=mask, other=0)
    scale_bytes_hi = tl.load(ptr + scale_idx_hi, mask=mask, other=0)
    min_bytes_lo = tl.load(ptr + min_idx_lo, mask=mask, other=0)
    min_bytes_hi = tl.load(ptr + min_idx_hi, mask=mask, other=0)

    scales = ((scale_bytes_hi & scale_msk_hi) >> scale_sft_hi) | ((scale_bytes_lo & scale_msk_lo) >> scale_sft_lo)
    mins = ((min_bytes_hi & min_msk_hi) >> min_sft_hi) | ((min_bytes_lo & min_msk_lo) >> min_sft_lo)

    return scales.to(tl.int8, bitcast=True), mins.to(tl.int8, bitcast=True)


@triton.jit
def _load_q4_k_qs(qblock_start_ptr, qblock_mask, subblk_idx):
    """
    qblock_start_ptr: (M, K1)
    qblock_mask: (M, K1)
    subblk_idx: (K2,)
    """

    # idx   bytes   mask
    # 0     0-32    0x0F
    # 1     0-32    0xF0
    # 2     32-64   0x0F
    # 3     32-64   0xF0
    # 4     64-96   0x0F
    # 5     64-96   0xF0
    # 6     96-128  0x0F
    # 7     96-128  0xF0

    idx = subblk_idx.to(tl.uint8)

    qs_start = (idx & 0xFE) << 4
    qs_idx = qs_start.expand_dims(1) + tl.arange(0, 32)
    qs_mask = 0x0F << ((idx & 1) << 2)
    qs_shift = (idx & 1) << 2

    ptr = (qblock_start_ptr + 16).expand_dims(2).expand_dims(3).to(tl.pointer_type(tl.uint8))
    mask = qblock_mask.expand_dims(2).expand_dims(3)
    qs_bytes = tl.load(ptr + qs_idx, mask=mask, other=0)

    qs = (qs_bytes & qs_mask.expand_dims(1)) >> qs_shift.expand_dims(1)

    return qs.to(tl.int8, bitcast=True)


@triton.jit
def mul_mat_q4_k_q8_1_triton(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    M: int,  # A 的列数
    N: int,  # B 的列数
    K: int,  # A 和 B 的行数
    qblock_num_in_K_direction_A: int,  # A 在 K 方向的量化块数量
    qblock_num_in_K_direction_B: int,  # B 在 K 方向的量化块数量
    Q4_K_BLOCK_SIZE: tl.constexpr,
    Q8_1_BLOCK_SIZE: tl.constexpr,
    Q4_K_SUBBLK_NUM: tl.constexpr,
    QK_K: tl.constexpr,
    Q8_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_K2: tl.constexpr,
):
    """
    Q4_K @ Q8_1
    
    C = (A @ B.T).T

    A: Q4_K
    B: Q8_1

    维度     元素                   size
    0(M/N)  A或B的一行              M或N
    1(K1)   A的一块，或B的8块       BLOCK_SIZE_K1
    2(K2)   Ah或B的子块             BLOCK_SIZE_K2
    3(K3)   qs                      32
    """
    # 当前块的索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算块的起始行和列
    col_start = pid_m * BLOCK_SIZE_M
    row_start = pid_n * BLOCK_SIZE_N

    # 创建块的行和列索引
    M_idx = col_start + tl.arange(0, BLOCK_SIZE_M)  # (M, K)
    N_idx = row_start + tl.arange(0, BLOCK_SIZE_N)  # (N, K)

    # 创建掩码以处理边界情况
    M_mask = M_idx < M
    N_mask = N_idx < N

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for k1_stride_idx in tl.range(qblock_num_in_K_direction_A // BLOCK_SIZE_K1):
        K1_idx = tl.arange(0, BLOCK_SIZE_K1) + BLOCK_SIZE_K1 * k1_stride_idx  # (K1,)
        K1_mask = K1_idx < qblock_num_in_K_direction_A
        M_K1_mask = M_mask.expand_dims(1) & K1_mask
        N_K1_mask = N_mask.expand_dims(1) & K1_mask

        A_qblock_idx = qblock_num_in_K_direction_A * M_idx.expand_dims(1) + K1_idx
        A_qblock_start = Q4_K_BLOCK_SIZE * A_qblock_idx
        A_qblock_ptr = A_ptr + A_qblock_start  # (M, K1)

        A_d_ptr = A_qblock_ptr.to(tl.pointer_type(tl.float16))
        A_d = tl.load(A_d_ptr, M_K1_mask, other=0).to(tl.float32)
        A_dmin_ptr = A_d_ptr + 1
        A_dmin = tl.load(A_dmin_ptr, M_K1_mask, other=0).to(tl.float32)

        for k2_stride_idx in tl.range(Q4_K_SUBBLK_NUM // BLOCK_SIZE_K2):
            K2_idx = tl.arange(0, BLOCK_SIZE_K2) + BLOCK_SIZE_K2 * k2_stride_idx  # (K2,)
            K2_mask = K2_idx < Q4_K_SUBBLK_NUM
            N_K1_K2_mask = N_K1_mask.expand_dims(2) & K2_mask

            # 加载A_scales, A_mins, A_qs
            A_scales_int8, A_mins_int8 = _load_q4_k_scales_mins(A_qblock_ptr, M_K1_mask, K2_idx)  # (M, K1, K2)
            A_scales = A_d.expand_dims(2) * A_scales_int8  # (M, K1, K2)
            A_mins = A_dmin.expand_dims(2) * A_mins_int8  # (M, K1, K2)
            A_qs = _load_q4_k_qs(A_qblock_ptr, M_K1_mask, K2_idx)  # (M, K1, K2, K3)

            # 加载B_scales, B_sc_qs_prow, B_qs
            B_qblock_idx = ((qblock_num_in_K_direction_B * N_idx.expand_dims(1)) + 8 * BLOCK_SIZE_K1 * K1_idx).expand_dims(2) + K2_idx
            B_qblock_start = Q8_1_BLOCK_SIZE * B_qblock_idx
            B_qblock_ptr = B_ptr + B_qblock_start  # (M, K1, K2)

            B_scales = tl.load(B_qblock_ptr.to(tl.pointer_type(tl.float16)), mask=N_K1_K2_mask, other=0)  # (N, K1, K2)
            B_sc_qs_prod = tl.load((B_qblock_ptr + 2).to(tl.pointer_type(tl.float16)), mask=N_K1_K2_mask, other=0)  # (N, K1, K2)
            B_qs_ptr = (B_qblock_ptr + 4).expand_dims(3) + tl.arange(0, Q8_K)
            B_qs = tl.load(B_qs_ptr, mask=N_K1_K2_mask.expand_dims(3), other=0)  # (N, K1, K2, K3)

            # dot
            a = A_qs.trans(1, 2, 3, 0).reshape(BLOCK_SIZE_K1 * BLOCK_SIZE_K2, QK_K // Q4_K_SUBBLK_NUM, BLOCK_SIZE_M)  # (K1*K2, K3, M)
            b = B_qs.trans(1, 2, 0, 3).reshape(BLOCK_SIZE_K1 * BLOCK_SIZE_K2, BLOCK_SIZE_N, Q8_K)  # (K1*K2, N, K3)
            R = tl.dot(b, a).trans(1, 2, 0).reshape(BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_K1, BLOCK_SIZE_K2)  # (N, M, K1, K2)

            # res = alpha - beta
            #     = A_scales * B_scales * R - A_mins * B_scales * sum(q_B_i)
            a_sc = A_scales.reshape(1, BLOCK_SIZE_M, BLOCK_SIZE_K1, BLOCK_SIZE_K2)
            a_m = A_mins.reshape(1, BLOCK_SIZE_M, BLOCK_SIZE_K1, BLOCK_SIZE_K2)
            b_sc = B_scales.reshape(BLOCK_SIZE_N, 1, BLOCK_SIZE_K1, BLOCK_SIZE_K2)
            b_sqp = B_sc_qs_prod.reshape(BLOCK_SIZE_N, 1, BLOCK_SIZE_K1, BLOCK_SIZE_K2)

            res = a_sc * b_sc * R - a_m * b_sqp
            res = res.reshape(BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_K1 * BLOCK_SIZE_K2)

            acc += tl.sum(res, 2)

    # ===== 存储结果 =====
    out_ptrs = C_ptr + M * N_idx.reshape(BLOCK_SIZE_N, 1) + M_idx.reshape(1, BLOCK_SIZE_M)
    out_mask = N_mask.reshape(BLOCK_SIZE_N, 1) & M_mask.reshape(1, BLOCK_SIZE_M)
    tl.store(out_ptrs, acc, mask=out_mask)


Q4_K_BLOCK_SIZE = 144  # bytes
Q8_1_BLOCK_SIZE = 36  # bytes
Q4_K_SUBBLK_NUM = 8
QK_K = 256
Q8_K = 32


# 辅助函数：启动Triton内核
def mmq_q4_k_q8_1(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    """
    执行Q4_K-Q8_1量化矩阵乘法: out = (A @ B.T).T
    
    Args:
        A: Q4_K 量化格式
        B: Q8_1 量化格式
        M: A 的行数
        N: B 的行数
        K: A 和 B 的列数 (应相等)
        
    Returns:
        输出矩阵 (float16)
    """

    # 定义 BLOCK_SIZE (之后可能会用autotune)
    # 每个 program 负责计算 BLOCK_SIZE_M * BLOCK_SIZE_N 区域大小的输出
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_N = 2
    BLOCK_SIZE_K1 = 1
    BLOCK_SIZE_K2 = 8

    # K方向的量化块数量
    assert (K % 256 == 0)
    qblock_num_in_K_direction_A = K // 256
    qblock_num_in_K_direction_B = K // 32

    # 创建输出张量
    C = torch.empty((N, M), device=A.device, dtype=torch.float16)

    # 启动Triton内核
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    mul_mat_q4_k_q8_1_triton[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        qblock_num_in_K_direction_A=qblock_num_in_K_direction_A,
        qblock_num_in_K_direction_B=qblock_num_in_K_direction_B,
        Q4_K_BLOCK_SIZE=Q4_K_BLOCK_SIZE,
        Q8_1_BLOCK_SIZE=Q8_1_BLOCK_SIZE,
        Q4_K_SUBBLK_NUM=Q4_K_SUBBLK_NUM,
        QK_K=QK_K,
        Q8_K=Q8_K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K1=BLOCK_SIZE_K1,
        BLOCK_SIZE_K2=BLOCK_SIZE_K2,
    )

    return C
