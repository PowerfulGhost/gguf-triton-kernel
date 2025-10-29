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

QK_K = 256
K_SCALE_SIZE = 12
Q8_K = 32
Q4_K_BLOCK_SIZE = 144  # bytes
Q8_1_BLOCK_SIZE = 36  # bytes


@triton.jit
def _load_2xint8_as_fp16(ptr, mask, other_is_zero=False):
    lower_byte = tl.load(ptr, mask=mask, other=0)
    higher_byte = tl.load(ptr + 1, mask=mask, other=0 if other_is_zero else 60)  # 低八位0，高八位60，组合起来是float16的1.0
    fp16 = (
        higher_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16) << 8 | \
        lower_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16)
    ).cast(tl.float16, bitcast=True)
    return fp16


@triton.jit
def _load_2x_q4_k_subblock_scales_mins(ptr, mask, idx):
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

    _ptr = (ptr + 4)[:, None] + tl.arange(0, 2)
    _mask = mask[:, None]

    # flags
    idx_ge_2 = (idx & 2) >> 1
    idx_is_odd = idx & 1

    idx_ge_2 = idx_ge_2.cast(tl.uint8)
    idx_is_odd = idx_is_odd.cast(tl.uint8)

    # scales
    scales_lower_start = 8 * idx_ge_2 + 2 * idx_is_odd
    scales_higher_start = 2 * idx_is_odd
    scales_lower_mask = (0x30 * (1 - idx_ge_2)) | 0x0F
    scales_higher_mask = (0x3F * (1 - idx_ge_2)) | (0xC0 * idx_ge_2)
    # scales_lower_shift = 0
    scales_higher_shift = 2 * idx_ge_2

    scales_lower_bytes = tl.load(_ptr + scales_lower_start, mask=_mask, other=0).cast(tl.uint8, bitcast=True)
    scales_higher_bytes = tl.load(_ptr + scales_higher_start, mask=_mask, other=0).cast(tl.uint8, bitcast=True)

    scales = ((scales_lower_bytes & scales_lower_mask)) | \
        ((scales_higher_bytes & scales_higher_mask) >> scales_higher_shift)

    # mins
    mins_lower_start = 4 + 2 * idx
    mins_higher_start = 4 + 2 * idx_is_odd
    mins_lower_mask = (0x3F * (1 - idx_ge_2)) | (0xF0 * idx_ge_2)
    mins_higher_mask = (0x3F * (1 - idx_ge_2)) | (0xC0 * idx_ge_2)
    mins_lower_shift = 4 * idx_ge_2
    mins_higher_shift = 2 * idx_ge_2

    mins_lower_bytes = tl.load(_ptr + mins_lower_start, mask=_mask, other=0).cast(tl.uint8, bitcast=True)
    mins_higher_bytes = tl.load(_ptr + mins_higher_start, mask=_mask, other=0).cast(tl.uint8, bitcast=True)

    mins = ((mins_lower_bytes & mins_lower_mask) >> mins_lower_shift) | \
        ((mins_higher_bytes & mins_higher_mask) >> mins_higher_shift)

    return scales, mins


@triton.jit
def _load_2x_q4_k_subblk_weights(ptr, mask):
    _ptr = ptr[:, None] + (tl.arange(0, 64) % 32)  # [0,1,2,...,32,0,1,2,...,32]
    _mask = mask[:, None]

    packed_weights = tl.load(_ptr, _mask, other=0).cast(tl.uint8, bitcast=True)

    res = tl.where(
        tl.arange(0, 64) < 32,
        packed_weights & 0x0F,
        packed_weights >> 4,
    )

    return res


@triton.jit
def mul_mat_q4_k_q8_1_triton(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    M,  # A 的列数
    N,  # B 的列数
    K,  # A 和 B 的行数
    qblock_num_in_K_direction_A: int,  # A 在 K 方向的量化块数量
    qblock_num_in_K_direction_B: int,  # B 在 K 方向的量化块数量
    q4_k_block_size_bytes: tl.constexpr,
    q8_1_block_size_bytes: tl.constexpr,
    QK_K: tl.constexpr,
    K_SCALE_SIZE: tl.constexpr,
    Q8_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Q4_K @ Q8_1
    
    C = (A @ B.T).T

    A: Q4_K
    B: Q8_1

    维度    说明
    0(M/N)  -
    1(K1)   q4k块索引
    2(K2)   每2个q4k子块和2个q81块相乘，这是“每2”维度，一个q4k块中有4个“每2”
    3(K3)   “每2”内部维度，取值0或1,对应单个q4k子块或q81块
    4(K4)   单个量化权重块内部维度
    """
    # 当前块的索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算块的起始行和列
    col_start = pid_m * BLOCK_SIZE_M
    row_start = pid_n * BLOCK_SIZE_N

    # 创建块的行和列索引
    cols = col_start + tl.arange(0, BLOCK_SIZE_M)[:, None]  # (N, K)
    rows = row_start + tl.arange(0, BLOCK_SIZE_N)[:, None]  # (M, K)

    # 创建掩码以处理边界情况
    A_mask = cols < M
    B_mask = rows < N

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    # 沿K方向遍历所有量化块
    # TODO: 是否可以不用遍历，而用tl.arange？
    for q4_k_block_idx_k in range(qblock_num_in_K_direction_A):  # dim: K1
        # 从 A 中读取量化块
        A_qblock_start = q4_k_block_size_bytes * (qblock_num_in_K_direction_A * cols + q4_k_block_idx_k)
        # 读取全局缩放因子
        A_d = _load_2xint8_as_fp16(A_ptr + A_qblock_start, A_mask).to(tl.float32)  # 用于子块的scale (M, K1)
        A_dmin = _load_2xint8_as_fp16(A_ptr + A_qblock_start + 2, A_mask).to(tl.float32)  # 用于子块的min (M, K1)

        # 遍历和这个q4_k块对应的8个q8_1块
        # 每次计算2个子块，循环四次
        for _idx in tl.static_range(4):  # dim: K2
            # A的子块的scales和min (M, K1, K2, K3)
            A_scales, A_mins = _load_2x_q4_k_subblock_scales_mins(A_ptr + A_qblock_start, A_mask, _idx)
            A_scales = A_scales.reshape(BLOCK_SIZE_M, 1, 1, 2)
            A_mins = A_mins.reshape(BLOCK_SIZE_M, 1, 1, 1, 2)
            # A的子块的权重
            A_subblk_weight_start = (A_qblock_start + 16)[:, None] + 32 * _idx
            A_weights: tl.tensor = _load_2x_q4_k_subblk_weights(A_ptr + A_subblk_weight_start, A_mask[:, None])
            A_weights = A_weights.reshape(BLOCK_SIZE_M, 1, 1, 2, 32)  # (M, K1, K2, K3, K4)
            A_weights = A_weights.cast(tl.int8, bitcast=True)  # bitcast安全，因为A的量化权重小于128，不可能cast成负数

            # 从 B 中读取量化块
            B_qblock_starts = q8_1_block_size_bytes * \
                (qblock_num_in_K_direction_B * rows + 8 * q4_k_block_idx_k + 2 * _idx + tl.arange(0, 2))
            # 缩放因子
            B_scale = _load_2xint8_as_fp16(B_ptr + B_qblock_starts, B_mask).to(tl.float32)
            B_scale = B_scale.reshape(BLOCK_SIZE_N, 1, 1, 2)  # (N, K1, K2, K3)
            # 缩放因子和权重和的乘积（预计算以减少重复计算）
            B_scale_qwsum_prod = _load_2xint8_as_fp16(B_ptr + B_qblock_starts + 2, B_mask, other_is_zero=True).to(tl.float32)
            B_scale_qwsum_prod = B_scale_qwsum_prod.reshape(BLOCK_SIZE_N, 1, 1, 2)  # (N, K1, K2, K3)
            # 权重
            B_weight_idx = (B_qblock_starts + 4).reshape(BLOCK_SIZE_N, 2, 1) + tl.arange(0, 32)
            B_weights = tl.load(B_ptr + B_weight_idx, mask=B_mask[:, None], other=0)
            B_weights = B_weights.reshape(BLOCK_SIZE_N, 1, 1, 2, 32)

            # 点积
            # res = alpha - beta
            #     = (A_d * A_scale) * B_scale * sum(q_A_i,*q_B_i) - (A_dmin * A_min) * B_scale * sum(q_B_i)

            A_weights = A_weights.trans(1, 2, 3, 4, 0).reshape(2, 32, BLOCK_SIZE_M)  # (K1*K2*K3, K4, M)
            B_weights = B_weights.trans(1, 2, 3, 0, 4).reshape(2, BLOCK_SIZE_N, 32)  # (K1*K2*K3, N, K4)

            R = tl.dot(B_weights, A_weights)  # (K1*K2*K3, N, M)
            R = R.trans(1, 2, 0).reshape(BLOCK_SIZE_N, BLOCK_SIZE_M, 1, 1, 2)  # (N, M, K1, K2, K3)

            # (N, M, K1, K2, K3)
            alpha = \
                  A_d.reshape(1, BLOCK_SIZE_M, 1, 1, 1) \
                * A_scales.reshape(1, BLOCK_SIZE_M, 1, 1, 2) \
                * B_scale.reshape(BLOCK_SIZE_N, 1, 1, 1, 2) \
                * R

            beta = \
                  A_dmin.reshape(1, BLOCK_SIZE_M, 1, 1, 1) \
                * A_mins.reshape(1, BLOCK_SIZE_M, 1, 1, 2) \
                * B_scale_qwsum_prod.reshape(BLOCK_SIZE_N, 1, 1, 1, 2)

            res = alpha - beta  # (N, M, K1, K2, K3)

            # 把两个子块的点积结果规约（K3维度），并累加到结果中
            res = tl.sum(res, -1)
            acc += res.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)

    # ===== 存储结果 =====
    out_ptrs = C_ptr + rows * M + cols.reshape(1, BLOCK_SIZE_M)
    out_mask = B_mask & A_mask.reshape(1, BLOCK_SIZE_M)
    tl.store(out_ptrs, acc, mask=out_mask)


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
        qblock_num_in_K_direction_A,
        qblock_num_in_K_direction_B,
        Q4_K_BLOCK_SIZE,
        Q8_1_BLOCK_SIZE,
        QK_K,
        K_SCALE_SIZE,
        Q8_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    return C
