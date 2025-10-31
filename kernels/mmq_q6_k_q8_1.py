"""
// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;
"""

import torch
import triton
import triton.language as tl

QK_K = 256
K_SCALE_SIZE = 12
Q8_K = 32
Q6_K_BLOCK_SIZE = 210  # bytes
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
def _load_q6_k_subblk_weights(ptr, mask, idx):
    """
    idx in [0~8)
    返回值已减去32
    """

    # ql 排布：
    # bytes                 0~15    16~31   32~47   48~63   64~79   80~95   96~111  112~127
    # hi-4bit-to-subblk     4       5       6       7       12      13      14      15
    # lo-4bit-to-subblk     0       1       2       3       8       9       10      11
    # qh 排布：
    # bytes                 0~15    16~31   32~47   48~63
    # mask 0x03             0       1       8       9
    # mask 0x0C             2       3       10      11
    # mask 0x30             4       5       12      13
    # mask 0xC0             6       7       14      15

    _idx = tl.full((1, ), idx, tl.uint8)

    ql_start = (((_idx & 0x8) >> 1) | (_idx & 0x3)) << 4
    ql_shift = _idx & 0x4
    ql_mask = 0x0F << (_idx & 0x4)
    qh_start = ((((_idx >> 3) & 0x1) << 1 | (_idx & 0x1)) << 4) + 128
    qh_shift = _idx & 0x6
    qh_mask = 0x03 << (2 * ((_idx >> 1) & 0x3))

    _ptr_ql = ptr[:, None] + tl.arange(0, 16) + ql_start
    _ptr_qh = ptr[:, None] + tl.arange(0, 16) + qh_start
    _mask = mask[:, None]

    lower_bits = tl.load(_ptr_ql, mask=_mask, other=0).to(tl.uint8, bitcast=True)
    higher_bits = tl.load(_ptr_qh, mask=_mask, other=0b10101010).to(tl.uint8, bitcast=True)

    weights = ((lower_bits & ql_mask) >> ql_shift) | (((higher_bits & qh_mask) >> qh_shift) << 4)

    return weights - 32


@triton.jit
def mul_mat_q6_k_q8_1_triton(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    M,  # A 的列数
    N,  # B 的列数
    K,  # A 和 B 的行数
    qblock_num_in_K_direction_A: int,  # A 在 K 方向的量化块数量
    qblock_num_in_K_direction_B: int,  # B 在 K 方向的量化块数量
    q6_k_block_size_bytes: tl.constexpr,
    q8_1_block_size_bytes: tl.constexpr,
    QK_K: tl.constexpr,
    K_SCALE_SIZE: tl.constexpr,
    Q8_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Q6_K @ Q8_1
    
    C = (A @ B.T).T

    A: Q6_K
    B: Q8_1

    维度    元素                    size
    0(M/N)  A或B的一行              M或N
    1(K1)   A的一块，或B的8块        qblock_num_in_K_direction_A
    2(K2)   A的子块，或B的半块       16
    3(K3)   qs                     16
    """
    # 当前块的索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算块的起始行和列
    col_start = pid_m * BLOCK_SIZE_M
    row_start = pid_n * BLOCK_SIZE_N

    # 创建块的行和列索引
    cols = col_start + tl.arange(0, BLOCK_SIZE_M)  # (N,)
    rows = row_start + tl.arange(0, BLOCK_SIZE_N)  # (M,)

    # 创建掩码以处理边界情况
    A_mask = cols < M  # (N,)
    B_mask = rows < N  # (M,)

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    # 沿K方向遍历所有量化块
    # TODO: 是否可以不用遍历，而用tl.arange？
    for q6_k_block_idx_k in range(qblock_num_in_K_direction_A):  # dim: K1
        A_qblock_start = q6_k_block_size_bytes * (qblock_num_in_K_direction_A * cols + q6_k_block_idx_k)  # (N,)
        # 读取全局缩放因子 (N,)
        A_d = _load_2xint8_as_fp16(A_ptr + A_qblock_start + q6_k_block_size_bytes - 2, A_mask).to(tl.float32)
        A_scales_start = A_qblock_start + 192

        # 遍历每个q6_k子块
        for q6_k_subblock_idx in tl.static_range(16):  # dim: K2
            # A的子块的scales
            A_scales_ptr = (A_ptr + A_scales_start).reshape(BLOCK_SIZE_M, 1, 1) + q6_k_subblock_idx  # (M, K1, K2)
            A_scales_int8 = tl.load(
                A_scales_ptr,
                A_mask.reshape(BLOCK_SIZE_M, 1, 1),
                q6_k_subblock_idx,
            ).reshape(BLOCK_SIZE_M, 1, 1)  # (M, K1, K2)
            A_scales = A_d.reshape(BLOCK_SIZE_M, 1, 1) * A_scales_int8  # (M, K1, K2)
            # A的子块的权重（已减去32）
            A_weights: tl.tensor = _load_q6_k_subblk_weights(
                (A_ptr + A_qblock_start).reshape(BLOCK_SIZE_M, 1, 1),
                A_mask.reshape(BLOCK_SIZE_M, 1, 1),
                q6_k_subblock_idx,
            ).reshape(BLOCK_SIZE_M, 1, 1, 16).to(tl.int8, bitcast=True)  # (M, K1, K2, K3)

            # 从 B 中读取量化块
            B_qblock_starts = q8_1_block_size_bytes * \
                (qblock_num_in_K_direction_B * rows + 8 * q6_k_block_idx_k + (q6_k_subblock_idx // 2)) # (N,)
            # 缩放因子
            B_scale = _load_2xint8_as_fp16(B_ptr + B_qblock_starts, B_mask).to(tl.float32)
            B_scale = B_scale.reshape(BLOCK_SIZE_N, 1, 1)  # (N, K1, K2)
            # 权重
            B_weight_idx = (B_qblock_starts + 4)[:, None] \
                + tl.arange(0, 16) + (16 * (q6_k_subblock_idx % 2)) # (N, K3)
            B_weights = tl.load(B_ptr + B_weight_idx, mask=B_mask[:, None], other=0)
            B_weights = B_weights.reshape(BLOCK_SIZE_N, 1, 1, 16)  # (N, K1, K2, K3)

            # 点积
            # res = B_scale * (
            #           A_scale_0 * dot(A_weight_0, B_weight[:16])
            #         + A_scale_1 * dot(A_weigth_1, B_weight[16:])
            #       )

            # (M, K1, K2, K3) -> (K1, K2, K3, M) -> (K1*K2, K3, M)
            A_weights = A_weights.trans(1, 2, 3, 0).reshape(1, 16, BLOCK_SIZE_M)
            # (N, K1, K2, K3) -> (K1, K2, N, K3) -> (K1*K2, N, K3)
            B_weights = B_weights.trans(1, 2, 0, 3).reshape(1, BLOCK_SIZE_N, 16)
            R = tl.dot(B_weights, A_weights)  # (K1*K2, N, M)
            res = \
                A_scales.reshape(1, BLOCK_SIZE_M, 1, 1) \
              * B_scale.reshape(BLOCK_SIZE_N, 1, 1, 1) \
              * R.reshape(BLOCK_SIZE_N, BLOCK_SIZE_M, 1, 1)
            acc += res.reshape(BLOCK_SIZE_N, BLOCK_SIZE_M)

    # ===== 存储结果 =====
    out_ptrs = C_ptr + M * rows.reshape(BLOCK_SIZE_N, 1) + cols.reshape(1, BLOCK_SIZE_M)
    out_mask = B_mask.reshape(BLOCK_SIZE_N, 1) & A_mask.reshape(1, BLOCK_SIZE_M)
    tl.store(out_ptrs, acc, mask=out_mask)


# 辅助函数：启动Triton内核
def mmq_q6_k_q8_1(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    """
    执行Q6_K-Q8_1量化矩阵乘法: out = (A @ B.T).T
    
    Args:
        A: Q6_K 量化格式
        B: Q8_1 量化格式
        M: A 的行数
        N: B 的行数
        K: A 和 B 的列数 (应相等)
        
    Returns:
        输出矩阵 (float16)
    """
    assert (K % 256 == 0)

    # 定义 BLOCK_SIZE (之后可能会用autotune)
    # 每个 program 负责计算 BLOCK_SIZE_M * BLOCK_SIZE_N 区域大小的输出
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_N = 2

    # K方向的量化块数量
    qblock_num_in_K_direction_A = K // 256
    qblock_num_in_K_direction_B = K // 32

    # 创建输出张量
    C = torch.empty((N, M), device=A.device, dtype=torch.float16)

    # 启动Triton内核
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    mul_mat_q6_k_q8_1_triton[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        qblock_num_in_K_direction_A,
        qblock_num_in_K_direction_B,
        Q6_K_BLOCK_SIZE,
        Q8_1_BLOCK_SIZE,
        QK_K,
        K_SCALE_SIZE,
        Q8_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    return C
