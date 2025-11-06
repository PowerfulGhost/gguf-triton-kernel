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


@triton.jit
def round_to_nearest_int(fval):
    val = fval + 12582912.0
    ival = val.cast(tl.int32, bitcast=True)
    return (ival & 0x007fffff) - 0x00400000


@triton.jit
def _load_q6_k_subblk_weights(qblock_start_ptr, qblock_mask, subblk_idx):
    """
    idx in [0~8)
    返回值已减去32

    qblock_start_ptr: (M, K1)
    qblock_mask: (M, K1)
    subblk_idx: (K2,)
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

    _idx = subblk_idx.to(tl.uint8)

    ql_start = (((_idx & 0x8) >> 1) | (_idx & 0x3)) << 4
    ql_shift = _idx & 0x4
    ql_mask = 0x0F << (_idx & 0x4)
    qh_start = ((((_idx >> 3) & 0x1) << 1 | (_idx & 0x1)) << 4) + 128
    qh_shift = _idx & 0x6
    qh_mask = 0x03 << (2 * ((_idx >> 1) & 0x3))

    _ptr_ql = (qblock_start_ptr.expand_dims(2) + ql_start).expand_dims(3) + tl.arange(0, 16)
    _ptr_qh = (qblock_start_ptr.expand_dims(2) + qh_start).expand_dims(3) + tl.arange(0, 16)
    _mask = qblock_mask.expand_dims(2).expand_dims(3)

    lower_bits = tl.load(_ptr_ql, mask=_mask, other=0).to(tl.uint8, bitcast=True)
    higher_bits = tl.load(_ptr_qh, mask=_mask, other=0b10101010).to(tl.uint8, bitcast=True)

    weights = (((lower_bits & ql_mask[:, None]) >> ql_shift[:, None]) | (((higher_bits & qh_mask[:, None]) >> qh_shift[:, None]) << 4)).to(tl.int8, bitcast=True)

    return weights - 32


@triton.jit
def mul_mat_q6_k_triton(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    M: int,  # A 的列数
    N: int,  # B 的列数
    A_qblock_num_in_K_direction: int,  # A 在 K 方向的量化块数量
    Q6_K_BLOCK_SIZE: tl.constexpr,
    Q6_K_SUBBLK_NUM: tl.constexpr,
    QK_K: tl.constexpr,
    QK8_1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_K2: tl.constexpr,
):
    """
    Q6_K @ fp16
    
    C = (A @ B.T).T

    A: Q6_K
    B: fp16

    维度     元素                   size
    0(M/N)  A或B的一行              M或N
    1(K1)   A的一块，或B的8块        BLOCK_SIZE_K1
    2(K2)   A的子块，或B的半块       BLOCK_SIZE_K2
    3(K3)   qs                     16
    """
    # 当前块的索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算块的起始行和列
    col_start = pid_m * BLOCK_SIZE_M
    row_start = pid_n * BLOCK_SIZE_N

    # 创建块的行和列索引
    M_idx = col_start + tl.arange(0, BLOCK_SIZE_M)  # (M,)
    N_idx = row_start + tl.arange(0, BLOCK_SIZE_N)  # (N,)

    # 创建掩码以处理边界情况
    M_mask = M_idx < M  # (M,)
    N_mask = N_idx < N  # (N,)

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    # 沿K1方向遍历所有量化块
    for k1_stride_idx in tl.range(0, tl.cdiv(A_qblock_num_in_K_direction, BLOCK_SIZE_K1)):

        K1_idx = BLOCK_SIZE_K1 * k1_stride_idx + tl.arange(0, BLOCK_SIZE_K1)  # (K1,)
        A_qblock_idx = A_qblock_num_in_K_direction * M_idx.reshape(BLOCK_SIZE_M, 1) + K1_idx
        A_qblock_start = Q6_K_BLOCK_SIZE * A_qblock_idx  # (M, K1)

        K1_mask = K1_idx < A_qblock_num_in_K_direction  # (K1,)
        M_K1_mask = M_mask.reshape(BLOCK_SIZE_M, 1) & K1_mask.reshape(1, BLOCK_SIZE_K1)  # (M, K1)

        # 加载A_d
        A_d_start = A_qblock_start + 208  # (M, K1)
        A_d = tl.load((A_ptr + A_d_start).to(tl.pointer_type(tl.float16)), mask=M_K1_mask, other=0)  # (M, K1)
        A_d = A_d.to(tl.float32)

        # 沿K2方向遍历所有子块
        for k2_stride_idx in tl.range(0, tl.cdiv(Q6_K_SUBBLK_NUM, BLOCK_SIZE_K2)):
            K2_idx = BLOCK_SIZE_K2 * k2_stride_idx + tl.arange(0, BLOCK_SIZE_K2)  # (K2,)
            K2_mask = K2_idx < Q6_K_SUBBLK_NUM  # (K2,)
            A_K1_K2_mask = M_K1_mask.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K1, 1) & K2_mask

            # 加载A_scales
            A_scales_start = (A_qblock_start + 192).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K1, 1) + K2_idx  # (M, K1, K2)
            A_scales_ptr = (A_ptr + A_scales_start).to(tl.pointer_type(tl.int8))
            A_scales = tl.load(A_scales_ptr, A_K1_K2_mask, 0)  # (M, K1, K2)
            A_scales = A_d.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K1, 1) * A_scales

            # 加载A_qs
            A_qs = _load_q6_k_subblk_weights(A_ptr + A_qblock_start, M_K1_mask, K2_idx)  # (M, K1, K2, K3)

            # 加载B
            K2_idx_B_block = (BLOCK_SIZE_K2 // 2) * k2_stride_idx + tl.arange(0, BLOCK_SIZE_K2 // 2)  # (K2/2,)
            K2_mask_B_block = K2_idx_B_block < (QK_K / QK8_1)
            N_K1_mask = N_mask.reshape(BLOCK_SIZE_N, 1) & K1_mask
            N_K1_K2_mask = N_K1_mask.expand_dims(2) & K2_mask_B_block  # (N, K1, K2/2)

            B_block_idx = (A_qblock_num_in_K_direction * 8 * N_idx.expand_dims(1) + 8 * BLOCK_SIZE_K1 * K1_idx).expand_dims(2) + K2_idx_B_block  # (N, K1, K2/2)
            B_block_start = QK8_1 * B_block_idx  # (N, K1, K2/2)
            B_weight_idx = B_block_start.expand_dims(3) + tl.arange(0, QK8_1)

            B = tl.load(B_ptr + B_weight_idx, N_K1_K2_mask.expand_dims(3), 0)  # (N, K1, K2/2, K3*2)

            # 量化B
            B_max = tl.max(tl.abs(B), 3)  # (N, K1, K2/2)
            B_scales = B_max / 127.0
            B_iscales = 127.0 / B_max
            B_qs = round_to_nearest_int(B * B_iscales.expand_dims(3)).to(tl.int8)  # (N, K1, K2/2, K3*2)
            B_qs = B_qs.reshape(BLOCK_SIZE_N, BLOCK_SIZE_K1, BLOCK_SIZE_K2 // 2, 2, 16)  # (N, K1, K2/2, 2, K3)

            # dot
            a = A_qs.trans(1, 2, 3, 0).reshape(BLOCK_SIZE_K1 * BLOCK_SIZE_K2, 16, BLOCK_SIZE_M)  # (K1*K2, K3, M)
            b = B_qs.trans(1, 2, 3, 0, 4).reshape(BLOCK_SIZE_K1 * BLOCK_SIZE_K2, BLOCK_SIZE_N, 16)  # (K1*K2, N, K3)
            R = tl.dot(b, a).trans(1, 2, 0).reshape(BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_K1, BLOCK_SIZE_K2)  # (N, M, K1, K2)

            a_sc = A_scales.reshape(1, BLOCK_SIZE_M, BLOCK_SIZE_K1, BLOCK_SIZE_K2)  # (1, M, K1, K2)
            b_sc = B_scales.reshape(BLOCK_SIZE_N, 1, BLOCK_SIZE_K1, BLOCK_SIZE_K2 // 2, 1).broadcast_to(BLOCK_SIZE_N, 1, BLOCK_SIZE_K1, BLOCK_SIZE_K2 // 2, 2).reshape(BLOCK_SIZE_N, 1, BLOCK_SIZE_K1, BLOCK_SIZE_K2)  # (N, 1, K1, K2)

            res = (a_sc * b_sc * R).reshape(BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_K1 * BLOCK_SIZE_K2)  # (N, M)
            res = tl.sum(res, 2)

            acc += res

    # ===== 存储结果 =====
    out_ptrs = C_ptr + M * N_idx.reshape(BLOCK_SIZE_N, 1) + M_idx.reshape(1, BLOCK_SIZE_M)
    out_mask = N_mask.reshape(BLOCK_SIZE_N, 1) & M_mask.reshape(1, BLOCK_SIZE_M)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


QK_K = 256
Q6_K_SUBBLK_NUM = 16
QK8_1 = 32
Q6_K_BLOCK_SIZE = 210  # bytes
Q8_1_BLOCK_SIZE = 36  # bytes


# 辅助函数：启动Triton内核
def mmq_q6_k(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    """
    执行Q6_K-fp16量化矩阵乘法: out = (A @ B.T).T
    
    Args:
        A: Q6_K 量化格式
        B: fp16
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
    BLOCK_SIZE_K1 = 1
    BLOCK_SIZE_K2 = 16

    # K方向的量化块数量
    A_qblock_num_in_K_direction = K // 256

    # 创建输出张量
    C = torch.empty((N, M), device=A.device, dtype=torch.float16)

    # 启动Triton内核
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    mul_mat_q6_k_triton[grid](
        A,
        B,
        C,
        M,
        N,
        A_qblock_num_in_K_direction=A_qblock_num_in_K_direction,
        Q6_K_BLOCK_SIZE=Q6_K_BLOCK_SIZE,
        Q6_K_SUBBLK_NUM=Q6_K_SUBBLK_NUM,
        QK_K=QK_K,
        QK8_1=QK8_1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K1=BLOCK_SIZE_K1,
        BLOCK_SIZE_K2=BLOCK_SIZE_K2,
    )

    return C
