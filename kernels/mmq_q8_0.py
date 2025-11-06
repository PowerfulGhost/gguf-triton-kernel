import torch
import triton
import triton.language as tl


@triton.jit
def round_to_nearest_int(fval):
    val = fval + 12582912.0
    ival = val.cast(tl.int32, bitcast=True)
    return (ival & 0x007fffff) - 0x00400000


@triton.jit
def mmq_q8_0_triton(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    M: int,  # A 的列数
    N: int,  # B 的列数
    A_qblock_num_in_K_direction: int,  # K 方向的量化块数量
    Q8_0_SIZE: tl.constexpr,
    QK8_1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    C = (A @ B^T)^T
z
    A: Q8_0
    B: fp16
    C: float16

    维度    元素                    size
    0(M/N)  A或B的一行              M或N
    1(K)    A的一块或B的32个元素     qblock_num_in_K_direction_A
    2(Q)    qs或B的元素             32
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
    qblks = 0 + tl.arange(0, BLOCK_SIZE_K)  # (K1,)

    # 创建掩码以处理边界情况
    K_mask = qblks < A_qblock_num_in_K_direction
    A_mask = (M_idx < M).expand_dims(1) & K_mask  # (M, K1)
    B_mask = ((N_idx < N).expand_dims(1) & K_mask).expand_dims(2)  # (M, K1, Q)

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), tl.float32)

    for k_stride_idx in tl.range(tl.cdiv(A_qblock_num_in_K_direction, BLOCK_SIZE_K)):
        # 加载A
        A_qblock_idx = A_qblock_num_in_K_direction * M_idx.expand_dims(1) + BLOCK_SIZE_K * k_stride_idx + tl.arange(0, BLOCK_SIZE_K)
        qblock_A_start = A_qblock_idx * Q8_0_SIZE
        # scale
        A_scale_start = qblock_A_start.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
        A_scale = tl.load((A_ptr + A_scale_start).to(tl.pointer_type(tl.float16)), mask=A_mask, other=0)  # (M, K1)
        A_scale = A_scale.cast(tl.float32)
        # qs
        A_qs_start = qblock_A_start.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K, 1) + 2 + tl.arange(0, 32)
        A_qs = tl.load(A_ptr + A_qs_start, mask=A_mask.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K, 1), other=0)  # (M, K, Q)

        # 加载B
        B_idx = QK8_1 * (A_qblock_num_in_K_direction * N_idx.expand_dims(1) + BLOCK_SIZE_K * k_stride_idx + tl.arange(0, BLOCK_SIZE_K)).expand_dims(2) + tl.arange(0, QK8_1)  # (N, K, Q)
        B = tl.load(B_ptr + B_idx, B_mask, other=0)  # (N, K, Q)
        # 量化B
        B_max = tl.max(tl.abs(B), 2)  # (N, K)
        B_scale = B_max / 127.0  # (N, K)
        B_iscale = 127.0 / B_max
        B_qs = round_to_nearest_int(B * B_iscale.expand_dims(2)).cast(tl.int8)

        # dot
        a = A_qs.trans(1, 2, 0)  # (K, Q, M)
        b = B_qs.trans(1, 0, 2)  # (K, N, Q)
        R = tl.dot(b, a)  # (K, N, M)
        R = R.trans(1, 2, 0)  # (N, M, K)
        _res = A_scale.reshape(1, BLOCK_SIZE_M, BLOCK_SIZE_K) * B_scale.reshape(BLOCK_SIZE_N, 1, BLOCK_SIZE_K) * R
        res = tl.sum(_res, 2)

        acc += res

    # ===== 存储结果 =====
    out_ptrs = C_ptr + M * N_idx.reshape(BLOCK_SIZE_N, 1) + M_idx.reshape(1, BLOCK_SIZE_M)
    out_mask = (N_idx < N)[:, None] & (M_idx < M)
    tl.store(out_ptrs, acc, mask=out_mask)


QK8_0 = 32  # 每个量化块中的元素数量
QK8_1 = 32
Q8_0_SIZE = 34  # bytes


# 辅助函数：启动Triton内核
def mmq_q8_0(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    """
    执行 Q8_0-fp16 量化矩阵乘法: out = (A @ B.T).T
    
    Args:
        A: Q8_0 量化格式
        B: fp16
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
    BLOCK_SIZE_K = 8

    # K方向的量化块数量
    assert (K % 32 == 0)
    A_qblock_num_in_K_direction = int(K / 32)

    # 创建输出张量
    C = torch.empty((N, M), device=A.device, dtype=torch.float16)

    # 启动Triton内核
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    mmq_q8_0_triton[grid](
        A,
        B,
        C,
        M,
        N,
        A_qblock_num_in_K_direction=A_qblock_num_in_K_direction,
        Q8_0_SIZE=Q8_0_SIZE,
        QK8_1=QK8_1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return C
