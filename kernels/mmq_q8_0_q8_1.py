import torch
import triton
import triton.language as tl

# Q8_0量化常量
QK8_0 = 32  # 每个量化块中的元素数量


@triton.jit
def _load_2bytes_as_fp16(ptr, mask):
    _ptr = ptr.to(tl.pointer_type(tl.float16))
    return tl.load(_ptr, mask, 0)


@triton.jit
def _mmq_q8_0_q8_1_triton(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    M: int,  # A 的列数
    N: int,  # B 的列数
    K: int,  # A 和 B 的行数
    qblock_num_in_K_direction: int,  # K 方向的量化块数量
    Q8_0_SIZE: tl.constexpr,
    Q8_1_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    C = (A @ B^T)^T

    A: Q8_0
    B: Q8_1
    C: float16

    维度    元素          size
    0(M/N)  A或B的一行    M或N
    1(K)    A或B的一块    qblock_num_in_K_direction_A
    2(Q)    qs           32
    """
    # 当前块的索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算块的起始行和列
    col_start = pid_m * BLOCK_SIZE_M
    row_start = pid_n * BLOCK_SIZE_N

    # 创建块的行和列索引
    cols = col_start + tl.arange(0, BLOCK_SIZE_M)  # (M,)
    rows = row_start + tl.arange(0, BLOCK_SIZE_N)  # (N,)
    qblks = 0 + tl.arange(0, BLOCK_SIZE_K)  # (K1,)

    # 创建掩码以处理边界情况
    K_mask = qblks < qblock_num_in_K_direction
    A_mask = (cols < M)[:, None] & K_mask
    B_mask = (rows < N)[:, None] & K_mask

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), tl.float32)

    for k_stride_idx in tl.range(tl.cdiv(qblock_num_in_K_direction, BLOCK_SIZE_K)):
        # 加载A
        qblock_A_idx = qblock_num_in_K_direction * cols.reshape(BLOCK_SIZE_M, 1) + \
            BLOCK_SIZE_K * k_stride_idx + \
            tl.arange(0, BLOCK_SIZE_K)
        qblock_A_start = qblock_A_idx * Q8_0_SIZE
        # scale
        A_scale_start = qblock_A_start.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
        A_scale = _load_2bytes_as_fp16(A_ptr + A_scale_start, A_mask.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K))  # (M, K)
        A_scale = A_scale.cast(tl.float32)
        # qs
        A_qs_start = qblock_A_start.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K, 1) + 2 + tl.arange(0, 32)
        A_qs = tl.load(A_ptr + A_qs_start, mask=A_mask.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K, 1), other=0)  # (M, K, Q)

        # 加载B
        qblock_B_idx = \
            qblock_num_in_K_direction * rows.reshape(BLOCK_SIZE_N, 1) + \
            BLOCK_SIZE_K * k_stride_idx + \
            tl.arange(0, BLOCK_SIZE_K)
        qblock_B_start = qblock_B_idx * Q8_1_SIZE
        # scale
        B_scale_start = qblock_B_start.reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
        B_scale = _load_2bytes_as_fp16(B_ptr + B_scale_start, B_mask.reshape(BLOCK_SIZE_N, BLOCK_SIZE_K))  # (N, K)
        B_scale = B_scale.cast(tl.float32)
        # qs
        B_qs_start = qblock_B_start.reshape(BLOCK_SIZE_N, BLOCK_SIZE_K, 1) + 4 + tl.arange(0, 32)
        B_qs = tl.load(B_ptr + B_qs_start, mask=B_mask.reshape(BLOCK_SIZE_N, BLOCK_SIZE_K, 1), other=0)  # (N, K, Q)

        # dot
        a = A_qs.trans(1, 2, 0)  # (K, Q, M)
        b = B_qs.trans(1, 0, 2)  # (K, N, Q)
        R = tl.dot(b, a)  # (K, N, M)
        R = R.trans(1, 2, 0)  # (N, M, K)
        _res = A_scale.reshape(1, BLOCK_SIZE_M, BLOCK_SIZE_K) * B_scale.reshape(BLOCK_SIZE_N, 1, BLOCK_SIZE_K) * R
        res = tl.sum(_res, 2)

        acc += res

    # ===== 存储结果 =====
    out_ptrs = C_ptr + M * rows.reshape(BLOCK_SIZE_N, 1) + cols.reshape(1, BLOCK_SIZE_M)
    out_mask = (rows < N)[:, None] & (cols < M)
    tl.store(out_ptrs, acc, mask=out_mask)


# 辅助函数：启动Triton内核
def mmq_q8_0_q8_1(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    """
    执行Q8_0-Q8_1量化矩阵乘法: out = (A @ B.T).T
    
    Args:
        A: Q8_0 量化格式
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
    BLOCK_SIZE_K = 8

    # K方向的量化块数量
    assert (K % 32 == 0)
    qblock_num_in_K_direction = int(K / 32)

    # 创建输出张量
    C = torch.empty((N, M), device=A.device, dtype=torch.float16)

    # 启动Triton内核
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    _mmq_q8_0_q8_1_triton[grid](A, B, C, M, N, K, qblock_num_in_K_direction, 34, 36, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    return C
