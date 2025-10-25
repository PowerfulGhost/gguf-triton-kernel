import torch
import triton
import triton.language as tl

# Q8_0量化常量
QK8_0 = 32  # 每个量化块中的元素数量


@triton.jit
def vec_dot_q8_0_q8_0(A_weights: tl.tensor, A_scale: tl.tensor, B_weights: tl.tensor, B_scale: tl.tensor):
    """
    计算两个Q8_0量化块的点积
    """
    return A_scale[None, :] * B_scale[:, None] * tl.dot(B_weights, A_weights.T)


@triton.jit
def _mmq_q8_0_q8_1_triton(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    M,  # A 的列数
    N,  # B 的列数
    K,  # A 和 B 的行数
    qblock_num_in_K_direction: int,  # K 方向的量化块数量
    q8_0_block_size_bytes: tl.constexpr,
    q8_1_block_size_bytes: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    C = (A @ B^T)^T

    A: Q8_0
    B: Q8_1
    C: float16
    """
    bytes_in_K_direction_A = q8_0_block_size_bytes * qblock_num_in_K_direction
    bytes_in_K_direction_B = q8_1_block_size_bytes * qblock_num_in_K_direction

    # 当前块的索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算块的起始行和列
    row_start = pid_n * BLOCK_SIZE_M
    col_start = pid_m * BLOCK_SIZE_N

    # 创建块的行和列索引
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    cols = col_start + tl.arange(0, BLOCK_SIZE_N)

    # 创建掩码以处理边界情况
    A_mask = cols < M
    B_mask = rows < N

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    # 沿K方向遍历所有量化块
    # TODO: 是否可以不用遍历，而用tl.arange？
    for qblock_idx_k in range(0, qblock_num_in_K_direction):
        # 从 A 中读取量化块
        # row = [0, 1], col = [0, 1]
        # A -> col
        A_qblock_idx = (cols * bytes_in_K_direction_A) + qblock_idx_k * q8_0_block_size_bytes
        # 缩放因子
        A_scale_lower_byte_idx = A_qblock_idx
        A_scale_higher_byte_idx = A_qblock_idx + 1
        A_scale_lower_byte = tl.load(A_ptr + A_scale_lower_byte_idx, mask=A_mask, other=60)  # 低八位60，高八位0，组合起来是float16的1.0
        A_scale_higher_byte = tl.load(A_ptr + A_scale_higher_byte_idx, mask=A_mask, other=0)
        A_scale = (
            A_scale_higher_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16) << 8 | \
            A_scale_lower_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16)
        ).cast(tl.float16, bitcast=True).to(tl.float32)
        # 量化后权重
        A_weight_idx = (A_qblock_idx + 2)[:, None] + tl.arange(0, 32)
        A_weights = tl.load(A_ptr + A_weight_idx, mask=A_mask[:, None], other=0)

        # 从 B 中读取量化块
        B_qblock_idx = (rows * bytes_in_K_direction_B) + qblock_idx_k * q8_1_block_size_bytes
        # 缩放因子
        B_scale_lower_byte_idx = B_qblock_idx
        B_scale_higher_byte_idx = B_qblock_idx + 1
        B_scale_lower_byte = tl.load(B_ptr + B_scale_lower_byte_idx, mask=B_mask, other=60)  # 低八位60，高八位0，组合起来是float16的1.0
        B_scale_higher_byte = tl.load(B_ptr + B_scale_higher_byte_idx, mask=B_mask, other=0)
        B_scale = (
            B_scale_higher_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16) << 8 | \
            B_scale_lower_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16)
        ).cast(tl.float16, bitcast=True).to(tl.float32)
        # 量化后权重
        B_weight_idx = (B_qblock_idx + 4)[:, None] + tl.arange(0, 32)  # +4 因为q8_1块内开头有4字节（d和s）
        B_weights = tl.load(B_ptr + B_weight_idx, mask=B_mask[:, None], other=0)

        # 计算点积并累加到结果中
        # c = A_scale * B_scale * sum( q_a_i * q_b_i ) for i in [0, 32)
        acc += (A_scale[None, :] * B_scale[:, None] * tl.dot(B_weights, A_weights.T)).to(tl.float16)

    # ===== 存储结果 =====
    out_ptrs = C_ptr + rows[:, None] * M + cols[None, :]
    out_mask = A_mask & B_mask[:, None]
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

    # K方向的量化块数量
    assert (K % 32 == 0)
    qblock_num_in_K_direction = int(K / 32)

    # 创建输出张量
    C = torch.empty((N, M), device=A.device, dtype=torch.float16)

    # 启动Triton内核
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    _mmq_q8_0_q8_1_triton[grid](A, B, C, M, N, K, qblock_num_in_K_direction, 34, 36, BLOCK_SIZE_M, BLOCK_SIZE_N)

    return C
