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
    """
    bytes_in_K_direction_A = q4_k_block_size_bytes * qblock_num_in_K_direction_A
    bytes_in_K_direction_B = q8_1_block_size_bytes * qblock_num_in_K_direction_B

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
        A_qblock_idx = (cols * bytes_in_K_direction) + qblock_idx_k * block_size_bytes
        # 缩放因子
        A_scale_lower_byte_idx = A_qblock_idx
        A_scale_higher_byte_idx = A_qblock_idx + 1
        A_scale_lower_byte = tl.load(A_ptr + A_scale_lower_byte_idx, mask=A_mask, other=60)  # 低八位60，高八位0，组合起来是float16的1.0
        A_scale_higher_byte = tl.load(A_ptr + A_scale_higher_byte_idx, mask=A_mask, other=0)
        A_scale = (
            A_scale_higher_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16) << 8 | \
            A_scale_lower_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16)
        ).cast(tl.float16, bitcast=True).to(tl.float32)
        # 权重
        A_weight_idx = (A_qblock_idx + 2)[:, None] + tl.arange(0, 32)
        A_weights = tl.load(A_ptr + A_weight_idx, mask=A_mask[:, None], other=0)

        # 从 B 中读取量化块
        B_qblock_idx = (rows * bytes_in_K_direction) + qblock_idx_k * block_size_bytes
        # 缩放因子
        B_scale_lower_byte_idx = B_qblock_idx
        B_scale_higher_byte_idx = B_qblock_idx + 1
        B_scale_lower_byte = tl.load(B_ptr + B_scale_lower_byte_idx, mask=B_mask, other=60)  # 低八位60，高八位0，组合起来是float16的1.0
        B_scale_higher_byte = tl.load(B_ptr + B_scale_higher_byte_idx, mask=B_mask, other=0)
        B_scale = (
            B_scale_higher_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16) << 8 | \
            B_scale_lower_byte.cast(tl.uint8, bitcast=True).cast(tl.uint16)
        ).cast(tl.float16, bitcast=True).to(tl.float32)
        # 权重
        B_weight_idx = (B_qblock_idx + 2)[:, None] + tl.arange(0, 32)
        B_weights = tl.load(B_ptr + B_weight_idx, mask=B_mask[:, None], other=0)

        # 计算这两个量化块的点积
        dot_result = vec_dot_q8_0_q8_0(
            A_weights,
            A_scale,
            B_weights,
            B_scale,
        ).to(tl.float16).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)

        # 累加到结果中
        acc += dot_result

    # ===== 存储结果 =====
    out_ptrs = C_ptr + rows[:, None] * M + cols[None, :]
    out_mask = A_mask & B_mask[:, None]
    tl.store(out_ptrs, acc, mask=out_mask)


# 辅助函数：启动Triton内核
def mmq_q4_k(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    """
    执行Q4_K-Q8_0量化矩阵乘法: out = (A @ B.T).T
    
    Args:
        A: Q8_0 量化格式
        B: Q8_0 量化格式
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

    mul_mat_q4_k_q8_1_triton[grid](A, B, C, M, N, K, qblock_num_in_K_direction, BLOCK_SIZE_M, BLOCK_SIZE_N)

    return C
