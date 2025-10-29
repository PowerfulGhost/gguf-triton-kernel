# %%
import torch


def parse_q6_k_from_bytes(bytes: torch.Tensor) -> tuple:
    """
    解析 q6_k 量化块（210 字节的 int8 张量）并返回解析结果。
    
    输入:
        bytes: torch.Tensor, shape=(210,), dtype=torch.int8
            210 字节长度的 int8 张量，表示一个 q6_k 量化块
    
    返回:
        tuple (d, scales, quant_weights)
        - d: torch.Tensor (float32, 标量)
        - scales: torch.Tensor (int8, shape=(16,))
        - quant_weights: torch.Tensor (int8, shape=(256,))
    
    注：quant_weight已经全部减去了32了
    """
    # 验证输入
    assert bytes.shape == (210, ), f"Expected 210 bytes, got {bytes.shape}"
    assert bytes.dtype == torch.int8, f"Expected torch.int8, got {bytes.dtype}"

    # 1. 加载 d (half, 2 字节) -> 转换为 float32 标量
    # 将字节解释为 uint8 以保持原始位模式，再 view 为 float16
    d = bytes[-2:].to(torch.uint8).view(torch.float16).to(torch.float32)

    # 3. 加载 scales
    scales = bytes[-2 - 16:-2]  # 转换为 uint8 以进行位操作

    # 3. 加载 ql 和 qh
    ql = bytes[:128].view(torch.uint8)
    qh = bytes[128:192].view(torch.uint8)

    # 4. 解包 ql 和 qh
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
    qs = torch.zeros((256, ), dtype=torch.int8)
    for blk_idx in range(16):
        # load lower 4bits
        if blk_idx < 8:
            ql_idx_range = (blk_idx % 4 * 16, blk_idx % 4 * 16 + 16)
            ql_mask = 0x0F if blk_idx < 4 else 0xF0
            ql_shift = 0 if blk_idx < 4 else 4
        else:
            ql_idx_range = (((blk_idx - 8) % 4 * 16) + 64, ((blk_idx - 8) % 4 * 16 + 16) + 64)
            ql_mask = 0x0F if (blk_idx - 8) < 4 else 0xF0
            ql_shift = 0 if (blk_idx - 8) < 4 else 4

        lower = (ql[ql_idx_range[0]:ql_idx_range[1]] & ql_mask) >> ql_shift

        # load higher 2bits
        mask_table = {0: 0x03, 1: 0x0C, 2: 0x30, 3: 0xC0}
        if blk_idx < 8:
            qh_idx_range = ((blk_idx % 2) * 16, (blk_idx % 2) * 16 + 16)
            qh_mask = mask_table[blk_idx // 2]
            qh_shift = (blk_idx // 2) * 2
        else:
            qh_idx_range = (((blk_idx - 8) % 2) * 16 + 32, ((blk_idx - 8) % 2) * 16 + 16 + 32)
            qh_mask = mask_table[(blk_idx - 8) // 2]
            qh_shift = ((blk_idx - 8) // 2) * 2

        higher = (qh[qh_idx_range[0]:qh_idx_range[1]] & qh_mask) >> qh_shift

        data = higher << 4 | lower

        qs_idx_range = (blk_idx * 16, blk_idx * 16 + 16)
        qs[qs_idx_range[0]:qs_idx_range[1]] |= data

    qs -= 32

    return d, scales, qs


def mmq_q6_k_q8_1_cpu(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int):
    """
    q6_k-Q8_1 量化矩阵相乘的cpu实现
    C = (A @ B.T).T

    Args:
        A: 矩阵A，int8，q6_k量化
        B: 矩阵B，int8，Q8_1量化
        M: 矩阵A的行数，int
        N: 矩阵B的行数，int
        K: 矩阵A和矩阵B的列数（应相等），int
    
    Returns:
        矩阵相乘结果，float16
    """

    q6_k_block_size_bytes = 210
    q8_1_block_size_bytes = 36

    # 确保A和B的元素数量正确
    assert K % 256 == 0
    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    assert A.numel() == M * K / 256 * q6_k_block_size_bytes
    assert B.numel() == N * K / 32 * q8_1_block_size_bytes

    A = A.cpu()
    B = B.cpu()
    C = torch.zeros((M, N), dtype=torch.float16)

    quant_block_per_row_A = K // 256
    quant_block_per_row_B = K // 32

    for m in range(M):
        for n in range(N):
            for quant_block_idx_A in range(K // 256):
                # 从A中加载量化块
                ptr_A = (quant_block_per_row_A * m + quant_block_idx_A) * q6_k_block_size_bytes
                quant_block_A = A[ptr_A:ptr_A + q6_k_block_size_bytes]
                d_A, scales_A, quant_weights_A = parse_q6_k_from_bytes(quant_block_A)

                for quant_block_idx_B in range(8):
                    # 从B中加载量化块
                    ptr_B = q8_1_block_size_bytes * (quant_block_per_row_B * n + 8 * quant_block_idx_A + quant_block_idx_B)
                    quant_block_B = B[ptr_B:ptr_B + q8_1_block_size_bytes]
                    scale_B = quant_block_B[:2].view(torch.float16)
                    scale_sum_product_B = quant_block_B[2:4].view(torch.float16)
                    quant_weights_B = quant_block_B[4:]

                    # 计算量化块相乘
                    # 2个A子块对应一个B块
                    A_subblk_1_idx = quant_block_idx_B * 2
                    A_subblk_1_scale = d_A * scales_A[A_subblk_1_idx]
                    A_subblk_1_qs = quant_weights_A[16 * A_subblk_1_idx:16 * A_subblk_1_idx + 16]

                    dot_1 = torch.dot(A_subblk_1_qs.to(torch.int32), quant_weights_B[:16].to(torch.int32))

                    A_subblk_2_idx = A_subblk_1_idx + 1
                    A_subblk_2_scale = d_A * scales_A[A_subblk_2_idx]
                    A_subblk_2_qs = quant_weights_A[16 * A_subblk_2_idx:16 * A_subblk_2_idx + 16]

                    dot_2 = torch.dot(A_subblk_2_qs.to(torch.int32), quant_weights_B[16:].to(torch.int32))

                    res = scale_B * (A_subblk_1_scale * dot_1 + A_subblk_2_scale * dot_2)

                    # 写入C
                    C[m, n] += res.item()

    return C.T


# %%
if __name__ == "__main__":
    torch.manual_seed(42)
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(root_dir)

    from utils.quantize.q8_1 import quantize_to_q8_1
    from utils.quantize.q6_k import quantize_to_q6_k

    from _utils import plot_hot_graph, calculate_snr

    for m_pow in range(0, 6, 2):
        for n_pow in range(0, 6, 2):
            for k_pow in range(8, 11):
                M, N, K = 2**m_pow, 2**n_pow, 2**k_pow
                print("=" * 20)
                print(f"{M=}, {N=}, {K=}")

                A = torch.randn((M, K), dtype=torch.float16)
                B = torch.randn((N, K), dtype=torch.float16)
                C = (A @ B.T).T

                quant_A = quantize_to_q6_k(A)
                quant_B = quantize_to_q8_1(B)
                C_need_test = mmq_q6_k_q8_1_cpu(quant_A, quant_B, M, N, K)

                noise = C - C_need_test

                snr = calculate_snr(C, noise)

                ratio_tensor = torch.abs(noise) / torch.abs(C)
                plot_hot_graph(
                    f"{M=}, {N=}, {K=}, {snr=} ",
                    ratio_tensor,
                    0,
                    1,
                )

# %%
