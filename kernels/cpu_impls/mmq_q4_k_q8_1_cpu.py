# %%
import torch


def parse_q4_k_from_bytes(bytes: torch.Tensor) -> tuple:
    """
    解析 q4_k 量化块（144 字节的 int8 张量）并返回解析结果。
    
    输入:
        bytes: torch.Tensor, shape=(144,), dtype=torch.int8
            144 字节长度的 int8 张量，表示一个 q4_k 量化块
    
    返回:
        tuple (d, dmin, scales, mins, quant_weights)
        - d: torch.Tensor (float32, 标量)
        - dmin: torch.Tensor (float32, 标量)
        - scales: torch.Tensor (float32, shape=(8,))
        - mins: torch.Tensor (float32, shape=(8,))
        - quant_weights: torch.Tensor (uint8, shape=(256,))
    """
    # 验证输入
    assert bytes.shape == (144, ), f"Expected 144 bytes, got {bytes.shape}"
    assert bytes.dtype == torch.int8, f"Expected torch.int8, got {bytes.dtype}"

    # 1. 解析 d (half, 2 字节) -> 转换为 float32 标量
    # 将字节解释为 uint8 以保持原始位模式，再 view 为 float16
    d = bytes[0:2].to(torch.uint8).view(torch.float16).to(torch.float32)

    # 2. 解析 dmin (half, 2 字节) -> 转换为 float32 标量
    dmin = bytes[2:4].to(torch.uint8).view(torch.float16).to(torch.float32)

    # 3. 解析 scales[12] (uint8[12])
    scales_arr = bytes[4:16].to(torch.uint8)  # 转换为 uint8 以进行位操作

    # 按结构拆分 scales 数组 (参考 C 结构和反量化实现)
    d_arr = scales_arr[0:4]  # 索引 0-3: EEAAAAAA, FFBBBBBB, GGCCCCCC, HHDDDDDD
    m_arr = scales_arr[4:8]  # 索引 4-7: eeaaaaaa, ffbbbbbb, ggcccccc, hhdddddd
    m_d_arr = scales_arr[8:12]  # 索引 8-11: eeeeEEEE, ffffFFFF, ggggGGGG, hhhhHHHH

    # 计算 scales 部分 (8 个值)
    sc_part1 = d_arr & 0x3F  # 低 6 位: AAAAAA, BBBBBB, CCCCCC, DDDDDD
    sc_part2 = (m_d_arr & 0x0F) | ((d_arr >> 2) & 0x30)  # 组合 EEEE/eeee 部分
    scales_out = torch.cat([sc_part1, sc_part2])  # 拼接为 8 个值

    # 计算 mins 部分 (8 个值)
    min_part1 = m_arr & 0x3F  # 低 6 位: aaaaaa, bbbbbb, cccccc, dddddd
    min_part2 = (m_d_arr >> 4) | ((m_arr >> 2) & 0x30)  # 组合 FFFF/ffff 部分
    mins_out = torch.cat([min_part1, min_part2])  # 拼接为 8 个值

    # 转换为 float32 张量
    scales_out = scales_out.to(torch.float32)
    mins_out = mins_out.to(torch.float32)

    # 4. 解析 qs[128] (4-bit 量化权重)
    qs = bytes[16:144].to(torch.uint8).view(4, 32)  # 128字节 -> (4, 32)结构
    quant_weights = torch.stack([qs & 0x0F, (qs >> 4) & 0x0F], dim=1).reshape(256)

    return (d, dmin, scales_out, mins_out, quant_weights)


def mmq_q4_k_q8_1_cpu(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int):
    """
    Q4_K-Q8_1 量化矩阵相乘的cpu实现
    C = (A @ B.T).T

    Args:
        A: 矩阵A，int8，Q4_K量化
        B: 矩阵B，int8，Q8_1量化
        M: 矩阵A的行数，int
        N: 矩阵B的行数，int
        K: 矩阵A和矩阵B的列数（应相等），int
    
    Returns:
        矩阵相乘结果，float16
    """

    q4_k_block_size_bytes = 144
    q8_1_block_size_bytes = 36

    # 确保A和B的元素数量正确
    assert K % 256 == 0
    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    assert A.numel() == M * K / 256 * q4_k_block_size_bytes
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
                ptr_A = (quant_block_per_row_A * m + quant_block_idx_A) * q4_k_block_size_bytes
                quant_block_A = A[ptr_A:ptr_A + q4_k_block_size_bytes]
                d_A, dmin_A, scales_A, mins_A, quant_weights_A = parse_q4_k_from_bytes(quant_block_A)

                for quant_block_idx_B in range(8):
                    # 从B中加载量化块
                    ptr_B = q8_1_block_size_bytes * (quant_block_per_row_B * n + 8 * quant_block_idx_A + quant_block_idx_B)
                    quant_block_B = B[ptr_B:ptr_B + q8_1_block_size_bytes]
                    scale_B = quant_block_B[:2].view(torch.float16)
                    scale_sum_product_B = quant_block_B[2:4].view(torch.float16)
                    quant_weights_B = quant_block_B[4:]

                    # 计算量化块相乘
                    quant_weights_A_subblock = quant_weights_A[32 * quant_block_idx_B:32 * (quant_block_idx_B + 1)]
                    dot_result = \
                        d_A * scales_A[quant_block_idx_B] * scale_B * torch.dot(quant_weights_A_subblock.to(torch.int32), quant_weights_B.to(torch.int32)) \
                        - dmin_A*mins_A[quant_block_idx_B] * scale_sum_product_B

                    # 写入C
                    C[m, n] += dot_result.item()

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
    from utils.quantize.q4_k import quantize_to_q4_k

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

                quant_A = quantize_to_q4_k(A)
                quant_B = quantize_to_q8_1(B)
                C_need_test = mmq_q4_k_q8_1_cpu(quant_A, quant_B, M, N, K)

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
