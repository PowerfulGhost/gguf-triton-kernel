# %%
import torch


def mmq_q8_0_q8_1_cpu(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int):
    """
    Q8_0-Q8_1 量化矩阵相乘的cpu实现
    C = (A @ B.T).T

    Args:
        A: 矩阵A，int8，Q8_0量化
        B: 矩阵B，int8，Q8_1量化
        M: 矩阵A的行数，int
        N: 矩阵B的行数，int
        K: 矩阵A和矩阵B的列数（应相等），int
    
    Returns:
        矩阵相乘结果，float16
    """

    q8_0_block_size_bytes = 34
    q8_1_block_size_bytes = 36

    # 确保A和B的元素数量正确
    assert K % 32 == 0
    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    assert A.numel() == M * K / 32 * q8_0_block_size_bytes
    assert B.numel() == N * K / 32 * q8_1_block_size_bytes

    A = A.cpu()
    B = B.cpu()
    C = torch.zeros((M, N), dtype=torch.float16)

    quant_block_per_row = K // 32

    for m in range(M):
        for n in range(N):
            for quant_block_idx in range(K // 32):
                # 从A中加载量化块
                ptr_A = (quant_block_per_row * m + quant_block_idx) * q8_0_block_size_bytes
                quant_block_A = A[ptr_A:ptr_A + q8_0_block_size_bytes]
                scale_A = quant_block_A[:2].view(torch.float16)
                quant_weights_A = quant_block_A[2:]
                # 从B中加载量化块
                ptr_B = (quant_block_per_row * n + quant_block_idx) * q8_1_block_size_bytes
                quant_block_B = B[ptr_B:ptr_B + q8_1_block_size_bytes]
                scale_B = quant_block_B[:2].view(torch.float16)
                quant_weights_B = quant_block_B[4:]
                # 计算量化块相乘
                int_dot = torch.dot(quant_weights_A.to(torch.int32), quant_weights_B.to(torch.int32))
                result = scale_A * scale_B * int_dot
                # 写入C
                C[m, n] += result.item()

    return C.T


# %%
if __name__ == "__main__":
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(root_dir)

    from utils.quantize.q8_0 import quantize_to_q8_0
    from utils.quantize.q8_1 import quantize_to_q8_1

    from matplotlib import pyplot as plt

    from _utils import plot_hot_graph, calculate_snr

    for m_pow in range(0, 6, 2):
        for n_pow in range(0, 6, 2):
            for k_pow in range(5, 10, 2):
                M, N, K = 2**m_pow, 2**n_pow, 2**k_pow

                A = torch.randn((M, K), dtype=torch.float16)
                B = torch.randn((N, K), dtype=torch.float16)
                C = (A @ B.T).T

                quant_A = quantize_to_q8_0(A)
                quant_B = quantize_to_q8_1(B)
                C_need_test = mmq_q8_0_q8_1_cpu(quant_A, quant_B, M, N, K)

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
