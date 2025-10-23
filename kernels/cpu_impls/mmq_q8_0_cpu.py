# %%
import torch


def mmq_q8_0_cpu(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int):
    """
    q8_0量化矩阵相乘的cpu实现
    C = (A @ B.T).T

    Args:
        A: 矩阵A，int8
        B: 矩阵B，int8
        M: 矩阵A的行数，int
        N: 矩阵B的行数，int
        K: 矩阵A和矩阵B的列数（应相等），int
    
    Returns:
        矩阵相乘结果，float16
    """
    # 确保A和B的元素数量正确
    assert K % 32 == 0
    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    assert A.numel() == M * K / 32 * 34
    assert B.numel() == N * K / 32 * 34

    A = A.cpu()
    B = B.cpu()
    C = torch.zeros((M, N), dtype=torch.float16)

    quant_block_per_row = K // 32
    quant_block_size_bytes = 34

    for m in range(M):
        for n in range(N):
            for quant_block_idx in range(K // 32):
                # 从A中加载量化块
                ptr_A = (quant_block_per_row * m + quant_block_idx) * quant_block_size_bytes
                quant_block_A = A[ptr_A:ptr_A + quant_block_size_bytes]
                scale_A = quant_block_A[:2].view(torch.float16)
                quant_weights_A = quant_block_A[2:]
                # 从B中加载量化块
                ptr_B = (quant_block_per_row * n + quant_block_idx) * quant_block_size_bytes
                quant_block_B = B[ptr_B:ptr_B + quant_block_size_bytes]
                scale_B = quant_block_B[:2].view(torch.float16)
                quant_weights_B = quant_block_B[2:]
                # 计算量化块相乘
                int_dot = torch.dot(quant_weights_A.to(torch.int32), quant_weights_B.to(torch.int32))
                result = scale_A * scale_B * int_dot
                # 写入C
                C[m, n] += result.item()

    return C.T


# %%
if __name__ == "__main__":
    import sys
    sys.path.append("/workspaces/gguf-triton-kernel")

    from utils.quantize import quantize_to_q8_0
    from matplotlib import pyplot as plt

    def plot_hot_graph(title: str, tensor: torch.Tensor, vmin, vmax):
        tensor_np = tensor.detach().cpu().numpy()
        plt.imshow(tensor_np, cmap="hot", interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(title)
        plt.show()

    def calculate_snr(signal, noise):
        """
        计算信噪比 SNR (dB)
        
        参数:
            signal (torch.Tensor): 纯净信号，任意形状
            noise  (torch.Tensor): 噪声，与 signal 形状相同
        
        返回:
            snr_db (float): 信噪比，单位为 dB
        """
        # 计算信号和噪声的能量（平方和）
        signal_power = torch.sum(signal**2)
        noise_power = torch.sum(noise**2)

        # 防止除以零或 log(0)
        if noise_power == 0:
            return float('inf')  # 噪声为零，SNR 无穷大

        snr = signal_power / noise_power
        snr_db = 10 * torch.log10(snr)

        return snr_db.item()  # 返回 Python 标量数值

    for m_pow in range(0, 6, 2):
        for n_pow in range(0, 6, 2):
            for k_pow in range(5, 10, 2):
                M, N, K = 2**m_pow, 2**n_pow, 2**k_pow

                A = torch.randn((M, K), dtype=torch.float16)
                B = torch.randn((N, K), dtype=torch.float16)
                C = A @ B.T

                quant_A = quantize_to_q8_0(A)
                quant_B = quantize_to_q8_0(B)
                C_need_test = mmq_q8_0_cpu(quant_A, quant_B, M, N, K)

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
