import torch
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
