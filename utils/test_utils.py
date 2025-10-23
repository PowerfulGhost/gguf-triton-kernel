import torch


def allclose(a, b, atol_ratio=0.01):
    if not torch.isnan(torch.max(torch.abs(b))):
        return torch.allclose(a, b, atol=atol_ratio * torch.max(torch.abs(b)))
    else:
        return False


if __name__ == "__main__":
    a = torch.tensor