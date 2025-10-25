import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import torch

from utils.quantize.q8_0 import quantize_to_q8_0
from utils.quantize.q8_1 import quantize_to_q8_1
from utils.test_utils import allclose

from kernels.mmq_q8_0_q8_1 import mmq_q8_0_q8_1
from kernels.cpu_impls.mmq_q8_0_q8_1_cpu import mmq_q8_0_q8_1_cpu

mnk_list = []
for m_pow in range(0, 6, 2):
    for n_pow in range(0, 6, 2):
        for k_pow in range(5, 10, 1):
            mnk_list.append((2**m_pow, 2**n_pow, 2**k_pow))

for M, N, K in mnk_list:
    print("=" * 20)
    print(f"{M=}, {N=}, {K=} ", end="")

    float_A = torch.randn(M, K, dtype=torch.float16)
    float_B = torch.randn(N, K, dtype=torch.float16)

    quant_A = quantize_to_q8_0(float_A)
    quant_B = quantize_to_q8_1(float_B)

    C_cpu = mmq_q8_0_q8_1_cpu(quant_A, quant_B, M, N, K)
    C_triton = mmq_q8_0_q8_1(quant_A.to("cuda:0"), quant_B.to("cuda:0"), M, N, K)

    # if M==1 and N==4 and K==32:
    #     print(f"{C_cpu=}")
    #     print(f"{C_triton=}")

    if allclose(C_cpu, C_triton.cpu(), 0.01):
        print("PASS")
    else:
        print("FAILED")
