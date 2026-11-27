"""Compare a Triton RMSNorm kernel with a PyTorch reference."""

import torch


def prepare(case):
    rows, columns = case["shape"]
    epsilon = case["epsilon"]
    source = torch.randn((rows, columns), device="cuda", dtype=torch.float32)
    weight = torch.randn(columns, device="cuda", dtype=source.dtype)
    expected = source * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + epsilon) * weight

    def bind(variant):
        output = torch.empty_like(expected)
        inverse_rms = torch.empty(rows, device=source.device, dtype=source.dtype)

        def run():
            variant.rmsnorm(source, weight, output, inverse_rms, epsilon)

        return run, output

    return expected, bind
