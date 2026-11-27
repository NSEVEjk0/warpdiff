"""Supply the input, reference and callable for a softmax case."""

import torch


def prepare(case):
    source = torch.randn(case["shape"], device="cuda", dtype=torch.float32) * 2
    expected = torch.softmax(source, dim=-1)

    def bind(variant):
        output = torch.empty_like(expected)

        def run():
            variant.softmax(source, output)

        return run, output

    return expected, bind
