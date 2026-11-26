import torch
import triton
import triton.language as tl


@triton.jit
def _softmax(source, output, row_count, width: tl.constexpr, tile: tl.constexpr, rows: tl.constexpr):
    row = tl.program_id(0) * rows + tl.arange(0, rows)
    col = tl.arange(0, tile)
    index = row[:, None] * width + col[None, :]
    mask = (row[:, None] < row_count) & (col[None, :] < width)
    values = tl.load(source + index, mask=mask, other=-float("inf"))
    shifted = values - tl.max(values, axis=1)[:, None]
    numerator = tl.exp(shifted)
    result = numerator / tl.sum(numerator, axis=1)[:, None]
    tl.store(output + index, result, mask=mask)


def softmax(source: torch.Tensor, output: torch.Tensor) -> None:
    """Write a row-wise FP32 softmax into the supplied output tensor."""
    row_count, width = source.shape
    tile = triton.next_power_of_2(width)
    rows = 1
    _softmax[(triton.cdiv(row_count, rows),)](source, output, row_count, width, tile, rows, num_warps=4)
