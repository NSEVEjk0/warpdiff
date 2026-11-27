"""FP32 RMSNorm over the hidden dimension of a batch of tokens."""

import triton
import triton.language as tl


@triton.jit
def _inverse_rms(source, inverse_rms, columns: tl.constexpr, block: tl.constexpr, epsilon: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, block)
    value = tl.load(source + row * columns + column, column < columns, other=0)
    inverse = tl.rsqrt(tl.sum(value * value, axis=0) / columns + epsilon)
    tl.store(inverse_rms + row, inverse)


@triton.jit
def _scale_rows(source, weight, output, inverse_rms, size, columns: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < size
    value = tl.load(source + index, mask, other=0)
    inverse = tl.load(inverse_rms + index // columns, mask, other=0)
    scale = tl.load(weight + index % columns, mask, other=0)
    tl.store(output + index, value * inverse * scale, mask)


def rmsnorm(source, weight, output, inverse_rms, epsilon):
    rows, columns = source.shape
    _inverse_rms[(rows,)](source, inverse_rms, columns, triton.next_power_of_2(columns), epsilon)
    _scale_rows[(triton.cdiv(rows * columns, 512),)](source, weight, output, inverse_rms, rows * columns, columns, 512)
