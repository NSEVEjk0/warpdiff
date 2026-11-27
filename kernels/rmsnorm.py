"""FP32 RMSNorm over the hidden dimension of a batch of tokens."""

import triton
import triton.language as tl


@triton.jit
def _normalize(source, weight, output, columns: tl.constexpr, block: tl.constexpr, epsilon: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, block)
    index = row * columns + column
    mask = column < columns
    value = tl.load(source + index, mask, other=0)
    inverse = tl.rsqrt(tl.sum(value * value, axis=0) / columns + epsilon)
    scale = tl.load(weight + column, mask, other=0)
    tl.store(output + index, value * inverse * scale, mask)


def rmsnorm(source, weight, output, _inverse_rms_buffer, epsilon):
    rows, columns = source.shape
    _normalize[(rows,)](source, weight, output, columns, triton.next_power_of_2(columns), epsilon)
