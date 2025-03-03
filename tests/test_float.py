# NOTE: this is just to check that the representation of the IEEE floating-point standard in the
# paper is correct - it is not testing any aspect of the implementation
import itertools
import struct

import numpy as np

Bits = tuple[int, ...]


def float_bits(f: float) -> Bits:
    # extract bytes from float and return bits as tuple of ints
    bytes = tuple("{:0>8b}".format(c) for c in struct.pack("d", f))
    bits = tuple(int(bit) for byte in itertools.chain(bytes) for bit in reversed(byte))
    return bits


def sign(bits: Bits) -> int:
    return (-1) ** bits[63]


def exponent(bits: Bits) -> int:
    return -1023 + sum(bits[52 + i] * 2**i for i in range(10 + 1))


def significand(bits: Bits) -> float:
    return sum(bits[i] * 2 ** (i - 52) for i in range(51 + 1))


def test_float_representation():
    for f in [0.001, -543.1231, 99.9999, 242341.0, 1.0, 9000.0]:
        bits = float_bits(f)
        assert sign(bits) * (1 + significand(bits)) * 2 ** exponent(bits) == f


def test_float_conversion():
    truncated_max = 2**64 - 1 >> 11  # truncate max unsigned integer to 53 bits
    assert truncated_max * 2 ** (-53) == np.nextafter(1.0, -1) == 1 - 2 ** (-53)  # last 64-bit float below 1.0

    truncated_mid = (2**64 // 2) >> 11  # truncate medium unsigned integer to 53 bits
    assert truncated_mid * 2 ** (-53) == 0.5

    truncated_min = 0 >> 11  # truncate min unsigned integer to 53 bits (will still equal zero)
    assert truncated_min * 2 ** (-53) == 0.0  # 0 * ... = 0.0
