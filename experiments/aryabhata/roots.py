"""Digit-by-digit square root (Gaṇitapada 4) next to integer Newton.

Both return the exact integer square root, so a learner trained on the root
would see the same label either way. The comparison is cost: digit-pair
steps, how many candidate digits get tried, the bit length of the operands
in each step, and wall time measured by the caller. math.isqrt is the
current library implementation.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class ISqrtResult:
    root: int
    steps: int
    digit_trials: int
    max_operand_bits: int
    median_operand_bits: int


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    return int(statistics.median(values))


def aryabhata_isqrt(n: int) -> ISqrtResult:
    """Floor square root by digit pairs, trying candidate digits from 9 downward.

    Each step's operand size is the bit length of the remainder brought down
    so far and of the product (20*root + digit) * digit.
    """
    if n < 0:
        raise ValueError("square root of a negative integer")
    if n < 2:
        return ISqrtResult(n, 1 if n else 0, 0, n.bit_length(), n.bit_length())

    digits = str(n)
    if len(digits) % 2:
        digits = "0" + digits
    root = 0
    remainder = 0
    steps = 0
    trials = 0
    operand_bits: list[int] = []
    for start in range(0, len(digits), 2):
        steps += 1
        remainder = remainder * 100 + int(digits[start : start + 2])
        digit = 0
        product = 0
        for candidate in range(9, -1, -1):
            trials += 1
            product = (20 * root + candidate) * candidate
            operand_bits.append(max(remainder.bit_length(), product.bit_length()))
            if product <= remainder:
                digit = candidate
                break
        remainder -= (20 * root + digit) * digit
        root = root * 10 + digit
    return ISqrtResult(root, steps, trials, max(operand_bits), _median_int(operand_bits))


def digit_trials(n: int) -> int:
    """How many candidate digits the Aryabhata loop actually tests."""
    return aryabhata_isqrt(n).digit_trials


def newton_isqrt(n: int) -> ISqrtResult:
    """Integer Newton from a bit-length start.

    Each iteration's operands are the full n and the current guess. The
    quotient n // x is the wide division.
    """
    if n < 0:
        raise ValueError("square root of a negative integer")
    if n < 2:
        bits = n.bit_length()
        return ISqrtResult(n, 0, 0, bits, bits)
    x = 1 << ((n.bit_length() + 1) // 2)
    iterations = 0
    operand_bits: list[int] = []
    while True:
        iterations += 1
        quotient = n // x
        operand_bits.append(max(n.bit_length(), x.bit_length(), quotient.bit_length()))
        nxt = (x + quotient) // 2
        if nxt >= x:
            return ISqrtResult(x, iterations, 0, max(operand_bits), _median_int(operand_bits))
        x = nxt


def library_isqrt(n: int) -> int:
    return math.isqrt(n)
