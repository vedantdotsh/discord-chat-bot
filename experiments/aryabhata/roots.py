"""Digit-by-digit square root (Gaṇitapada 4) next to integer Newton.

Both return the exact integer square root. The comparison is the number of
iterations, not the accuracy. Aryabhata's method takes one step per pair of
decimal digits and, inside each step, tries digits from 9 downward. Newton,
started from the bit length, roughly doubles the number of correct bits
each iteration. math.isqrt is the current library implementation.
"""

from __future__ import annotations

import math


def aryabhata_isqrt(n: int) -> tuple[int, int]:
    """Return (floor(sqrt(n)), digit-pair steps).

    A step is one pair of decimal digits. The digit search inside the step
    is extra work and is counted separately by digit_trials().
    """
    if n < 0:
        raise ValueError("square root of a negative integer")
    if n < 2:
        return n, 1 if n else 0

    digits = str(n)
    if len(digits) % 2:
        digits = "0" + digits
    root = 0
    remainder = 0
    steps = 0
    for start in range(0, len(digits), 2):
        steps += 1
        remainder = remainder * 100 + int(digits[start : start + 2])
        # Largest digit x with (20*root + x) * x <= remainder.
        digit = 0
        for candidate in range(9, -1, -1):
            if (20 * root + candidate) * candidate <= remainder:
                digit = candidate
                break
        remainder -= (20 * root + digit) * digit
        root = root * 10 + digit
    return root, steps


def digit_trials(n: int) -> int:
    """How many candidate digits the Aryabhata loop actually tests."""
    if n < 2:
        return 0
    digits = str(n)
    if len(digits) % 2:
        digits = "0" + digits
    root = 0
    remainder = 0
    trials = 0
    for start in range(0, len(digits), 2):
        remainder = remainder * 100 + int(digits[start : start + 2])
        digit = 0
        for candidate in range(9, -1, -1):
            trials += 1
            if (20 * root + candidate) * candidate <= remainder:
                digit = candidate
                break
        remainder -= (20 * root + digit) * digit
        root = root * 10 + digit
    return trials


def newton_isqrt(n: int) -> tuple[int, int]:
    """Integer Newton from a bit-length start. Returns (root, iterations)."""
    if n < 0:
        raise ValueError("square root of a negative integer")
    if n < 2:
        return n, 0
    x = 1 << ((n.bit_length() + 1) // 2)
    iterations = 0
    while True:
        iterations += 1
        nxt = (x + n // x) // 2
        if nxt >= x:
            return x, iterations
        x = nxt


def library_isqrt(n: int) -> int:
    return math.isqrt(n)
