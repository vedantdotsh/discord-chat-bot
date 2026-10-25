"""Kuttaka, the pulverizer, next to the extended Euclidean algorithm.

Gaṇitapada 32-33 solves ax + c = by in integers by continued division and a
column of quotients built from the bottom. That column is the same
arithmetic as the modern extended Euclidean algorithm. This module keeps
the column construction, checks the sign by substitution, and counts
divisions so the two can be compared on identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearSolution:
    gcd: int
    x: int
    y: int
    divisions: int

    def check(self, a: int, b: int) -> int:
        return a * self.x + b * self.y


def extended_euclid(a: int, b: int) -> LinearSolution:
    """Return gcd, x, y and the division count for a*x + b*y = gcd."""
    a, b = int(a), int(b)
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    divisions = 0
    while r != 0:
        q = old_r // r
        divisions += 1
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return LinearSolution(abs(old_r), old_s, old_t, divisions)


def kuttaka(a: int, b: int) -> LinearSolution:
    """Pulverizer column for a*x + b*y = gcd(a, b).

    Quotients are stored while the remainders are produced, then a coefficient
    column is filled from the bottom: each entry is the next coefficient
    times the quotient on this row, plus the coefficient below that.
    The worked case a=414, b=189 yields x=-5, y=11, gcd=9
    (11*189 - 5*414 = 9), which is the column in Sathaye's notes on the
    Aryabhatiya rule.
    """
    a, b = int(a), int(b)
    if a == 0 and b == 0:
        raise ValueError("gcd(0, 0) is undefined")

    remainders = [a, b]
    quotients: list[int] = []
    while remainders[-1] != 0:
        quotients.append(remainders[-2] // remainders[-1])
        remainders.append(remainders[-2] - quotients[-1] * remainders[-1])
    divisions = len(quotients)
    gcd = abs(remainders[-2])

    if not quotients:
        # b == 0. a*1 + b*0 = a.
        x = 1 if a >= 0 else -1
        return LinearSolution(abs(a), x, 0, 0)

    column = [0] * len(remainders)
    column[-1] = 1
    column[-2] = 0
    for index in range(len(remainders) - 3, -1, -1):
        column[index] = column[index + 1] * quotients[index] + column[index + 2]

    # column[0] * b - column[1] * a = ±gcd for the non-negative example.
    x = -column[1]
    y = column[0]
    if a * x + b * y == -gcd:
        x, y = -x, -y
    elif a * x + b * y != gcd:
        # Sign conventions differ when a leading remainder is negative.
        # Fall through to a direct search of the four sign pairs before giving up.
        found = False
        for sx in (1, -1):
            for sy in (1, -1):
                if a * (sx * x) + b * (sy * y) == gcd:
                    x, y = sx * x, sy * y
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError(f"kuttaka column failed for a={a}, b={b}")
    return LinearSolution(gcd, x, y, divisions)


def solve(a: int, b: int, c: int, method: str = "kuttaka") -> LinearSolution:
    """Scale a Bézout solution so a*x + b*y = c. Raises if gcd does not divide c."""
    base = kuttaka(a, b) if method == "kuttaka" else extended_euclid(a, b)
    if c % base.gcd != 0:
        raise ValueError(f"{base.gcd} does not divide {c}")
    scale = c // base.gcd
    return LinearSolution(base.gcd, base.x * scale, base.y * scale, base.divisions)
