"""Aryabhata's Rsine table and the second-difference recurrence behind it.

The printed table is Gitikapada 12 of the Aryabhatiya (499 CE): 24 first
differences of jya, at steps of 225 arcminutes (3 deg 45 arcmin). The radius
3438 arcminutes is the radian, rounded, from his own pi: a full circle is
21600 arcminutes, so R = 21600 / (2 * pi) with pi = 62832/20000.

The geometric fact he used is that the second difference of sine is
proportional to sine. In modern notation, for step h,

    delta_n - delta_{n-1} = -4 * sin(h/2)**2 * sin((n-1)*h)

which is exact. His from-scratch constants replace sin(h) with the arc h
and replace the coefficient 4*sin(h/2)**2 with 1/225 in unit-sine
coordinates at his native step. Those two numbers are close (about 0.00428
versus 1/225 = 0.00444) and that is why the table works.
"""

from __future__ import annotations

import math

# First differences, in arcminutes, in the order encoded by Gitikapada 12.
JYA_DIFFERENCES: tuple[int, ...] = (
    225,
    224,
    222,
    219,
    215,
    210,
    205,
    199,
    191,
    183,
    174,
    164,
    154,
    143,
    131,
    119,
    106,
    93,
    79,
    65,
    51,
    37,
    22,
    7,
)

RADIUS_ARCMINUTES = 3438
STEP_ARCMINUTES = 225
NATIVE_STEPS = 24
# Unit-sine form of "subtract the sine divided by the first sine".
ARYABHATA_ALPHA = 1.0 / STEP_ARCMINUTES


def aryabhata_pi() -> float:
    """Gaṇitapada 10: circumference 62832 for diameter 20000."""
    return 62832 / 20000


def radius_from_pi(pi: float | None = None) -> float:
    """Radian length in arcminutes. 21600 / (2 * pi)."""
    return 21600 / (2 * (aryabhata_pi() if pi is None else pi))


def published_jya() -> list[int]:
    """Cumulative Rsines in arcminutes. The last entry is the radius, 3438."""
    total = 0
    values: list[int] = []
    for difference in JYA_DIFFERENCES:
        total += difference
        values.append(total)
    return values


def modern_jya(radius: float = RADIUS_ARCMINUTES, steps: int = NATIVE_STEPS) -> list[float]:
    """radius * sin(k * pi / (2 * steps)) for k = 1..steps. The current definition."""
    return [radius * math.sin(k * math.pi / (2 * steps)) for k in range(1, steps + 1)]


def modern_rounded_jya(radius: int = RADIUS_ARCMINUTES, steps: int = NATIVE_STEPS) -> list[int]:
    """Same 24 nodes and arcminute precision as the printed table, filled by rounding math.sin.

    This is the lookup-table control: same size, same unit, current values.
    """
    return [int(round(value)) for value in modern_jya(radius, steps)]


def rollout(steps: int, alpha: float, delta0: float) -> list[float]:
    """Second-difference recurrence. s grows by delta, delta shrinks by alpha*s.

    Returns s after 1..steps updates. steps=0 returns an empty list.
    s_0 = 0 and the first addition uses delta0, so rollout(1) == delta0.
    """
    delta = delta0
    value = 0.0
    out: list[float] = []
    for _ in range(steps):
        value += delta
        out.append(value)
        delta -= alpha * value
    return out


def native_step_radians() -> float:
    return math.pi / (2 * NATIVE_STEPS)


def oracle_alpha(step_radians: float) -> float:
    """Exact finite-difference coefficient 4*sin(step/2)**2."""
    return 4.0 * math.sin(step_radians / 2.0) ** 2


def small_angle_alpha(step_radians: float) -> float:
    """Leading term of the exact coefficient: step**2. Needs no sine call."""
    return step_radians * step_radians


def aryabhata_unit_sines() -> list[float]:
    """His rule at the native step, in unit sine (divide arcminutes by 3438).

    delta starts at 225/3438, which is the arc in radians, and each next
    difference subtracts the current sine over 225.
    """
    delta0 = STEP_ARCMINUTES / RADIUS_ARCMINUTES
    return rollout(NATIVE_STEPS, ARYABHATA_ALPHA, delta0)


def small_angle_unit_sines(steps: int = NATIVE_STEPS) -> list[float]:
    """Same loop as Aryabhata, with the modern from-scratch constant h**2."""
    step = math.pi / (2 * steps)
    return rollout(steps, small_angle_alpha(step), step)


def oracle_unit_sines(steps: int = NATIVE_STEPS) -> list[float]:
    """Same loop with exact constants. This calls math.sin to set them.

    It is the identity behind the table, not an independent way to discover
    sine: the first step is math.sin(h) and the coefficient uses math.sin too.
    """
    step = math.pi / (2 * steps)
    return rollout(steps, oracle_alpha(step), math.sin(step))
