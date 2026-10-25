"""Recurrence robustness on two frozen training sets.

The integer-step recurrence can match sine on [0, 30, 60, 90] at more than
one frequency. Those aliases share a training loss, down to float dust once
noise is added, and they disagree on the other integer degrees. Fits are
ranked by training loss only. Starts within TIE_ABS of the lowest loss are
one tie. The earliest fixed start is the tie-break and is not an identified
frequency. The raw float minimum is stored separately because dust can point
at a wrong alias. Test error is recorded afterwards and never used to pick.

Half-degree scores are linear interpolation between the integer-step values.
They are not produced by casting the angle to int.

|prediction - sin| * 3438 is a scaled sine-coordinate error. It is not an
angle in arcminutes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from experiments.aryabhata.approx import _nelder_mead
from experiments.aryabhata.jya import rollout

H = math.pi / 180.0
EVEN_TRAIN = (0, 30, 60, 90)
IRREGULAR_TRAIN = (0, 29, 61, 90)
EXCLUDED = frozenset(EVEN_TRAIN + IRREGULAR_TRAIN)
HELD_OUT = tuple(k for k in range(0, 91) if k not in EXCLUDED)
HALF_DEGREES = tuple(k + 0.5 for k in range(0, 90))
NOISE_LEVELS = (0.0, 0.0001, 0.001)
# One frozen standard-normal vector per training set. Levels scale this vector.
NOISE_SEED = 0
TIE_ABS = 1e-18

RESULTS = Path(__file__).resolve().parent / "results" / "robustness.json"


def family_start(step_radians: float) -> tuple[float, float]:
    """Parameters that make one step of the recurrence equal sin(step)."""
    alpha = 4.0 * math.sin(step_radians / 2.0) ** 2
    delta0 = math.sin(step_radians)
    return alpha, delta0


def fixed_starts() -> list[dict]:
    """Starts fixed before any test score. n=0 is the 1° frequency."""
    starts = []
    for n in (0, 1, 2, 3, 4, 5, -1, -2):
        omega = H + (2.0 * math.pi * n) / 30.0
        alpha, delta0 = family_start(omega)
        starts.append(
            {
                "name": f"alias_n{n}",
                "omega": omega,
                "alpha": alpha,
                "delta0": delta0,
            }
        )
    alpha_h, _ = family_start(H)
    starts.extend(
        [
            {"name": "arc_init", "omega": H, "alpha": alpha_h, "delta0": H},
            {"name": "alpha_0.05_delta_0.05", "omega": None, "alpha": 0.05, "delta0": 0.05},
            {"name": "alpha_1_delta_0.5", "omega": None, "alpha": 1.0, "delta0": 0.5},
            {"name": "alpha_2.5_delta_0.01", "omega": None, "alpha": 2.5, "delta0": 0.01},
            {
                "name": "historical_1_over_225",
                "omega": None,
                "alpha": 1.0 / 225.0,
                "delta0": 225.0 / 3438.0,
            },
        ]
    )
    return starts


def integer_sequence(alpha: float, delta0: float, last: int = 90) -> list[float] | None:
    """Values at integer degrees 0..last. None if the rollout overflows."""
    if not math.isfinite(alpha) or not math.isfinite(delta0):
        return None
    try:
        tail = rollout(last, alpha, delta0)
    except OverflowError:
        return None
    if not all(math.isfinite(v) and abs(v) < 1e6 for v in tail):
        return None
    return [0.0, *tail]


def values_at(seq: list[float], degrees: tuple[int, ...] | tuple[float, ...]) -> np.ndarray:
    return np.array([seq[int(k)] for k in degrees], dtype=float)


def interpolate_half_degrees(seq: list[float]) -> np.ndarray:
    """Linear interpolation. Labeled as interpolation, not as a recurrence step."""
    out = []
    for degree in HALF_DEGREES:
        lo = math.floor(degree)
        hi = lo + 1
        out.append(0.5 * (seq[lo] + seq[hi]))
    return np.array(out, dtype=float)


def train_loss(alpha: float, delta0: float, degrees: tuple[int, ...], targets: np.ndarray) -> float:
    seq = integer_sequence(alpha, delta0)
    if seq is None:
        return 1e6
    err = values_at(seq, degrees) - targets
    return float(np.mean(err * err))


def local_fit(
    alpha0: float, delta0: float, degrees: tuple[int, ...], targets: np.ndarray
) -> tuple[float, float, float]:
    """Nelder-Mead from a fixed start. Expansion can walk into a nearby alias.

    The search is not given 4*sin(h/2)^2 or sin(h). A start already near an
    alias frequency can land on that frequency. The step size does not by
    itself keep the fit on the 1° solution.
    """

    def loss(point: np.ndarray) -> float:
        return train_loss(float(point[0]), float(point[1]), degrees, targets)

    start = np.array([alpha0, delta0], dtype=float)
    point, score = _nelder_mead(loss, start, step=1e-4, max_iter=80)
    return float(point[0]), float(point[1]), float(score)


def implied_step(alpha: float) -> float | None:
    """Radians per integer step, when the recurrence oscillates: cos ω = 1 - alpha/2."""
    cosine = 1.0 - alpha / 2.0
    if cosine < -1.0 or cosine > 1.0 or not math.isfinite(cosine):
        return None
    return float(math.acos(cosine))


def score_pair(alpha: float, delta0: float) -> dict:
    seq = integer_sequence(alpha, delta0)
    if seq is None:
        return {
            "heldout_integer_mae": None,
            "scaled_sine_coordinate_error": None,
            "half_degree_interpolation_mae": None,
        }
    held = values_at(seq, HELD_OUT)
    truth = np.sin(np.deg2rad(np.array(HELD_OUT, dtype=float)))
    mae = float(np.mean(np.abs(held - truth)))
    half = interpolate_half_degrees(seq)
    half_truth = np.sin(np.deg2rad(np.array(HALF_DEGREES, dtype=float)))
    return {
        "heldout_integer_mae": mae,
        "scaled_sine_coordinate_error": mae * 3438.0,
        "half_degree_interpolation_mae": float(np.mean(np.abs(half - half_truth))),
    }


def raw_argmin_index(rows: list[dict]) -> int:
    """Float minimum of training loss. Dust of order 1e-21 can pick any alias."""
    return min(range(len(rows)), key=lambda i: (rows[i]["train_loss_after"], i))


def tie_indices(rows: list[dict]) -> list[int]:
    """Starts whose training loss is within TIE_ABS of the lowest.

    On the even grid with noise, the alias family differs by about 1e-21.
    That is inside this tolerance. A loss 1e-12 higher is outside it.
    """
    best = min(row["train_loss_after"] for row in rows)
    return [i for i, row in enumerate(rows) if row["train_loss_after"] - best <= TIE_ABS]


def select_index(rows: list[dict]) -> int:
    """Earliest start within TIE_ABS of the lowest training loss.

    List order is the tie-break. It is not evidence that training loss
    identified that frequency. Test columns are ignored.
    """
    return tie_indices(rows)[0]


def frozen_targets() -> dict:
    rng = np.random.default_rng(NOISE_SEED)
    return {
        "even": rng.standard_normal(len(EVEN_TRAIN)),
        "irregular": rng.standard_normal(len(IRREGULAR_TRAIN)),
    }


def run_condition(
    name: str,
    degrees: tuple[int, ...],
    noise: np.ndarray,
    sigma: float,
    starts: list[dict],
) -> dict:
    clean = np.sin(np.deg2rad(np.array(degrees, dtype=float)))
    targets = clean + sigma * noise
    rows = []
    for start in starts:
        before = train_loss(start["alpha"], start["delta0"], degrees, targets)
        alpha, delta0, after = local_fit(start["alpha"], start["delta0"], degrees, targets)
        row = {
            "start": start["name"],
            "start_alpha": start["alpha"],
            "start_delta0": start["delta0"],
            "train_loss_at_start": before,
            "alpha": alpha,
            "delta0": delta0,
            "train_loss_after": after,
            "implied_step_radians": implied_step(alpha),
            "implied_step_degrees": None
            if implied_step(alpha) is None
            else implied_step(alpha) * 180.0 / math.pi,
        }
        row.update(score_pair(alpha, delta0))
        rows.append(row)
    tied = tie_indices(rows)
    chosen = tied[0]
    raw = raw_argmin_index(rows)
    tied_losses = [rows[i]["train_loss_after"] for i in tied]
    exact = [row["start"] for row in rows if row["train_loss_at_start"] <= 1e-12]
    return {
        "training_degrees": list(degrees),
        "sigma": sigma,
        "noise": [float(v) for v in noise],
        "targets": [float(v) for v in targets],
        "chosen_by_train_loss": rows[chosen]["start"],
        "selection_unique": len(tied) == 1,
        "tied_by_train_loss": [rows[i]["start"] for i in tied],
        "raw_argmin_train_loss": rows[raw]["start"],
        "train_loss_spread_among_tied": max(tied_losses) - min(tied_losses),
        "tie_tolerance": TIE_ABS,
        "exact_fits_at_start": exact,
        "starts": rows,
    }


def build_report() -> dict:
    noises = frozen_targets()
    starts = fixed_starts()
    conditions = []
    for set_name, degrees, noise in (
        ("even", EVEN_TRAIN, noises["even"]),
        ("irregular", IRREGULAR_TRAIN, noises["irregular"]),
    ):
        for sigma in NOISE_LEVELS:
            conditions.append(
                {
                    "training_set": set_name,
                    **run_condition(set_name, degrees, noise, sigma, starts),
                }
            )
    return {
        "protocol": {
            "training_sets": {"even": list(EVEN_TRAIN), "irregular": list(IRREGULAR_TRAIN)},
            "held_out_integer_degrees": list(HELD_OUT),
            "half_degrees": "0.5, 1.5, ..., 89.5",
            "half_degree_method": (
                "Linear interpolation between recurrence values at the bracketing "
                "integer degrees. Not int() truncation."
            ),
            "noise": (
                f"targets = sin(degrees) + sigma * z, sigma in {list(NOISE_LEVELS)}. "
                f"z is one standard-normal vector per training set from Generator({NOISE_SEED})."
            ),
            "selection": (
                "Minimize training mean square error after local Nelder-Mead. "
                f"Starts within {TIE_ABS} of the lowest loss are one tie. "
                "chosen_by_train_loss is the earliest fixed start in that tie "
                "and is list order, not an identified frequency. "
                "raw_argmin_train_loss is the float minimum and can differ by dust. "
                "Test error is not an input."
            ),
            "scaled_sine_coordinate_error": (
                "held-out mean |prediction - sin| multiplied by 3438. "
                "This is a scaled sine coordinate, not an angle in arcminutes."
            ),
            "integer_grid": (
                "A near-zero error on integer degrees is the recurrence evaluated "
                "on its own step. It does not transfer to half degrees."
            ),
            "root_benchmark": (
                "All-nines operands favor the descending digit search, because 9 "
                "is tried first and succeeds. This pass does not repeat that timing."
            ),
        },
        "conditions": conditions,
    }


def main() -> None:
    report = build_report()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS}")
    for cond in report["conditions"]:
        chosen = cond["chosen_by_train_loss"]
        row = next(item for item in cond["starts"] if item["start"] == chosen)
        one_deg = next(item for item in cond["starts"] if item["start"] == "alias_n0")
        print(
            f"{cond['training_set']} sigma={cond['sigma']:<8} "
            f"unique={str(cond['selection_unique']):<5} "
            f"earliest_tie={chosen:<16} "
            f"raw_argmin={cond['raw_argmin_train_loss']:<16} "
            f"n_tied={len(cond['tied_by_train_loss']):<2} "
            f"train={row['train_loss_after']:.3e} "
            f"spread={cond['train_loss_spread_among_tied']:.3e} "
            f"one_deg_heldout={one_deg['heldout_integer_mae']:.3e} "
            f"one_deg_half={one_deg['half_degree_interpolation_mae']:.3e}"
        )


if __name__ == "__main__":
    main()
