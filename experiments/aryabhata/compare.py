"""Run every Aryabhata-versus-modern comparison and write the plots.

Ground truth for sine is math.sin. Nothing here is tuned against the test
split: training angles are an even grid, the test angles are the integer
degrees that grid left out, and the frozen historical rules ignore the
training set entirely.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.aryabhata.approx import (
    cordic_sin,
    fit_alpha,
    fit_alpha_and_delta0,
    fit_odd_polynomial,
    linear_interpolation,
    mae,
    max_abs,
    predict_mlp,
    predict_odd_polynomial,
    predict_recurrence,
    taylor_sin,
    train_mlp,
)
from experiments.aryabhata.jya import (
    RADIUS_ARCMINUTES,
    aryabhata_pi,
    aryabhata_unit_sines,
    modern_jya,
    modern_rounded_jya,
    oracle_alpha,
    oracle_unit_sines,
    published_jya,
    radius_from_pi,
    rollout,
    small_angle_alpha,
    small_angle_unit_sines,
)
from experiments.aryabhata.kuttaka import extended_euclid, kuttaka, solve
from experiments.aryabhata.roots import aryabhata_isqrt, digit_trials, library_isqrt, newton_isqrt
from experiments.aryabhata.sums import (
    fit_sum_squares,
    loop_sum,
    predict_polynomial,
    sum_cubes,
    sum_naturals,
    sum_squares,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TRAIN_SIZES = (2, 4, 6, 8, 12, 18, 30)
MLP_SEEDS = 7
MLP_STEPS = 6000
MLP_LR = 0.01
MLP_HIDDEN = 8

_COLORS = {
    "Aryabhata printed table": "#9a3412",
    "Modern 24-entry table": "#0f766e",
    "Frozen modern 24-entry table": "#0f766e",
    "Aryabhata recurrence": "#c2410c",
    "Small-angle recurrence h^2": "#b45309",
    "Power series, 3 terms": "#0f766e",
    "Power series, 5 terms": "#0369a1",
    "Power series, 7 terms": "#1d4ed8",
    "CORDIC, 8 shifts": "#7c3aed",
    "CORDIC, 16 shifts": "#6d28d9",
    "Linear interpolation": "#64748b",
    "Learned curvature (1 param)": "#c2410c",
    "Learned curvature and start (2 params)": "#9a3412",
    "Odd polynomial, 3 terms": "#0369a1",
    "Odd polynomial, 5 terms": "#1d4ed8",
    "MLP, 8 hidden units": "#7c3aed",
    "Frozen h^2 recurrence": "#a16207",
    "Frozen Aryabhata table": "#9a3412",
}


def even_degrees(count: int, stop: int = 90) -> np.ndarray:
    raw = np.linspace(0, stop, count)
    return np.unique(np.rint(raw).astype(int))


def _finite_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.shape != truth.shape or not np.all(np.isfinite(pred)):
        return 1.0
    return mae(pred, truth)


def formula_report() -> list[dict]:
    """Accuracy of each from-scratch sine method. No training data."""
    node_deg = np.arange(1, 25) * 3.75
    node_rad = np.deg2rad(node_deg)
    truth_node = np.sin(node_rad)
    grid_deg = np.linspace(0.0, 90.0, 361)
    grid_rad = np.deg2rad(grid_deg)
    truth_grid = np.sin(grid_rad)

    def interpolate(node_values: np.ndarray) -> np.ndarray:
        xs = np.concatenate([[0.0], node_deg])
        ys = np.concatenate([[0.0], node_values])
        return np.interp(grid_deg, xs, ys)

    series = [
        (
            "Aryabhata printed table",
            "Gitikapada 12, the 24 integers he published",
            24,
            np.array(published_jya(), dtype=float) / RADIUS_ARCMINUTES,
        ),
        (
            "Modern 24-entry table",
            "Same 24 nodes and arcminute rounding, filled from math.sin",
            24,
            np.array(modern_rounded_jya(), dtype=float) / RADIUS_ARCMINUTES,
        ),
        (
            "Aryabhata recurrence",
            "Same loop, his constants: start at the arc, divide by 225",
            0,
            np.array(aryabhata_unit_sines(), dtype=float),
        ),
        (
            "Small-angle recurrence h^2",
            "Same loop, modern from-scratch constant alpha = h^2",
            0,
            np.array(small_angle_unit_sines(), dtype=float),
        ),
        (
            "Exact difference identity",
            "Same loop with math.sin baked into the constants. A ceiling, not a method.",
            0,
            np.array(oracle_unit_sines(), dtype=float),
        ),
    ]

    rows: list[dict] = []
    for name, source, stored, node_pred in series:
        if name == "Exact difference identity":
            direct = []
            for theta in grid_rad:
                if theta == 0.0:
                    direct.append(0.0)
                else:
                    step = float(theta) / 8.0
                    direct.append(rollout(8, oracle_alpha(step), math.sin(step))[-1])
            grid_pred = np.array(direct)
            grid_note = (
                "Evaluated directly. Constants come from math.sin, so this is a ceiling, not a method."
            )
        else:
            grid_pred = interpolate(node_pred)
            grid_note = "Off-node values are linear interpolation between the 3.75° nodes."
        rows.append(
            {
                "name": name,
                "source": source,
                "stored_numbers": stored,
                "node_mae": mae(node_pred, truth_node),
                "node_max": max_abs(node_pred, truth_node),
                "node_max_arcmin": max_abs(node_pred, truth_node) * RADIUS_ARCMINUTES,
                "grid_mae": mae(grid_pred, truth_grid),
                "grid_max": max_abs(grid_pred, truth_grid),
                "grid_note": grid_note,
                "grid_deg": grid_deg,
                "grid_error": np.abs(grid_pred - truth_grid),
            }
        )

    continuous = []
    for terms in (3, 5, 7):
        continuous.append(
            (
                f"Power series, {terms} terms",
                "Madhava c. 1400, Taylor 1715. Same coefficients.",
                0,
                lambda t, terms=terms: taylor_sin(t, terms),
            )
        )
    for shifts in (8, 16):
        continuous.append(
            (
                f"CORDIC, {shifts} shifts",
                "Current shift-add method. Auxiliary table of arctangents, no sine table.",
                shifts,
                lambda t, shifts=shifts: cordic_sin(t, shifts),
            )
        )
    for name, source, stored, fn in continuous:
        node_pred = np.array([fn(t) for t in node_rad])
        grid_pred = np.array([fn(t) for t in grid_rad])
        rows.append(
            {
                "name": name,
                "source": source,
                "stored_numbers": stored,
                "node_mae": mae(node_pred, truth_node),
                "node_max": max_abs(node_pred, truth_node),
                "node_max_arcmin": max_abs(node_pred, truth_node) * RADIUS_ARCMINUTES,
                "grid_mae": mae(grid_pred, truth_grid),
                "grid_max": max_abs(grid_pred, truth_grid),
                "grid_note": "Evaluated at each angle. No table interpolation.",
                "grid_deg": grid_deg,
                "grid_error": np.abs(grid_pred - truth_grid),
            }
        )
    return rows


def _score_learners(
    train_deg: np.ndarray,
    test_deg: np.ndarray,
    mlp_seeds: int,
    mlp_steps: int,
) -> dict[str, float | dict]:
    """Fit every learner on train_deg and score sine on test_deg."""
    step = math.pi / 180.0
    train_rad = np.deg2rad(train_deg.astype(float))
    test_rad = np.deg2rad(test_deg.astype(float))
    train_y = np.sin(train_rad)
    truth = np.sin(test_rad)

    alpha = fit_alpha(train_deg, train_y, step)
    pred_alpha = predict_recurrence(test_deg, alpha, step)
    alpha2, delta2 = fit_alpha_and_delta0(train_deg, train_y, alpha, step)
    pred_two = predict_recurrence(test_deg, alpha2, delta2)

    poly3, scale3 = fit_odd_polynomial(train_rad, train_y, 3)
    poly5, scale5 = fit_odd_polynomial(train_rad, train_y, 5)
    interp = linear_interpolation(train_rad, train_y, test_rad)

    mlp_maes = []
    for seed in range(mlp_seeds):
        weights = train_mlp(
            train_rad, train_y, hidden=MLP_HIDDEN, steps=mlp_steps, lr=MLP_LR, seed=seed
        )
        mlp_maes.append(_finite_mae(predict_mlp(test_rad, weights), truth))
    mlp_maes_arr = np.array(mlp_maes)

    frozen_h2 = predict_recurrence(test_deg, small_angle_alpha(step), step)
    # Published integers, not the raw recurrence. The raw loop drifts by about a degree.
    native_deg = np.concatenate([[0.0], np.arange(1, 25) * 3.75])
    native_y = np.concatenate([[0.0], np.array(published_jya(), dtype=float) / RADIUS_ARCMINUTES])
    frozen_table = np.interp(test_deg.astype(float), native_deg, native_y)
    modern_y = np.concatenate([[0.0], np.array(modern_rounded_jya(), dtype=float) / RADIUS_ARCMINUTES])
    frozen_modern = np.interp(test_deg.astype(float), native_deg, modern_y)

    return {
        "train_size": int(len(train_deg)),
        "test_size": int(len(test_deg)),
        "learned_alpha": alpha,
        "learned_alpha_2": alpha2,
        "learned_delta0": delta2,
        "oracle_alpha": oracle_alpha(step),
        "oracle_delta0": math.sin(step),
        "Linear interpolation": _finite_mae(interp, truth),
        "Learned curvature (1 param)": _finite_mae(pred_alpha, truth),
        "Learned curvature and start (2 params)": _finite_mae(pred_two, truth),
        "Odd polynomial, 3 terms": _finite_mae(predict_odd_polynomial(test_rad, poly3, scale3), truth),
        "Odd polynomial, 5 terms": _finite_mae(predict_odd_polynomial(test_rad, poly5, scale5), truth),
        "MLP, 8 hidden units": float(np.median(mlp_maes_arr)),
        "mlp_q25": float(np.quantile(mlp_maes_arr, 0.25)),
        "mlp_q75": float(np.quantile(mlp_maes_arr, 0.75)),
        "Frozen h^2 recurrence": _finite_mae(frozen_h2, truth),
        "Frozen Aryabhata table": _finite_mae(frozen_table, truth),
        "Frozen modern 24-entry table": _finite_mae(frozen_modern, truth),
    }


def learning_report(mlp_seeds: int = MLP_SEEDS, mlp_steps: int = MLP_STEPS) -> list[dict]:
    rows = []
    universe = np.arange(0, 91)
    for count in TRAIN_SIZES:
        train = even_degrees(count, stop=90)
        test = np.setdiff1d(universe, train)
        rows.append(_score_learners(train, test, mlp_seeds, mlp_steps))
    return rows


def extrapolation_report(mlp_seeds: int = MLP_SEEDS, mlp_steps: int = MLP_STEPS) -> dict:
    """Train on an even grid of 0..60 degrees. Test 61..90, outside that grid."""
    train = even_degrees(13, stop=60)
    test = np.arange(61, 91)
    scored = _score_learners(train, test, mlp_seeds, mlp_steps)
    scored["train_degrees"] = [int(d) for d in train]
    scored["protocol"] = "Train on 13 even degrees from 0 to 60. Test every degree from 61 to 90."
    return scored


def pi_report() -> list[dict]:
    true = math.pi
    diameter = 20000
    true_circumference = diameter * true
    candidates = [
        ("Aryabhata 62832/20000", aryabhata_pi(), "Gaṇitapada 10, 499 CE"),
        ("Archimedes 22/7", 22 / 7, "c. 250 BCE"),
        ("Zu Chongzhi 355/113", 355 / 113, "c. 480 CE"),
    ]
    rows = []
    for name, value, source in candidates:
        rows.append(
            {
                "name": name,
                "source": source,
                "value": value,
                "abs_error": abs(value - true),
                "circumference_error_on_diameter_20000": abs(diameter * value - true_circumference),
                "implied_radius_arcmin": radius_from_pi(value),
            }
        )
    rows.append(
        {
            "name": "Aryabhata radius 3438",
            "source": "round(21600 / (2 * 62832/20000))",
            "value": RADIUS_ARCMINUTES,
            "abs_error": abs(RADIUS_ARCMINUTES - radius_from_pi(true)),
            "circumference_error_on_diameter_20000": None,
            "implied_radius_arcmin": radius_from_pi(aryabhata_pi()),
        }
    )
    return rows


def kuttaka_report(trials: int = 400) -> dict:
    rng = random.Random(0)
    mismatches = 0
    division_gaps = []
    for _ in range(trials):
        a = rng.randint(1, 10**12)
        b = rng.randint(1, 10**12)
        old = kuttaka(a, b)
        modern = extended_euclid(a, b)
        if old.gcd != modern.gcd or old.check(a, b) != old.gcd or modern.check(a, b) != modern.gcd:
            mismatches += 1
        division_gaps.append(abs(old.divisions - modern.divisions))

    example_a, example_b, example_c = 414, 189, 54
    by_kuttaka = solve(example_a, example_b, example_c, method="kuttaka")
    by_euclid = solve(example_a, example_b, example_c, method="euclid")

    # Timing on wider integers, where the loop cost is visible.
    rng_time = random.Random(1)
    samples = [(rng_time.randrange(10**60), rng_time.randrange(10**60)) for _ in range(30)]
    for fn in (kuttaka, extended_euclid):
        for a, b in samples[:3]:
            fn(a, b)
    times = {}
    for label, fn in (("kuttaka", kuttaka), ("extended_euclid", extended_euclid)):
        start = time.perf_counter()
        for a, b in samples:
            fn(a, b)
        times[label] = (time.perf_counter() - start) / len(samples)

    return {
        "trials": trials,
        "mismatches": mismatches,
        "mean_abs_division_gap": float(np.mean(division_gaps)),
        "example": {
            "equation": "414*x + 189*y = 54",
            "kuttaka": {"x": by_kuttaka.x, "y": by_kuttaka.y, "divisions": by_kuttaka.divisions},
            "extended_euclid": {"x": by_euclid.x, "y": by_euclid.y, "divisions": by_euclid.divisions},
            "kuttaka_checks": by_kuttaka.check(example_a, example_b) == example_c,
            "euclid_checks": by_euclid.check(example_a, example_b) == example_c,
        },
        "mean_seconds_60_digit": times,
        "note": (
            "Same divisions, same gcd, both identities check out. "
            "Kuttaka is the extended Euclidean algorithm written as a quotient column."
        ),
    }


def sqrt_report() -> list[dict]:
    rows = []
    for digits in (8, 40, 200, 800):
        n = int("9" * digits)
        start = time.perf_counter()
        root_a, steps_a = aryabhata_isqrt(n)
        seconds_a = time.perf_counter() - start
        start = time.perf_counter()
        root_n, steps_n = newton_isqrt(n)
        seconds_n = time.perf_counter() - start
        root_lib = library_isqrt(n)
        rows.append(
            {
                "digits": digits,
                "aryabhata_root_matches": root_a == root_lib,
                "newton_root_matches": root_n == root_lib,
                "aryabhata_digit_pair_steps": steps_a,
                "aryabhata_digit_trials": digit_trials(n),
                "newton_iterations": steps_n,
                "aryabhata_seconds": seconds_a,
                "newton_seconds": seconds_n,
            }
        )
    return rows


def sums_report() -> dict:
    mismatches = {"naturals": 0, "squares": 0, "cubes": 0}
    for n in range(1, 501):
        if sum_naturals(n) != loop_sum(n, 1):
            mismatches["naturals"] += 1
        if sum_squares(n) != loop_sum(n, 2):
            mismatches["squares"] += 1
        if sum_cubes(n) != loop_sum(n, 3):
            mismatches["cubes"] += 1
    target_n = 400
    truth = sum_squares(target_n)
    fitted = {}
    for degree in (2, 3):
        coef = fit_sum_squares(8, degree)
        pred = predict_polynomial(target_n, coef)
        fitted[f"degree_{degree}_trained_on_1_to_8"] = {
            "prediction_at_400": pred,
            "abs_error": abs(pred - truth),
            "coefficients": [float(c) for c in coef],
        }
    return {
        "checked_through_n": 500,
        "closed_form_mismatches": mismatches,
        "closed_form_at_400": truth,
        "learned_polynomials": fitted,
        "note": (
            "The closed forms match the loops exactly, and they match the formulas "
            "still taught. A degree-2 fit cannot be the sum of squares. A degree-3 "
            "fit, given 8 examples, recovers it."
        ),
    }


def _plot_errors(formulas: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    skip = {"Exact difference identity"}
    for row in formulas:
        if row["name"] in skip:
            continue
        error = np.clip(row["grid_error"], 1e-16, None)
        ax.plot(
            row["grid_deg"],
            error,
            label=row["name"],
            color=_COLORS.get(row["name"], "#334155"),
            linewidth=1.6,
        )
    ax.set_yscale("log")
    ax.set_xlim(0, 90)
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Absolute error against math.sin")
    ax.set_title("Sine error from 0° to 90°")
    ax.grid(True, which="both", axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_learning(rows: list[dict], path: Path) -> None:
    methods = [
        "Frozen Aryabhata table",
        "Frozen modern 24-entry table",
        "Frozen h^2 recurrence",
        "Learned curvature (1 param)",
        "Learned curvature and start (2 params)",
        "Odd polynomial, 5 terms",
        "Linear interpolation",
        "MLP, 8 hidden units",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sizes = [row["train_size"] for row in rows]
    for name in methods:
        series = [row[name] for row in rows]
        ax.plot(
            sizes,
            np.clip(series, 1e-16, None),
            marker="o",
            label=name,
            color=_COLORS.get(name, "#334155"),
            linewidth=1.6,
        )
    mlp_low = [row["mlp_q25"] for row in rows]
    mlp_high = [row["mlp_q75"] for row in rows]
    ax.fill_between(sizes, mlp_low, mlp_high, color="#7c3aed", alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("Training samples (evenly spaced degrees, 0° to 90°)")
    ax.set_ylabel("Test mean absolute error")
    ax.set_title("Sample efficiency on held-out degrees")
    ax.grid(True, which="both", axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_extrapolation(row: dict, path: Path) -> None:
    names = [
        "Frozen Aryabhata table",
        "Frozen modern 24-entry table",
        "Frozen h^2 recurrence",
        "Learned curvature (1 param)",
        "Learned curvature and start (2 params)",
        "Odd polynomial, 3 terms",
        "Odd polynomial, 5 terms",
        "Linear interpolation",
        "MLP, 8 hidden units",
    ]
    values = [max(row[name], 1e-16) for name in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(names))
    ax.barh(y, values, color=[_COLORS.get(name, "#334155") for name in names])
    ax.set_yticks(y, labels=names)
    ax.set_xscale("log")
    ax.set_xlabel("Mean absolute error on 61° to 90°")
    ax.set_title("Error on 61°–90° after seeing only 0°–60°")
    ax.grid(True, which="both", axis="x", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _strip_arrays(rows: list[dict]) -> list[dict]:
    kept = []
    for row in rows:
        kept.append({key: value for key, value in row.items() if key not in {"grid_deg", "grid_error"}})
    return kept


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def build_summary(mlp_seeds: int = MLP_SEEDS, mlp_steps: int = MLP_STEPS, kuttaka_trials: int = 400) -> dict:
    started = time.perf_counter()
    formulas = formula_report()
    learning = learning_report(mlp_seeds=mlp_seeds, mlp_steps=mlp_steps)
    extra = extrapolation_report(mlp_seeds=mlp_seeds, mlp_steps=mlp_steps)
    pi = pi_report()
    kuttaka = kuttaka_report(kuttaka_trials)
    roots = sqrt_report()
    sums = sums_report()
    summary = {
        "ground_truth": "math.sin, math.pi, math.isqrt",
        "native_step_degrees": 3.75,
        "radius_arcminutes": RADIUS_ARCMINUTES,
        "radius_from_his_pi": radius_from_pi(),
        "modern_jya_at_30deg": modern_jya()[7],
        "oracle_alpha_native_step": oracle_alpha(math.pi / 48),
        "small_angle_alpha_native_step": small_angle_alpha(math.pi / 48),
        "aryabhata_alpha_native_step": 1 / 225,
        "runtime_seconds": time.perf_counter() - started,
        "mlp": {"hidden": MLP_HIDDEN, "steps": mlp_steps, "seeds": mlp_seeds, "parameters": 1 * MLP_HIDDEN + MLP_HIDDEN + MLP_HIDDEN + 1},
        "sine_formulas": _strip_arrays(formulas),
        "sample_efficiency": learning,
        "extrapolation": {key: value for key, value in extra.items()},
        "pi": pi,
        "kuttaka": kuttaka,
        "square_roots": roots,
        "sums": sums,
    }
    summary["_plot_cache"] = {"formulas": formulas, "learning": learning, "extrapolation": extra}
    return summary


def write_results(summary: dict, directory: Path = RESULTS) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    cache = summary.pop("_plot_cache")
    paths = {
        "sine_error": directory / "sine_error.png",
        "sample_efficiency": directory / "sample_efficiency.png",
        "extrapolation": directory / "extrapolation.png",
        "summary": directory / "summary.json",
    }
    _plot_errors(cache["formulas"], paths["sine_error"])
    _plot_learning(cache["learning"], paths["sample_efficiency"])
    _plot_extrapolation(cache["extrapolation"], paths["extrapolation"])
    paths["summary"].write_text(json.dumps(_clean(summary), indent=2) + "\n", encoding="utf-8")
    summary["_plot_cache"] = cache
    return paths


def main() -> None:
    summary = build_summary()
    paths = write_results(summary)
    print(f"wrote {paths['summary']}")
    for row in summary["sine_formulas"]:
        print(
            f"{row['name']:<32} node max {row['node_max_arcmin']:.4f} arcmin   "
            f"grid mae {row['grid_mae']:.3e}"
        )
    print("sample efficiency (test MAE):")
    for row in summary["sample_efficiency"]:
        print(
            f"  n={row['train_size']:<3}  "
            f"curvature {row['Learned curvature (1 param)']:.3e}  "
            f"mlp {row['MLP, 8 hidden units']:.3e}  "
            f"poly5 {row['Odd polynomial, 5 terms']:.3e}  "
            f"interp {row['Linear interpolation']:.3e}"
        )


if __name__ == "__main__":
    main()
