"""Checks that the historical algorithms and the learners do what they claim."""

from __future__ import annotations

import math
import random
import unittest

import numpy as np

from experiments.aryabhata.approx import (
    cordic_sin,
    fit_alpha,
    mae,
    predict_mlp,
    predict_recurrence,
    taylor_sin,
    train_mlp,
)
from experiments.aryabhata.jya import (
    JYA_DIFFERENCES,
    RADIUS_ARCMINUTES,
    angle_error_arcmin,
    aryabhata_pi,
    modern_jya,
    modern_rounded_jya,
    oracle_alpha,
    oracle_unit_sines,
    published_jya,
    radius_from_pi,
    rollout,
    rsine_length_arcmin,
)
from experiments.aryabhata.kuttaka import extended_euclid, kuttaka, solve
from experiments.aryabhata.roots import aryabhata_isqrt, newton_isqrt
from experiments.aryabhata.sums import fit_sum_squares, loop_sum, predict_polynomial, sum_cubes, sum_naturals, sum_squares


class JyaTests(unittest.TestCase):
    def test_published_table_checkpoints(self):
        values = published_jya()
        self.assertEqual(len(values), 24)
        self.assertEqual(len(JYA_DIFFERENCES), 24)
        self.assertEqual(values[0], 225)
        self.assertEqual(values[4], 1105)
        self.assertEqual(values[7], 1719)
        self.assertEqual(values[-1], RADIUS_ARCMINUTES)
        self.assertEqual(sum(JYA_DIFFERENCES), RADIUS_ARCMINUTES)

    def test_pi_and_radius(self):
        self.assertEqual(aryabhata_pi(), 3.1416)
        self.assertEqual((100 + 4) * 8 + 62000, 62832)
        self.assertEqual(round(radius_from_pi()), RADIUS_ARCMINUTES)

    def test_printed_table_rsine_length_is_within_one_arcminute(self):
        # Residual in the table's own length unit, not the recovered angle.
        truth = modern_jya()
        worst = max(
            rsine_length_arcmin(pred / RADIUS_ARCMINUTES, got / RADIUS_ARCMINUTES)
            for pred, got in zip(published_jya(), truth)
        )
        self.assertLess(worst, 1.0)

    def test_length_residual_is_not_the_angle_error(self):
        theta = math.radians(60)
        true_unit = math.sin(theta)
        pred = true_unit + 1.0 / RADIUS_ARCMINUTES
        self.assertAlmostEqual(rsine_length_arcmin(pred, true_unit), 1.0, places=6)
        # cos(60 deg) = 1/2, so one arcminute of Rsine is about two of angle.
        self.assertAlmostEqual(angle_error_arcmin(pred, theta), 2.0, delta=0.02)
        small = 1.0 / RADIUS_ARCMINUTES
        self.assertAlmostEqual(angle_error_arcmin(small, 0.0), 1.0, delta=0.01)
        with self.assertRaises(ValueError):
            angle_error_arcmin(1.01, math.pi / 2)

    def test_same_size_modern_table_rounds_within_half_an_arcminute(self):
        truth = modern_jya()
        rounded = modern_rounded_jya()
        self.assertEqual(len(rounded), len(published_jya()))
        worst = max(abs(pred - got) for pred, got in zip(rounded, truth))
        self.assertLessEqual(worst, 0.5)

    def test_thirty_degrees_is_half_the_radius(self):
        # 30 degrees is the 8th node (index 7): 3438 * sin(30 deg) = 1719 exactly.
        self.assertEqual(published_jya()[7], 1719)
        self.assertAlmostEqual(modern_jya()[7], 1719.0, places=6)

    def test_exact_identity_reproduces_sine(self):
        got = oracle_unit_sines(24)
        truth = [math.sin(k * math.pi / 48) for k in range(1, 25)]
        self.assertLess(max(abs(a - b) for a, b in zip(got, truth)), 1e-12)

    def test_one_degree_identity(self):
        step = math.pi / 180
        values = rollout(90, oracle_alpha(step), math.sin(step))
        worst = max(abs(values[k - 1] - math.sin(k * step)) for k in range(1, 91))
        self.assertLess(worst, 1e-12)


class KuttakaTests(unittest.TestCase):
    def test_worked_example(self):
        # 11*189 - 5*414 = 9, the column reconstruction of the pulverizer.
        got = kuttaka(414, 189)
        self.assertEqual(got.gcd, 9)
        self.assertEqual(414 * got.x + 189 * got.y, 9)
        self.assertEqual(got.divisions, extended_euclid(414, 189).divisions)

    def test_scales_to_the_constant(self):
        got = solve(414, 189, 54, method="kuttaka")
        other = solve(414, 189, 54, method="euclid")
        self.assertEqual(414 * got.x + 189 * got.y, 54)
        self.assertEqual(414 * other.x + 189 * other.y, 54)

    def test_matches_extended_euclid_on_random_pairs(self):
        rng = random.Random(2)
        for _ in range(100):
            a = rng.randint(0, 5000)
            b = rng.randint(0, 5000)
            if a == 0 and b == 0:
                continue
            old = kuttaka(a, b)
            modern = extended_euclid(a, b)
            self.assertEqual(old.gcd, modern.gcd)
            self.assertEqual(old.check(a, b), old.gcd)
            self.assertEqual(modern.check(a, b), modern.gcd)
            self.assertEqual(old.divisions, modern.divisions)

    def test_rejects_an_impossible_constant(self):
        with self.assertRaises(ValueError):
            solve(4, 6, 1)


class RootTests(unittest.TestCase):
    def test_matches_library(self):
        samples = [0, 1, 2, 3, 10, 99, 100, 10**6, 10**12 + 12345, int("9" * 60)]
        for n in samples:
            old = aryabhata_isqrt(n)
            newton = newton_isqrt(n)
            self.assertEqual(old.root, math.isqrt(n))
            self.assertEqual(newton.root, math.isqrt(n))
            self.assertGreaterEqual(old.max_operand_bits, 0)
            self.assertGreaterEqual(newton.max_operand_bits, n.bit_length() if n else 0)

    def test_newton_uses_fewer_iterations_on_a_long_integer(self):
        n = int("9" * 200)
        old = aryabhata_isqrt(n)
        newton = newton_isqrt(n)
        self.assertGreater(old.steps, newton.steps)
        # Newton divides the full integer. The digit method's median operand is smaller.
        self.assertGreater(newton.median_operand_bits, old.median_operand_bits)
        self.assertGreater(old.digit_trials, 0)


class SumTests(unittest.TestCase):
    def test_closed_forms_match_loops(self):
        for n in range(1, 200):
            self.assertEqual(sum_naturals(n), loop_sum(n, 1))
            self.assertEqual(sum_squares(n), loop_sum(n, 2))
            self.assertEqual(sum_cubes(n), loop_sum(n, 3))

    def test_degree_three_recovers_sum_of_squares(self):
        coef = fit_sum_squares(8, 3)
        self.assertLess(abs(predict_polynomial(400, coef) - sum_squares(400)), 1e-3)
        quadratic = fit_sum_squares(8, 2)
        self.assertGreater(abs(predict_polynomial(400, quadratic) - sum_squares(400)), 10_000)


class LearnerTests(unittest.TestCase):
    def test_series_and_cordic_near_thirty_degrees(self):
        theta = math.pi / 6
        self.assertAlmostEqual(taylor_sin(theta, 7), 0.5, places=7)
        self.assertAlmostEqual(cordic_sin(theta, 20), 0.5, places=5)
        self.assertLess(abs(cordic_sin(0.0, 16)), 1e-4)
        self.assertLess(abs(cordic_sin(math.pi / 2, 16) - 1.0), 1e-6)

    def test_learned_curvature_finds_the_exact_coefficient(self):
        step = math.pi / 180
        degrees = np.array([0, 20, 40, 60, 80])
        targets = np.sin(np.deg2rad(degrees.astype(float)))
        found = fit_alpha(degrees, targets, math.sin(step))
        self.assertAlmostEqual(found, oracle_alpha(step), places=8)

    def test_mlp_beats_a_constant_on_a_handful_of_samples(self):
        degrees = np.arange(0, 91, 5)
        angles = np.deg2rad(degrees.astype(float))
        targets = np.sin(angles)
        weights = train_mlp(angles, targets, hidden=8, steps=1500, seed=0)
        test = np.deg2rad(np.arange(0, 91).astype(float))
        error = mae(predict_mlp(test, weights), np.sin(test))
        constant = mae(np.full_like(test, 0.5), np.sin(test))
        self.assertLess(error, constant)
        self.assertLess(error, 0.05)
        self.assertEqual(weights.parameter_count, 25)

    def test_recurrence_prediction_matches_rollout(self):
        got = predict_recurrence(np.array([1, 2, 4]), alpha=0.01, delta0=0.2)
        self.assertAlmostEqual(got[0], 0.2, places=12)
        self.assertEqual(len(got), 3)


if __name__ == "__main__":
    unittest.main()
