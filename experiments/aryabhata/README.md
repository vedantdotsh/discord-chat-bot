# Aryabhata vs the current methods

A small, separate experiment. The Discord bot does not import this package.

Aryabhata (499 CE) published a 24-entry sine table, a value of pi, the kuttaka rule for linear Diophantine equations, a digit-by-digit square root, and the usual summation formulas. This folder runs those next to the methods used now, and trains a one-parameter model with his sine recurrence as the structure against a small neural net and a learned power series.

## Run

```bash
pip install -r experiments/aryabhata/requirements.txt
python3 -m experiments.aryabhata
python3 -m unittest experiments.aryabhata.test_aryabhata
```

Plots and `summary.json` land in `experiments/aryabhata/results/`.

## What is compared

- **Sine, from a rule.** His printed table, a modern 24-entry table with the same arcminute rounding, and the raw second-difference loop with divisor 225. The loop is not the table: it drifts by about a degree, and the 24 integers he published stay inside one arcminute. The same loop with the modern from-scratch constant `h^2` is also scored, along with the Madhava/Taylor series and CORDIC. `math.sin` is the reference, not a contestant.
- **Sine, from samples.** A 1-parameter fit of his curvature, a 2-parameter fit, odd polynomials, linear interpolation, and an 8-unit tanh network (25 parameters). Training angles are an even grid. Test angles are the integer degrees left out.
- **Extrapolation.** The same learners, trained only up to 60°, scored from 61° to 90°.
- **Pi.** `62832/20000` against `22/7`, `355/113`, and `math.pi`, on a circle of diameter 20000 (his own statement of the verse).
- **Kuttaka.** The quotient column against the extended Euclidean algorithm, on the worked example `414x + 189y = 54` and on random pairs.
- **Square roots.** Digit pairs against integer Newton, both checked with `math.isqrt`.
- **Sums.** His closed forms against direct loops, then a polynomial fit that has to rediscover the sum of squares.

## Adding a theory

Put the rule in its own function, score it against the same `math.sin` grid inside `formula_report` or `_score_learners`, and give it a row name. Keep the historical constant and the learned constant separate so a good fit is not mistaken for what he printed.

Planned next, in order: the solar manda epicycle against Kepler's equation of center, then Bhaskara I's rational sine (labeled as Bhaskara, not Aryabhata).

## Sources

The integers in `jya.py` are the first differences decoded from Gitikapada 12. Pi is Ganitapada 10: add 4 to 100, multiply by 8, add 62000, for diameter 20000. The radius 3438 arcminutes is `round(21600 / (2 * 62832/20000))`. Kuttaka is Ganitapada 32-33; the column for 414 and 189 follows the reconstruction in Sathaye's notes on that rule. The exact coefficient `4*sin(h/2)^2` is the modern identity for the second difference he described, and the code labels it as a ceiling because computing it already uses sine.
