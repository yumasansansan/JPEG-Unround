# Phase 3: the reference implementation against the Python one

Written by `experiments/phase3_agreement.py`. The reference implementation (Rust), through the package's
binding, against the Python implementation, on every JPEG file of the test images as libjpeg-turbo and
mozjpeg wrote them: the synthetic images, and the photographs. Each entry is the worst over the files of
the group, and each distance a fraction of its bound (1 is the bound), as docs/math.md, 9, derives the
bounds and the conformance cases check them:

- Model to the bit: the files whose intervals, weights of the data term, and centres other than the
  inexact MMSE ones, are the same to the last bit in both.
- Centres: the MMSE centres of AC coefficients of levels other than 0, whose distance is bounded by
  $(4|q| + 40)\, u Q$: each implementation's within $(2|q| + 20)\, u Q$ of the exact mean of its bin.
- Canvas: the Euclidean distance of the canvases after the decoder of the centres, or after
  10 iterations of TV or TGV with the package's other defaults, against the bound of both
  implementations' rounding; the median too.
- Primal values: those of the records (every 5 iterations), against their bounds.
- Brackets: the records, where the gap is certified, whose dual value in each implementation is at
  most the other's primal value, within the rounding of both.
- Gaps: TV's gaps, where certified, against the bound of their difference.

The subgradient method has no bound (docs/math.md, 9.6): its canvas's distance and its primal values'
relative difference are those measured. Its direction is the gradient over its norm, and 0 where the
gradient is 0: on the flat areas of the synthetic pictures, the first iterates of the two
implementations lie within about $10^{-11}$ of each other, but some tens of thousands of pixels are
flat to the last bit in one and not in the other, where the direction is 0 in one and of length 1 in
the other, and the two runs part.

## The synthetic images

| Encoder | Sampling | Method | Files | Model to the bit | Centres | Canvas (worst) | Canvas (median) | Primal values | Brackets | Gaps |
|---|---|---|---|---|---|---|---|---|---|---|
| libjpeg-turbo | grey | mmse | 120 | 120/120 | 0.06 | 2.7e-02 | 2.1e-02 |  |  |  |
| libjpeg-turbo | grey | subgradient | 120 | 120/120 | 0.06 | 1.4e+02 (no bound) |  | 1.7e-02 (relative, no bound) |  |  |
| libjpeg-turbo | grey | tgv | 120 | 120/120 | 0.06 | 2.3e-04 | 7.9e-05 | 1.1e-05 | 120/120 |  |
| libjpeg-turbo | grey | tv | 120 | 120/120 | 0.06 | 1.8e-04 | 7.3e-05 | 9.0e-06 | 120/120 | 1.4e-06 |
| libjpeg-turbo | 420 | mmse | 180 | 180/180 | 0.06 | 3.2e-02 | 2.0e-02 |  |  |  |
| libjpeg-turbo | 420 | tgv | 180 | 180/180 | 0.06 | 2.1e-04 | 8.4e-05 | 1.4e-05 | 0/0 |  |
| libjpeg-turbo | 420 | tv | 180 | 180/180 | 0.06 | 1.9e-04 | 7.7e-05 | 1.2e-05 | 0/0 |  |
| libjpeg-turbo | 422 | mmse | 180 | 180/180 | 0.06 | 3.0e-02 | 1.9e-02 |  |  |  |
| libjpeg-turbo | 422 | tgv | 180 | 180/180 | 0.06 | 2.4e-04 | 8.3e-05 | 1.5e-05 | 0/0 |  |
| libjpeg-turbo | 422 | tv | 180 | 180/180 | 0.06 | 1.9e-04 | 7.5e-05 | 6.6e-06 | 0/0 |  |
| libjpeg-turbo | 444 | mmse | 180 | 180/180 | 0.06 | 2.9e-02 | 1.9e-02 |  |  |  |
| libjpeg-turbo | 444 | tgv | 180 | 180/180 | 0.06 | 2.5e-04 | 8.4e-05 | 2.5e-05 | 180/180 |  |
| libjpeg-turbo | 444 | tv | 180 | 180/180 | 0.06 | 1.9e-04 | 7.5e-05 | 1.4e-05 | 180/180 | 5.3e-07 |
| mozjpeg | grey | mmse | 120 | 120/120 | 0.02 | 2.8e-02 | 2.2e-02 |  |  |  |
| mozjpeg | grey | subgradient | 120 | 120/120 | 0.02 | 1.5e+02 (no bound) |  | 1.8e-02 (relative, no bound) |  |  |
| mozjpeg | grey | tgv | 120 | 120/120 | 0.02 | 2.2e-04 | 9.2e-05 | 1.0e-05 | 120/120 |  |
| mozjpeg | grey | tv | 120 | 120/120 | 0.02 | 1.9e-04 | 7.9e-05 | 1.3e-05 | 120/120 | 2.3e-06 |
| mozjpeg | 420 | mmse | 180 | 180/180 | 0.15 | 3.5e-02 | 2.0e-02 |  |  |  |
| mozjpeg | 420 | tgv | 180 | 180/180 | 0.15 | 2.1e-04 | 8.7e-05 | 7.0e-06 | 0/0 |  |
| mozjpeg | 420 | tv | 180 | 180/180 | 0.15 | 1.9e-04 | 8.0e-05 | 6.6e-06 | 0/0 |  |
| mozjpeg | 422 | mmse | 180 | 180/180 | 0.15 | 3.2e-02 | 1.9e-02 |  |  |  |
| mozjpeg | 422 | tgv | 180 | 180/180 | 0.15 | 2.5e-04 | 9.0e-05 | 1.8e-05 | 0/0 |  |
| mozjpeg | 422 | tv | 180 | 180/180 | 0.15 | 1.9e-04 | 7.7e-05 | 8.2e-06 | 0/0 |  |
| mozjpeg | 444 | mmse | 180 | 180/180 | 0.15 | 3.0e-02 | 1.9e-02 |  |  |  |
| mozjpeg | 444 | tgv | 180 | 180/180 | 0.15 | 2.6e-04 | 9.0e-05 | 7.6e-06 | 180/180 |  |
| mozjpeg | 444 | tv | 180 | 180/180 | 0.15 | 2.0e-04 | 7.7e-05 | 9.8e-06 | 180/180 | 2.6e-07 |

Comparisons beyond their bounds, and failures: 0.

## The photographs

The decoder of the centres alone: the model, the Laplace scales and the centres, of natural pictures.

| Encoder | Sampling | Method | Files | Model to the bit | Centres | Canvas (worst) | Canvas (median) | Primal values | Brackets | Gaps |
|---|---|---|---|---|---|---|---|---|---|---|
| libjpeg-turbo | grey | mmse | 432 | 432/432 | 0.08 | 5.1e-02 | 3.3e-02 |  |  |  |
| libjpeg-turbo | 420 | mmse | 432 | 432/432 | 0.08 | 4.5e-02 | 2.7e-02 |  |  |  |
| libjpeg-turbo | 422 | mmse | 432 | 432/432 | 0.08 | 4.4e-02 | 2.6e-02 |  |  |  |
| libjpeg-turbo | 444 | mmse | 432 | 432/432 | 0.08 | 4.2e-02 | 2.6e-02 |  |  |  |
| mozjpeg | grey | mmse | 432 | 432/432 | 0.00 | 5.0e-02 | 3.2e-02 |  |  |  |
| mozjpeg | 420 | mmse | 432 | 432/432 | 0.03 | 4.6e-02 | 2.6e-02 |  |  |  |
| mozjpeg | 422 | mmse | 432 | 432/432 | 0.08 | 4.4e-02 | 2.5e-02 |  |  |  |
| mozjpeg | 444 | mmse | 432 | 432/432 | 0.00 | 4.3e-02 | 2.5e-02 |  |  |  |

Comparisons beyond their bounds, and failures: 0.
