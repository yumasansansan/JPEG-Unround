# Phase 1: the optimization methods compared, on greyscale files

Phase 1 wrote down the mathematics of the reconstruction, built it in Python for
greyscale files, and compared ways to solve it. The tables behind the numbers
here are in [phase1-tuning.md](phase1-tuning.md) and
[phase1-comparison.md](phase1-comparison.md), which
`experiments/phase1_tuning.py` and `experiments/phase1_comparison.py` write
(measured 2026-09-25).

## What there is

- **The mathematics** (`docs/math.md`): the rules of the arithmetic (rational
  quantities exact and the same in every implementation, floating point in
  binary64 at least, and the output in binary64 to the last bit); the DCT and
  the quantization constraint set, with JPEG's level shift folded into the DC
  intervals; the Laplace model of the AC coefficients, with the
  maximum-likelihood scale in closed form and the MMSE centre of a bin; finite
  differences, their adjoints and norms; the TV and TGV models; the primal–dual
  method of Chambolle and Pock, with the ratio of its steps; when to stop (the
  duality gap of TV in closed form, of TGV through a feasible dual, and a
  partial gap); the defaults; and the subgradient method of jpeg2png's kind.
- **The reference implementation** (`python/src/unround`). A greyscale file is
  decoded by (d) the MMSE decoder, by (b) TV or (c) TGV with the primal–dual
  method, or by (a) the subgradient method. The result is the solver's canvas
  itself, in binary64, and `unround.tiff` writes it as a TIFF of 64-bit
  floating-point samples without changing a bit.
- **Tests** (226). Their tolerances come from error bounds (`tests/rounding.py`),
  and what holds exactly is checked exactly: the adjoints with rationals, the
  norms with the eigenvalues of the operators' matrices, the Laplace scale and
  the MMSE centres against 60-digit decimals, and the TIFF files bit for bit
  through another reader. How far the solvers get in a number of iterations is
  a measurement, kept as a check at about ten times what was measured.
- **The experiments** (`experiments/phase1_*.py`): the tuning of the solvers on
  the tuning images (496 runs), and the comparison on the test images (504
  runs), both greyscale.

## Tuning: the ratio of the steps and when to stop

The tuning images are the 8 greyscale ones, encoded by libjpeg-turbo at 6
qualities: 48 files for TV, and the first image of each kind (24 files) for TGV,
whose iterations cost 2.5 times as much.

**The ratio of the steps.** Theorem 1 of Chambolle and Pock bounds the gap of
the averaged iterates by a sum that is least at τ/σ = ‖z* − z⁰‖² / ‖y* − y⁰‖²,
a ratio of squared distances. Measured from the last points of long runs, that
ratio is 2 to 540 for TV and 150 to 2900 for TGV, falling with the quality. The
ratios that brought the gap down fastest were 10 to 30 for TV and 3 to 10 for
TGV: the bound is far from tight, and the ratio has to be measured. Smaller
ratios bring the primal value down faster still (1 to 3 for TV), but not the
gap, which is what tells the solver to stop.

**The gap does not say how near the least point an iterate is.** The objective
is strongly convex only in the AC coefficients, with the modulus μ/Q², which is
about 10⁻⁷ at low qualities: a gap of 10⁻² per sample bounds the coefficients'
mean squared distance from the least point only to 2·10⁻²/μ = 20 steps². The
least point is unique in its AC coefficients, but the objective is almost flat
around it. On the tuning images the iterates of TV still move by 0.036 grey
levels (RMS, median over the files; up to 0.4) between 2000 and 6000
iterations, long after the objective and the PSNR have settled.

**The tolerance** is therefore chosen by what stopping changes of the result.
Declared before the runs: on every tuning file, stopping where the gap per
sample first falls within the tolerance must change the PSNR of the binary64
result by at most 0.01 dB, and the SSIM of its 8-bit samples by at most 10⁻⁴,
against the last point of a long run (6000 iterations for TV, 8000 for TGV).
The stops were measured exactly, at the first record within each tolerance.

| | τ/σ | gap per sample | iterations on the tuning files | largest change of PSNR / SSIM |
|---|---|---|---|---|
| TV | 30 | 5·10⁻⁴ | 930 to 5310 (median 1815) | 0.0083 dB / 6.4·10⁻⁵ |
| TGV | 3 | 10⁻² | 1600 to 7300 (median 3925) | 0.084 dB / 5.4·10⁻⁵ |

For TV only τ/σ = 30 at 5·10⁻⁴ met the criterion; with 10 and 20 no tolerance
tried did. For TGV no tolerance that every file reached within 8000 iterations
met it, and the long runs themselves still moved by up to 0.025 dB over their
last 1300 iterations: TGV's default is the least tolerance every file reached,
until its convergence is improved. Both are the defaults of `unround.pdhg` for a
weight α (α1 for TGV) of 1; for another weight the ratio is divided by α² and
the tolerance multiplied by α. The most iterations allowed is 20000 for both.

A fixed tolerance fits some files better than others. The gap per sample at
which the PSNR settles to 0.01 dB differs by a factor of about 400 between the
tuning files (120 for the gap relative to the primal value). The hardest are
charts at high qualities, where the error is small and a change of 0.01 dB is a
small change: 5·10⁻⁴ is set by a few files, and most stop later than they need
to.

**The start of TGV's w.** Starting w at 0 rather than at the gradient of the
start's canvas puts it 3 to 10 times nearer the solution's w, and divides the
bound's ratio by 10 to 100. On two files the gap fell no faster with the ratios
300 to 3000, and with the ratio 3 at most 15 % sooner on one file and no sooner
on the other. The start stays at the gradient.

## The methods compared on the test images

The test images are the 12 greyscale ones, encoded by libjpeg-turbo and by
mozjpeg at 6 qualities: 144 files. TGV ran on libjpeg-turbo's 72: on mozjpeg's
it would take thousands of iterations each to show what TV shows of them.

**libjpeg-turbo**, the means over the images (PSNR and SSIM of the 8-bit
samples):

| quality | standard | (d) MMSE | (a) 50 | (a) 500 | (b) TV | (c) TGV |
|---|---|---|---|---|---|---|
| 10 | 25.74 dB / 0.8989 | 25.56 / 0.9092 | 26.04 / 0.9476 | 26.17 / 0.9546 | 26.13 / 0.9547 | 26.16 / 0.9521 |
| 30 | 30.34 / 0.9473 | 30.28 / 0.9573 | 31.63 / 0.9852 | 31.75 / 0.9870 | 31.69 / 0.9871 | 31.76 / 0.9868 |
| 50 | 33.06 / 0.9640 | 32.94 / 0.9716 | 34.71 / 0.9930 | 34.65 / 0.9935 | 34.62 / 0.9936 | 34.67 / 0.9934 |
| 70 | 36.19 / 0.9772 | 35.96 / 0.9820 | 38.09 / 0.9967 | 37.78 / 0.9968 | 37.74 / 0.9967 | 37.79 / 0.9967 |
| 90 | 44.13 / 0.9941 | 43.81 / 0.9950 | 45.94 / 0.9994 | 45.55 / 0.9994 | 45.56 / 0.9995 | 45.56 / 0.9994 |

- **(b) and (c) against the standard decoder**: PSNR +0.4 dB at quality 10 and
  +1.4 to +1.6 dB from 30 up; SSIM from 0.899 to 0.955 at quality 10; PSNR-B,
  which charges blocking, +2.2 dB at quality 10 (23.05 to 25.27).
- **TV and TGV** end within 0.11 dB of each other: TGV is ahead in the PSNR of
  8-bit samples at every quality, by 0.004 to 0.11 dB (most at 20 and 30), and
  slightly behind in SSIM at every quality. The test images are charts, text,
  line art and illustrations, flat between their edges: TV's model, which TGV
  reduces to there.
- **The least point is not the best picture.** At qualities 50 to 90, the
  subgradient method stopped after 50 iterations, far from the least objective,
  has a higher PSNR than the least point: +0.08 dB at 50, +0.35 dB at 70, +0.38
  dB at 90. The model's weights (α = 1, μ = 10⁻³ at every quality) smooth too
  much where the intervals are narrow. Choosing them for the quality is Phase
  2's; this comparison is of the solvers.
- **The MMSE decoder alone** is slightly below the standard decoder here (−0.06
  to −0.32 dB in 8 bits), though higher in SSIM: its shrinkage assumes Laplace-distributed
  coefficients, which images of flat regions and sharp edges do not have, and
  its binary64 output is not clamped to 0–255.

**mozjpeg.** (b) gains less: +0.3 to +1.7 dB of 8-bit PSNR from quality 20 up,
and 0.07 dB less than the standard decoder at quality 10. Its binary64 result is
below the standard decoder's up to quality 70: trellis quantization and
overshoot deringing put the original outside its intervals (Phase 0), and the
solvers keep to them. The 8-bit output, clamped to 0–255,
takes most of the overshoot back. A slack or a data term that charges for
leaving the interval is for Phase 2.

## The acceptance criteria

1. **(b) reaches (a)'s objective in fewer iterations: met, on all 144 files.**
   The primal–dual method reached the objective that the subgradient method had
   after 50 iterations within 20 to 40 iterations, and the one it had after 500
   within 60 to 490 (medians 90 at quality 90 to 325 at quality 10). An
   iteration also costs less: 11.2 ms against 12.2 ms (the median over the test
   images, one process). The figures in phase1-comparison.md show the objective
   against iterations and against time.
2. **Every output within the QCS in binary64, to 10⁻⁴·Q: met.** The canvases
   of (a), (b) and (c) have their coefficients within their intervals up to
   3.8·10⁻¹³ steps: the coefficients the solvers leave at an end of an interval,
   through the round trip of the DCT. The MMSE decoder's are within exactly.

What a file encoded from the result would hold is another matter, as Phase 0
expected. The blocks within the picture are the canvas's and stay within. At
the right and bottom edges of a picture that is not whole blocks, an encoder
pads with the last samples, where the canvas has its own: those blocks are up to
3.4 steps out (libjpeg-turbo's files) and 8.1 (mozjpeg's). Rounding to 8 bits
moves coefficients by up to 2.6 steps where the steps are small. And for
mozjpeg's files, clamping the overshoot moves them by up to 15 steps.

## Time

In Python, one process, per image of 0.08 to 0.23 megapixels: TV took a median
of 14 to 33 seconds by quality (up to 101), TGV 60 to 138 (up to 435). An
iteration of TGV costs about 2.5 times one of TV. NumPy is bound by memory
bandwidth: 12 processes run only about 2.7 times as many iterations as one,
which the experiments' budgets had to allow for. The time that matters is that
of the C++ and Rust implementations (Phases 3 and 4).

## What it means for Phase 2

- **TGV needs a better solver** before it can be the default: diagonal
  preconditioning (Pock and Chambolle 2011) and adaptive steps, and then its
  ratio and tolerance chosen again.
- **Stopping**: a tolerance per sample set by the hardest files makes the
  others iterate longer than they need; a stop that adapts to the file (relative
  gap, or the change of the result) is worth trying.
- **The model's weights** have to depend on the quantization: the least point is
  below an early-stopped iterate at high qualities.
- **mozjpeg's files**: slack, or a data term for leaving the interval.
- **Colour** with chroma subsampling, and **photographs** (Kodak, BSDS500),
  whose smooth gradients are where TGV should beat TV.
- If a picture has to stay consistent after it is encoded again, the canvas's
  padding would have to be the encoder's: a constraint the model does not have.
