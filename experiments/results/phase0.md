# Phase 0: the foundation, and the conventions measured

Phase 0 built what the three implementations share and measured the conventions
that the model rests on, before any of the model is written. The tables behind
the numbers here are in [phase0-verification.md](phase0-verification.md), which
`experiments/phase0_verification.py` writes (measured 2026-09-24).

## What there is

- **The C layer** (`c/jpegio`, C23, over libjpeg-turbo 3.2.0 linked statically).
  It reads a JPEG file's DCT coefficients, quantization tables, sampling factors,
  ICC profile and EXIF orientation, and decodes the component planes without
  upsampling or colour conversion. libjpeg's `longjmp` stays inside it. Its tests
  make files with libjpeg's own encoder from chosen coefficients, in 9 layouts, 8
  coding modes and 7 sizes, and read them back exactly: 33,263 checks. It is also
  fuzzed with libFuzzer on every push and daily.
- **Its faces**: a C++ header (`cpp/include/unround/jpegio.hpp`), a Rust crate
  (`rust/jpegio-sys`), and a Python module over a shared library that exports the
  C API and nothing of libjpeg's (`python/src/unround/jpegio.py`).
- **The build**: CMake presets (debug, release, sanitize, tsan, fuzz) with LLVM
  23 (Clang, LLD, LLVM's archivers) on Windows, Ubuntu and macOS. CI builds and
  tests every preset on all three and checks the Rust, the Python, clang-tidy,
  the layout, and `(void)` parameter lists.
- **Test images** (`scripts/test_images.py`): 50 synthetic originals from fixed
  seeds. There are 30 RGB (text, chart, gradient, line art, UI, illustration; 5
  of each) and 20 grey. 2 of each category are for tuning and 3 for testing. Two
  of the five sizes have sides that are not multiples of 8, to exercise the
  edges. The pictures are drawn with exactly rounded arithmetic only and are
  byte for byte the same on Windows and Linux, as `scripts/test-images.sha256`
  records. They are encoded at quality 10, 20, 30, 50, 70 and 90, and at 4:4:4,
  4:2:2 and 4:2:0 (grey: once), 660 files for each encoder:
  - libjpeg-turbo 3.2.0's `cjpeg`, the submodule's build;
  - mozjpeg 4.1.5's `cjpeg` with its defaults: progressive, trellis
    quantization, overshoot deringing, quantization table 3.

  Both encoders write the same bytes on Windows and on Linux.
- **The tools compared against** (`scripts/comparison-tools.sh`): jpeg2png
  v1.01, jpegqs 1.20260122, knusperli (commit of 2026-03-09) and mozjpeg 4.1.5.
  They are built from their sources at pinned commits with the LLVM toolchain.
  Each passes a test that runs it on a JPEG file on Windows and Linux; macOS is
  built by CI (`.github/workflows/comparison-tools.yml`). Their builds are
  described in `scripts/comparison-tools/CMakeLists.txt`, because the tools' own
  builds (GNU make, Bazel) do not run everywhere.
- **Photographs** (`scripts/fetch_datasets.py`): the script fetches Kodak's
  Photo CD sampler and BSDS500's test set, and checks each file against the
  SHA-256 digests in `scripts/datasets.sha256`. None of the sources publishes
  digests, so a first fetch has to record them. LIVE1 is given out on request
  only and is put in place by hand. Classic5 is not used: no source states its
  terms, and it includes Lena. `list` shows each set's terms.
- **Command-line text on Windows**: every program of the project is linked with
  a manifest that makes UTF-8 its code page (`cmake/utf8.manifest`). Arguments
  then reach `main()` in UTF-8, and `fopen()`, `std::ifstream` and
  `std::filesystem` accept UTF-8 names, as on Linux and macOS.
  `cpp/tests/test_utf8_arguments.cpp` checks this with kanji, Latin-1 letters
  and a character outside the BMP.
  - Without the manifest, the same test on Japanese Windows gets the argument in
    code page 932: "ü" becomes "u" and the emoji "??".
  - Rust and Python get arguments and names from Windows in UTF-16 and need
    nothing.

## The conventions

### 0. The C layer reads what jpeglib reads

It reads the coefficients, quantization tables and sampling factors of every
component of all 1320 files as jpeglib 1.0.2 does, with no difference. The
Python tests compare the two on files of every coding mode as well.

### 1. The DCT convention holds

Each 8×8 block of a decoded component plane is shifted by −128, put through the
orthonormal DCT-II (`scipy.fft.dctn(..., norm="ortho")`, natural order `[v][u]`)
and compared with q·Q. In blocks with no sample at 0 or 255:

| | quality 10–70 | quality 90 |
|---|---|---|
| coefficients within their interval, both encoders | 100 % | 99.87–99.95 % |
| RMS of the difference, in coefficient units | 0.07–0.36 | 0.17–0.20 |

What is left is the error of libjpeg's integer inverse DCT and the rounding of
the planes to 8 bits, whose RMS is sqrt(1/12) ≈ 0.29 per coefficient when the
rounding errors are spread evenly (flat blocks, of which the synthetic images
have many, round with less). At quality 90 many steps are Q = 2 or 3, and that
error now and then exceeds Q/2.

The criterion that this check set out with, "every coefficient within its
interval and an RMS of at most 0.35", holds in those blocks up to quality 70.
It cannot hold in blocks that the decoder clamps to 0 or 255, since their
samples are no longer the inverse DCT of q·Q. The synthetic images have many
such blocks (black text on white). Over all blocks, 98.1–99.7 % of luma coefficients are within their
interval for libjpeg-turbo and 95.2–99.4 % for mozjpeg; chroma is 99.9–100 %
for both. The convention is confirmed, and the criterion is restated as: within
the interval in blocks that are not clamped, up to the 8-bit rounding.

### 2. A standard decoder's output is not in the QCS

djpeg's output (fancy upsampling, integer YCbCr → RGB, 8 bits) is taken back to
YCbCr, averaged to each component's grid, and transformed. For libjpeg-turbo's
files:

- luma: 97.4–99.7 % of the coefficients are within their interval;
- chroma at 4:4:4: 99.6–99.998 %;
- chroma at 4:2:0: 98.4–99.99 %.

Upsampling followed by averaging is not the identity, and clamping and rounding
add to that. Projecting a standard decode onto the QCS therefore moves a few per
cent of its coefficients. The share is largest at high quality, where the
intervals are narrow.

### 3. Where the original lies

**libjpeg-turbo.** Taken through the model in floating point (JFIF YCbCr, the
averaging S with the encoder's edge padding), 98.9–99.99 % of the original's
coefficients lie within their interval. The ones outside exceed it by at most
0.37·Q up to quality 70, and by 1.27·Q at quality 90 (luma). Taken through
libjpeg's own integer arithmetic instead (fixed-point YCbCr rounded to 8 bits,
downsampling with its alternating bias), 99.75–99.999 % are within, and none
exceeds its interval by more than 0.088·Q: that is the error of the integer
forward DCT.

The difference between the two is the encoder's rounding of YCbCr to 8 bits. In
a flat block that rounding moves every sample alike, so the DC coefficient moves
by up to 8 × ½ = 4, which at quality 90 (Q_DC = 3) is more than a whole step.

Two details of libjpeg's encoder matter to the model S. Down a component, it
completes only the last group of rows that downsampling averages, and fills the
rows of blocks below that after downsampling, by repeating the last downsampled
row (jcprepct.c); across, it repeats each row's last sample at full resolution
(jcsample.c). Its downsampling adds an alternating bias before shifting: 1, 2,
1, 2, … for 4:2:0 and 0, 1, 0, 1, … for 4:2:2. With both modelled, the integer
pipeline reproduces the encoder's input to its DCT.

**mozjpeg** does two things on purpose that put the original outside its
intervals:

- **Overshoot deringing.** Before its DCT, mozjpeg pushes the white (255)
  samples of a block that is not all white beyond 255, by up to 31. A decoder
  clamps them back, and the ringing at the edge goes with them. In such blocks
  the original lies up to 15·Q outside (luma), because the file encodes a
  picture brighter than white.
- **Trellis quantization.** In the blocks that deringing leaves alone:

  | | within their interval | outside by, 99th percentile | largest |
  |---|---|---|---|
  | luma | 98.4–99.1 % | about 1.0–1.1·Q | 2.1·Q |
  | chroma | 99.3–99.9 % | 0.25–0.31·Q | 0.67·Q |

  At quality 10 the luma figures are 99.1 %, 1.0·Q and 1.7·Q.

## What it means for the next phases

- **The QCS is right** for files of libjpeg's encoder: the original lies in it,
  apart from the rounding of the encoder's own arithmetic. Up to quality 70, at
  most 0.3 % of the original's coefficients lie outside, by at most 0.37·Q. At
  quality 90 the original can lie more than a step outside at DC, where the
  encoder rounds its YCbCr. A model that rounds as the encoder does is not worth
  building for that: the output is not asked to be the original.
- **`--slack` for trellis files** has to be about a whole step, 1.1·Q on each
  side of the interval, to take in 99 % of the coefficients that the trellis
  moves out of it. That widens each interval from 1·Q to over 3·Q, and it does
  so for the 98–99 % of coefficients that need no slack. Phases 1 and 2 should
  compare a uniform slack with a data term that charges for leaving the interval
  rather than forbidding it.
- **Deringing** means that a file can encode samples beyond 255 on purpose. A
  hard constraint of the picture to [0, 255] (the optional constraint of the
  model) would contradict such files. The range belongs to the output, which is
  clamped, not to the QCS.
- **The comparisons** of Phase 2 run on the synthetic test set and on Kodak and
  BSDS500 once they have been fetched and their digests recorded. The tuning
  images and the test images stay apart.

A check that the tools run is not a comparison, but it shows that the builds
work. On five files of three test images (a chart at quality 10; a gradient and
an illustration at quality 10 and 30; all 4:2:0), jpeg2png gained 0.4–2.0 dB of
PSNR over djpeg. jpegqs gained 0.3–1.2 dB, except on the chart and on the
gradient at quality 10, where it lost up to 0.2 dB (it does not move DC, and
the blocks of a gradient at low quality are steps of DC). knusperli gained 1.0
dB on the gradient at quality 10, and between −0.24 and +0.25 dB on the others.
