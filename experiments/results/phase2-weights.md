<!-- Written by experiments/phase2_weights.py; do not edit by hand. -->

# Phase 2: the weights of the model

- Files: the tuning images, as libjpeg-turbo encoded them unless a section says otherwise: the synthetic greyscale images (48 files), the photographs in grey (24 of BSDS500's train split, 144 files), the synthetic colour images in 4:2:0 and 4:4:4 (144 files), and the photographs in 4:2:0 and 4:4:4 (288 files).
- TV with $\alpha = 1$ unless a section says otherwise, the weights $\mu / Q^2$ and no weight on DC, stopped at the package's defaults; the data term's centres the MMSE ones or the middles of the intervals (docs/math.md, 2.2 and 4.1).
- The measures are those of the 8-bit result, rounded and clamped as the standard decoder's (libjpeg-turbo, through Pillow), against the original; of colour files, of RGB, the SSIM the mean over R, G and B.
- The reference implementation, unround 0.1.0 (Rust, C interface); NumPy 2.5.3, scikit-image 0.26.0.

## Synthetic images: the 8-bit PSNR against the standard decoder

Median over the files, by quality, of the 8-bit PSNR less the standard decoder's (dB).

| quality | midpoint 10 | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 | midpoint 3000 | midpoint 10000 | mmse 0.001 | mmse 10 | mmse 30 | mmse 100 | mmse 300 | mmse 1000 | mmse 3000 | mmse 10000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | +0.48 | +0.56 | +0.74 | +1.02 | +1.27 | +1.00 | +0.44 | +0.41 | +0.45 | +0.49 | +0.58 | +0.63 | +0.62 | +0.44 | +0.08 |
| 20 | +0.71 | +0.86 | +1.22 | +1.59 | +1.71 | +1.23 | +0.50 | +0.58 | +0.67 | +0.78 | +0.94 | +0.99 | +0.81 | +0.38 | -0.01 |
| 30 | +1.44 | +1.65 | +2.11 | +2.55 | +2.38 | +1.43 | +0.46 | +1.26 | +1.38 | +1.51 | +1.68 | +1.62 | +1.12 | +0.38 | -0.03 |
| 50 | +1.24 | +1.64 | +2.35 | +2.82 | +2.39 | +1.15 | +0.33 | +0.96 | +1.13 | +1.34 | +1.56 | +1.46 | +0.86 | +0.25 | -0.00 |
| 70 | +1.42 | +1.96 | +2.87 | +3.26 | +1.95 | +0.77 | +0.19 | +0.97 | +1.31 | +1.61 | +1.71 | +1.43 | +0.80 | +0.18 | -0.10 |
| 90 | +2.19 | +3.32 | +3.85 | +2.41 | +0.87 | +0.29 | +0.07 | +0.84 | +1.79 | +2.23 | +1.99 | +1.27 | +0.49 | +0.12 | -0.05 |

## Synthetic images: the SSIM against the standard decoder

Likewise, of the SSIM of the 8-bit result.

| quality | midpoint 10 | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 | midpoint 3000 | midpoint 10000 | mmse 0.001 | mmse 10 | mmse 30 | mmse 100 | mmse 300 | mmse 1000 | mmse 3000 | mmse 10000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | +0.0422 | +0.0432 | +0.0455 | +0.0487 | +0.0456 | +0.0319 | +0.0169 | +0.0417 | +0.0421 | +0.0427 | +0.0440 | +0.0449 | +0.0410 | +0.0338 | +0.0242 |
| 20 | +0.0353 | +0.0359 | +0.0372 | +0.0376 | +0.0319 | +0.0207 | +0.0087 | +0.0348 | +0.0352 | +0.0357 | +0.0364 | +0.0359 | +0.0309 | +0.0238 | +0.0170 |
| 30 | +0.0301 | +0.0306 | +0.0314 | +0.0313 | +0.0256 | +0.0152 | +0.0061 | +0.0297 | +0.0300 | +0.0304 | +0.0308 | +0.0299 | +0.0253 | +0.0193 | +0.0138 |
| 50 | +0.0227 | +0.0231 | +0.0236 | +0.0228 | +0.0169 | +0.0089 | +0.0034 | +0.0223 | +0.0226 | +0.0229 | +0.0231 | +0.0219 | +0.0181 | +0.0135 | +0.0089 |
| 70 | +0.0148 | +0.0151 | +0.0155 | +0.0144 | +0.0094 | +0.0042 | +0.0012 | +0.0145 | +0.0147 | +0.0150 | +0.0151 | +0.0139 | +0.0106 | +0.0067 | +0.0045 |
| 90 | +0.0035 | +0.0036 | +0.0035 | +0.0025 | +0.0010 | +0.0004 | +0.0001 | +0.0033 | +0.0034 | +0.0035 | +0.0034 | +0.0026 | +0.0014 | +0.0008 | +0.0006 |

## Synthetic images: the best mu, with the mmse centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 300: 4, 1000: 1, 3000: 3 | +0.27 |
| 20 | 144.3 | 100: 2, 300: 4, 1000: 2 | +0.45 |
| 30 | 95.7 | 100: 5, 300: 1, 1000: 2 | +0.59 |
| 50 | 57.6 | 30: 1, 100: 4, 300: 3 | +0.70 |
| 70 | 34.5 | 30: 1, 100: 5, 300: 2 | +0.92 |
| 90 | 11.5 | 30: 6, 100: 2 | +1.43 |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.88, and $\mu = 4.31$ at a mean step of 1.

## Synthetic images: the best mu, with the midpoint centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 1000: 5, 3000: 3 | +1.05 |
| 20 | 144.3 | 300: 3, 1000: 4, 3000: 1 | +1.35 |
| 30 | 95.7 | 300: 4, 1000: 4 | +1.63 |
| 50 | 57.6 | 100: 1, 300: 3, 1000: 4 | +2.22 |
| 70 | 34.5 | 100: 2, 300: 6 | +2.88 |
| 90 | 11.5 | 30: 2, 100: 6 | +3.47 |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.91, and $\mu = 8.98$ at a mean step of 1.

## Photographs: the 8-bit PSNR against the standard decoder

Median over the files, by quality, of the 8-bit PSNR less the standard decoder's (dB).

| quality | midpoint 10 | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 | midpoint 3000 | mmse 0.001 | mmse 10 | mmse 30 | mmse 100 | mmse 300 | mmse 1000 | mmse 3000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | -0.41 | -0.36 | -0.22 | +0.00 | +0.20 | +0.09 | -0.46 | -0.42 | -0.36 | -0.25 | -0.10 | +0.04 | +0.10 |
| 20 | -0.56 | -0.47 | -0.20 | +0.14 | +0.26 | +0.14 | -0.62 | -0.56 | -0.48 | -0.27 | -0.02 | +0.15 | +0.21 |
| 30 | -0.58 | -0.45 | -0.18 | +0.25 | +0.38 | +0.19 | -0.65 | -0.58 | -0.46 | -0.24 | +0.07 | +0.25 | +0.26 |
| 50 | -0.63 | -0.45 | -0.02 | +0.46 | +0.51 | +0.26 | -0.76 | -0.64 | -0.47 | -0.13 | +0.23 | +0.39 | +0.38 |
| 70 | -0.70 | -0.40 | +0.13 | +0.67 | +0.59 | +0.23 | -0.92 | -0.70 | -0.44 | -0.01 | +0.39 | +0.55 | +0.51 |
| 90 | -0.22 | +0.62 | +1.51 | +1.22 | +0.47 | +0.15 | -0.90 | -0.22 | +0.51 | +1.40 | +1.65 | +1.49 | +1.35 |

## Photographs: the SSIM against the standard decoder

Likewise, of the SSIM of the 8-bit result.

| quality | midpoint 10 | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 | midpoint 3000 | mmse 0.001 | mmse 10 | mmse 30 | mmse 100 | mmse 300 | mmse 1000 | mmse 3000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | -0.0193 | -0.0179 | -0.0135 | -0.0026 | +0.0101 | +0.0065 | -0.0201 | -0.0193 | -0.0180 | -0.0139 | -0.0061 | +0.0049 | +0.0042 |
| 20 | -0.0252 | -0.0209 | -0.0112 | +0.0010 | +0.0051 | +0.0038 | -0.0279 | -0.0252 | -0.0212 | -0.0123 | -0.0025 | +0.0023 | +0.0027 |
| 30 | -0.0212 | -0.0168 | -0.0074 | +0.0018 | +0.0047 | +0.0034 | -0.0247 | -0.0213 | -0.0170 | -0.0087 | -0.0017 | +0.0022 | +0.0029 |
| 50 | -0.0169 | -0.0123 | -0.0038 | +0.0027 | +0.0051 | +0.0026 | -0.0205 | -0.0169 | -0.0126 | -0.0058 | +0.0006 | +0.0036 | +0.0046 |
| 70 | -0.0116 | -0.0072 | -0.0004 | +0.0044 | +0.0041 | +0.0019 | -0.0148 | -0.0117 | -0.0074 | -0.0015 | +0.0026 | +0.0048 | +0.0045 |
| 90 | -0.0001 | +0.0029 | +0.0061 | +0.0052 | +0.0022 | +0.0008 | -0.0036 | -0.0002 | +0.0026 | +0.0059 | +0.0067 | +0.0057 | +0.0050 |

## Photographs: the best mu, with the mmse centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 1000: 9, 3000: 15 | +0.57 |
| 20 | 144.3 | 300: 3, 1000: 7, 3000: 14 | +0.83 |
| 30 | 95.7 | 100: 1, 300: 4, 1000: 6, 3000: 13 | +0.96 |
| 50 | 57.6 | 100: 1, 300: 3, 1000: 9, 3000: 11 | +1.16 |
| 70 | 34.5 | 100: 1, 300: 3, 1000: 9, 3000: 11 | +1.50 |
| 90 | 11.5 | 100: 1, 300: 19, 1000: 4 | +2.53 |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.49, and $\mu = 151$ at a mean step of 1.

## Photographs: the best mu, with the midpoint centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 300: 1, 1000: 23 | +0.70 |
| 20 | 144.3 | 300: 8, 1000: 16 | +0.93 |
| 30 | 95.7 | 300: 10, 1000: 14 | +1.09 |
| 50 | 57.6 | 300: 11, 1000: 13 | +1.31 |
| 70 | 34.5 | 300: 20, 1000: 4 | +1.63 |
| 90 | 11.5 | 100: 19, 300: 5 | +2.38 |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.60, and $\mu = 37.3$ at a mean step of 1.

## Synthetic images, 4:2:0: the 8-bit PSNR against the standard decoder

Median over the files, by quality, of the 8-bit PSNR less the standard decoder's (dB).

| quality | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 | midpoint 3000 | mmse 30 | mmse 100 | mmse 300 | mmse 1000 | mmse 3000 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | +0.27 | +0.45 | +0.63 | +0.65 | +0.55 | +0.23 | +0.25 | +0.24 | +0.17 | -0.12 |
| 20 | +0.43 | +0.82 | +1.29 | +1.31 | +0.85 | +0.34 | +0.44 | +0.57 | +0.38 | +0.17 |
| 30 | +1.68 | +1.86 | +1.92 | +1.57 | +0.97 | +1.59 | +1.58 | +1.42 | +1.00 | +0.47 |
| 50 | +1.83 | +2.06 | +2.35 | +2.09 | +0.97 | +1.70 | +1.72 | +1.50 | +0.96 | +0.36 |
| 70 | +2.37 | +2.59 | +3.03 | +1.98 | +0.94 | +2.30 | +2.17 | +1.80 | +1.09 | +0.63 |
| 90 | +3.69 | +4.07 | +3.37 | +2.09 | +1.71 | +3.38 | +3.25 | +2.79 | +2.18 | +1.78 |

## Synthetic images, 4:2:0: the SSIM against the standard decoder

Likewise, of the SSIM of the 8-bit result.

| quality | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 | midpoint 3000 | mmse 30 | mmse 100 | mmse 300 | mmse 1000 | mmse 3000 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | +0.0551 | +0.0558 | +0.0568 | +0.0545 | +0.0424 | +0.0544 | +0.0540 | +0.0535 | +0.0512 | +0.0423 |
| 20 | +0.0463 | +0.0474 | +0.0484 | +0.0428 | +0.0281 | +0.0459 | +0.0463 | +0.0458 | +0.0402 | +0.0313 |
| 30 | +0.0396 | +0.0403 | +0.0404 | +0.0354 | +0.0218 | +0.0393 | +0.0394 | +0.0385 | +0.0340 | +0.0265 |
| 50 | +0.0285 | +0.0289 | +0.0283 | +0.0233 | +0.0137 | +0.0283 | +0.0282 | +0.0273 | +0.0233 | +0.0180 |
| 70 | +0.0192 | +0.0193 | +0.0183 | +0.0143 | +0.0065 | +0.0190 | +0.0188 | +0.0178 | +0.0146 | +0.0107 |
| 90 | +0.0086 | +0.0084 | +0.0074 | +0.0037 | +0.0006 | +0.0085 | +0.0082 | +0.0074 | +0.0053 | +0.0036 |

## Synthetic images, 4:2:0: the best mu, with the mmse centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 30: 2, 100: 2, 300: 5, 1000: 1, 3000: 2 |  |
| 20 | 144.3 | 100: 4, 300: 6, 1000: 2 |  |
| 30 | 95.7 | 30: 1, 100: 8, 300: 1, 1000: 2 |  |
| 50 | 57.6 | 30: 2, 100: 7, 300: 3 |  |
| 70 | 34.5 | 30: 5, 100: 4, 300: 2, 3000: 1 |  |
| 90 | 11.5 | 30: 9, 100: 3 |  |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.61, and $\mu = 9.66$ at a mean step of 1.

## Synthetic images, 4:2:0: the best mu, with the midpoint centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 300: 4, 1000: 5, 3000: 3 |  |
| 20 | 144.3 | 30: 1, 300: 5, 1000: 6 |  |
| 30 | 95.7 | 30: 1, 100: 2, 300: 4, 1000: 5 |  |
| 50 | 57.6 | 30: 1, 100: 3, 300: 7, 1000: 1 |  |
| 70 | 34.5 | 100: 4, 300: 7, 3000: 1 |  |
| 90 | 11.5 | 30: 4, 100: 7, 1000: 1 |  |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.69, and $\mu = 16$ at a mean step of 1.

## Synthetic images, 4:4:4: the 8-bit PSNR against the standard decoder

Median over the files, by quality, of the 8-bit PSNR less the standard decoder's (dB).

| quality | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 | midpoint 3000 | mmse 30 | mmse 100 | mmse 300 | mmse 1000 | mmse 3000 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | +0.12 | +0.26 | +0.47 | +0.83 | +0.60 | +0.02 | +0.09 | +0.04 | +0.02 | -0.12 |
| 20 | +0.06 | +0.35 | +0.81 | +1.04 | +0.78 | -0.00 | +0.06 | +0.10 | +0.20 | +0.12 |
| 30 | +1.42 | +1.87 | +2.26 | +2.07 | +1.12 | +1.12 | +1.22 | +1.08 | +0.55 | +0.22 |
| 50 | +2.27 | +2.53 | +2.96 | +2.18 | +0.98 | +1.88 | +1.91 | +1.60 | +0.80 | +0.31 |
| 70 | +2.36 | +3.11 | +3.30 | +1.88 | +0.70 | +1.83 | +1.73 | +1.31 | +0.69 | +0.31 |
| 90 | +2.96 | +3.55 | +2.17 | +0.77 | +0.33 | +2.28 | +1.92 | +1.17 | +0.53 | +0.30 |

## Synthetic images, 4:4:4: the SSIM against the standard decoder

Likewise, of the SSIM of the 8-bit result.

| quality | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 | midpoint 3000 | mmse 30 | mmse 100 | mmse 300 | mmse 1000 | mmse 3000 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | +0.0494 | +0.0513 | +0.0525 | +0.0486 | +0.0405 | +0.0486 | +0.0494 | +0.0490 | +0.0457 | +0.0416 |
| 20 | +0.0429 | +0.0440 | +0.0436 | +0.0354 | +0.0230 | +0.0422 | +0.0430 | +0.0419 | +0.0358 | +0.0272 |
| 30 | +0.0367 | +0.0378 | +0.0375 | +0.0296 | +0.0180 | +0.0364 | +0.0370 | +0.0364 | +0.0308 | +0.0238 |
| 50 | +0.0264 | +0.0271 | +0.0266 | +0.0209 | +0.0107 | +0.0261 | +0.0264 | +0.0255 | +0.0215 | +0.0154 |
| 70 | +0.0176 | +0.0178 | +0.0167 | +0.0109 | +0.0048 | +0.0174 | +0.0174 | +0.0162 | +0.0124 | +0.0083 |
| 90 | +0.0048 | +0.0045 | +0.0032 | +0.0017 | +0.0006 | +0.0047 | +0.0044 | +0.0035 | +0.0024 | +0.0017 |

## Synthetic images, 4:4:4: the best mu, with the mmse centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 30: 4, 100: 2, 300: 2, 1000: 2, 3000: 2 |  |
| 20 | 144.3 | 30: 1, 100: 3, 300: 5, 1000: 3 |  |
| 30 | 95.7 | 30: 2, 100: 7, 300: 1, 1000: 2 |  |
| 50 | 57.6 | 30: 5, 100: 5, 300: 2 |  |
| 70 | 34.5 | 30: 7, 100: 3, 300: 2 |  |
| 90 | 11.5 | 30: 9, 100: 3 |  |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.60, and $\mu = 8.19$ at a mean step of 1.

## Synthetic images, 4:4:4: the best mu, with the midpoint centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 300: 3, 1000: 5, 3000: 4 |  |
| 20 | 144.3 | 300: 5, 1000: 6, 3000: 1 |  |
| 30 | 95.7 | 100: 2, 300: 6, 1000: 4 |  |
| 50 | 57.6 | 100: 5, 300: 4, 1000: 3 |  |
| 70 | 34.5 | 30: 2, 100: 3, 300: 7 |  |
| 90 | 11.5 | 30: 6, 100: 6 |  |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.94, and $\mu = 5.65$ at a mean step of 1.

## Synthetic images, the weights mu / Q: the 8-bit PSNR against the standard decoder

Median over the files, by quality, of the 8-bit PSNR less the standard decoder's (dB).

| quality | midpoint 1 | midpoint 3 | midpoint 10 | midpoint 30 | midpoint 100 |
|---|---|---|---|---|---|
| 10 | +0.63 | +0.82 | +1.04 | +0.67 | +0.12 |
| 20 | +0.98 | +1.26 | +1.41 | +0.82 | +0.24 |
| 30 | +1.73 | +2.10 | +2.18 | +1.26 | +0.39 |
| 50 | +1.72 | +2.42 | +2.57 | +1.47 | +0.45 |
| 70 | +1.93 | +2.80 | +2.96 | +1.54 | +0.46 |
| 90 | +2.26 | +3.46 | +3.38 | +1.64 | +0.51 |

## Synthetic images, the weights mu / Q: the SSIM against the standard decoder

Likewise, of the SSIM of the 8-bit result.

| quality | midpoint 1 | midpoint 3 | midpoint 10 | midpoint 30 | midpoint 100 |
|---|---|---|---|---|---|
| 10 | +0.0445 | +0.0471 | +0.0454 | +0.0278 | +0.0117 |
| 20 | +0.0365 | +0.0378 | +0.0358 | +0.0205 | +0.0065 |
| 30 | +0.0310 | +0.0318 | +0.0302 | +0.0176 | +0.0062 |
| 50 | +0.0233 | +0.0238 | +0.0225 | +0.0130 | +0.0044 |
| 70 | +0.0151 | +0.0156 | +0.0145 | +0.0079 | +0.0026 |
| 90 | +0.0035 | +0.0036 | +0.0033 | +0.0018 | +0.0006 |

## Synthetic images, the weights mu / Q: the best mu, with the midpoint centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 3: 2, 10: 4, 30: 2 |  |
| 20 | 144.3 | 3: 3, 10: 5 |  |
| 30 | 95.7 | 3: 4, 10: 4 |  |
| 50 | 57.6 | 3: 4, 10: 4 |  |
| 70 | 34.5 | 3: 4, 10: 4 |  |
| 90 | 11.5 | 3: 4, 10: 4 |  |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.15, and $\mu = 3.31$ at a mean step of 1.

## Synthetic images, DC weighted as AC: the 8-bit PSNR against the standard decoder

Median over the files, by quality, of the 8-bit PSNR less the standard decoder's (dB).

| quality | midpoint 100 | midpoint 300 | midpoint 1000 |
|---|---|---|---|
| 10 | +0.99 | +1.24 | +1.39 |
| 20 | +1.34 | +1.62 | +1.75 |
| 30 | +2.25 | +2.70 | +2.56 |
| 50 | +2.39 | +2.92 | +2.45 |
| 70 | +2.99 | +3.41 | +2.12 |
| 90 | +3.87 | +2.48 | +0.90 |

## Synthetic images, DC weighted as AC: the SSIM against the standard decoder

Likewise, of the SSIM of the 8-bit result.

| quality | midpoint 100 | midpoint 300 | midpoint 1000 |
|---|---|---|---|
| 10 | +0.0468 | +0.0494 | +0.0428 |
| 20 | +0.0371 | +0.0370 | +0.0305 |
| 30 | +0.0312 | +0.0306 | +0.0243 |
| 50 | +0.0232 | +0.0221 | +0.0161 |
| 70 | +0.0155 | +0.0145 | +0.0096 |
| 90 | +0.0035 | +0.0025 | +0.0011 |

## Synthetic images, DC weighted as AC: the best mu, with the midpoint centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 300: 1, 1000: 7 |  |
| 20 | 144.3 | 300: 3, 1000: 5 |  |
| 30 | 95.7 | 300: 4, 1000: 4 |  |
| 50 | 57.6 | 100: 1, 300: 3, 1000: 4 |  |
| 70 | 34.5 | 100: 2, 300: 6 |  |
| 90 | 11.5 | 100: 8 |  |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.68, and $\mu = 21.8$ at a mean step of 1.

## Synthetic images, TGV: the 8-bit PSNR against the standard decoder

Median over the files, by quality, of the 8-bit PSNR less the standard decoder's (dB).

| quality | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 |
|---|---|---|---|---|
| 10 | +0.34 | +0.54 | +0.97 | +1.29 |
| 20 | +0.91 | +1.29 | +1.60 | +1.71 |
| 30 | +1.69 | +2.17 | +2.58 | +2.40 |
| 50 | +1.54 | +2.39 | +2.89 | +2.41 |
| 70 | +2.00 | +2.94 | +3.29 | +1.95 |
| 90 | +3.34 | +3.84 | +2.40 | +0.88 |

## Synthetic images, TGV: the SSIM against the standard decoder

Likewise, of the SSIM of the 8-bit result.

| quality | midpoint 30 | midpoint 100 | midpoint 300 | midpoint 1000 |
|---|---|---|---|---|
| 10 | +0.0421 | +0.0446 | +0.0480 | +0.0457 |
| 20 | +0.0357 | +0.0371 | +0.0377 | +0.0321 |
| 30 | +0.0305 | +0.0314 | +0.0313 | +0.0258 |
| 50 | +0.0230 | +0.0236 | +0.0229 | +0.0170 |
| 70 | +0.0151 | +0.0155 | +0.0145 | +0.0095 |
| 90 | +0.0036 | +0.0035 | +0.0025 | +0.0010 |

## Synthetic images, TGV: the best mu, with the midpoint centres

For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit PSNR), and the median gain of the best over the former default (MMSE centres, $\mu = 10^{-3}$).

| quality | mean step | best mu: files | gain (dB) |
|---|---|---|---|
| 10 | 288.1 | 1000: 8 |  |
| 20 | 144.3 | 300: 4, 1000: 4 |  |
| 30 | 95.7 | 300: 4, 1000: 4 |  |
| 50 | 57.6 | 100: 1, 300: 3, 1000: 4 |  |
| 70 | 34.5 | 100: 2, 300: 6 |  |
| 90 | 11.5 | 30: 2, 100: 6 |  |

The line of $\log \mu$ on $\log \bar Q$ over the files: slope 0.78, and $\mu = 13.7$ at a mean step of 1.

## The rule

$\mu = 9\, \bar Q^{0.9}$ with the middles as centres, by quality: the files, the median gain of the 8-bit PSNR and SSIM over the standard decoder, and what the rule gives up against each file's best $\mu$ of the grid (median / largest, dB), where there is a grid.

### Synthetic images, grey

| quality | files | PSNR | SSIM | given up |
|---|---|---|---|---|
| 10 | 8 | +1.25 | +0.0410 | +0.07 / +0.26 |
| 20 | 8 | +1.74 | +0.0337 | +0.05 / +0.31 |
| 30 | 8 | +2.60 | +0.0294 | +0.09 / +0.30 |
| 50 | 8 | +2.86 | +0.0225 | +0.08 / +0.27 |
| 70 | 8 | +3.30 | +0.0150 | +0.12 / +0.33 |
| 90 | 8 | +3.93 | +0.0035 | +0.06 / +0.40 |

### Synthetic images, grey, mozjpeg

| quality | files | PSNR | SSIM | given up |
|---|---|---|---|---|
| 10 | 8 | -0.02 | +0.0083 |  |
| 20 | 8 | +0.22 | +0.0076 |  |
| 30 | 8 | +0.31 | +0.0069 |  |
| 50 | 8 | +0.56 | +0.0064 |  |
| 70 | 8 | +0.82 | +0.0041 |  |
| 90 | 8 | +1.86 | +0.0017 |  |

### Photographs, grey

| quality | files | PSNR | SSIM | given up |
|---|---|---|---|---|
| 10 | 24 | +0.19 | +0.0096 | +0.05 / +0.18 |
| 20 | 24 | +0.29 | +0.0051 | +0.00 / +0.21 |
| 30 | 24 | +0.34 | +0.0036 | +0.01 / +0.14 |
| 50 | 24 | +0.49 | +0.0034 | +0.01 / +0.24 |
| 70 | 24 | +0.53 | +0.0034 | +0.08 / +0.35 |
| 90 | 24 | +1.42 | +0.0059 | +0.10 / +0.50 |

### Synthetic images, 4:2:0

| quality | files | PSNR | SSIM | given up |
|---|---|---|---|---|
| 10 | 12 | +0.62 | +0.0506 | +0.11 / +0.69 |
| 20 | 12 | +1.30 | +0.0438 | +0.07 / +0.80 |
| 30 | 12 | +1.73 | +0.0385 | +0.10 / +0.89 |
| 50 | 12 | +2.29 | +0.0278 | +0.10 / +1.04 |
| 70 | 12 | +2.85 | +0.0186 | +0.17 / +1.12 |
| 90 | 12 | +3.98 | +0.0084 | +0.09 / +1.40 |

### Synthetic images, 4:4:4

| quality | files | PSNR | SSIM | given up |
|---|---|---|---|---|
| 10 | 12 | +0.67 | +0.0461 | +0.10 / +0.58 |
| 20 | 12 | +0.97 | +0.0369 | +0.07 / +0.83 |
| 30 | 12 | +2.20 | +0.0338 | +0.12 / +1.02 |
| 50 | 12 | +2.79 | +0.0261 | +0.15 / +1.18 |
| 70 | 12 | +3.27 | +0.0172 | +0.17 / +1.38 |
| 90 | 12 | +3.40 | +0.0046 | +0.19 / +1.81 |

## The rule, read off the grids

Each file's gain of the 8-bit PSNR over the standard decoder, with the middles, as a function of $\log \mu$, piecewise linear between the $\mu$ of its grid and flat beyond them. A rule $\mu = s \bar Q^r$ gains what the curves give at its $\mu$: its mean over the files, and its loss to each file's best point of the grid (median / largest, dB). The best rule is that of the largest mean, over $s$ from 1 to 300 and $r$ from 0.3 to 1.2; for every set together, the mean of the sets' means.

| files | count | mean gain at 9, 0.9 | loss (median / largest) | best scale, power | its mean gain |
|---|---|---|---|---|---|
| synthetic, grey | 48 | +2.530 | 0.167 / 0.572 | 18.4, 0.7 | +2.562 |
| photographs, grey | 144 | +0.542 | 0.079 / 0.500 | 14.5, 0.85 | +0.583 |
| synthetic, 4:2:0 | 72 | +2.454 | 0.126 / 1.081 | 18.4, 0.675 | +2.486 |
| synthetic, 4:4:4 | 72 | +2.265 | 0.139 / 1.428 | 2.26, 1.07 | +2.317 |
| every set, each a weight of 1 | 336 | +1.948 | 0.118 / 1.428 | 16.3, 0.725 | +1.978 |

## The chroma's weight

With the middles and $\mu$ by the rule, each component's of its own table, the chroma's times a factor: by sampling, quality and factor, the median gain of the 8-bit PSNR over the standard decoder (dB).

### Synthetic images

| sampling | quality | 0.1 | 0.3 | 1 | 3 | 10 |
|---|---|---|---|---|---|---|
| 420 | 10 | +0.75 | +0.72 | +0.62 | +0.45 | +0.31 |
| 420 | 20 | +1.26 | +1.41 | +1.30 | +0.93 | +0.71 |
| 420 | 30 | +2.08 | +1.99 | +1.73 | +1.45 | +1.24 |
| 420 | 50 | +2.41 | +2.42 | +2.29 | +2.10 | +1.76 |
| 420 | 70 | +2.90 | +2.91 | +2.85 | +2.60 | +1.93 |
| 420 | 90 | +4.11 | +4.14 | +3.98 | +3.54 | +2.65 |
| 444 | 10 | +0.69 | +0.79 | +0.67 | +0.54 | +0.42 |
| 444 | 20 | +0.84 | +1.10 | +0.97 | +0.76 | +0.64 |
| 444 | 30 | +2.42 | +2.46 | +2.20 | +1.71 | +1.27 |
| 444 | 50 | +2.94 | +3.02 | +2.79 | +2.15 | +1.70 |
| 444 | 70 | +3.62 | +3.66 | +3.27 | +2.42 | +1.59 |
| 444 | 90 | +3.87 | +3.79 | +3.40 | +2.64 | +1.59 |

### Photographs

| sampling | quality | 0.1 | 0.3 | 1 |
|---|---|---|---|---|
| 420 | 10 | +0.24 | +0.21 | +0.11 |
| 420 | 20 | +0.27 | +0.25 | +0.21 |
| 420 | 30 | +0.32 | +0.31 | +0.25 |
| 420 | 50 | +0.48 | +0.45 | +0.39 |
| 420 | 70 | +0.56 | +0.53 | +0.47 |
| 420 | 90 | +1.07 | +1.09 | +1.01 |
| 444 | 10 | +0.27 | +0.23 | +0.17 |
| 444 | 20 | +0.31 | +0.27 | +0.24 |
| 444 | 30 | +0.39 | +0.36 | +0.31 |
| 444 | 50 | +0.51 | +0.48 | +0.46 |
| 444 | 70 | +0.52 | +0.51 | +0.50 |
| 444 | 90 | +1.23 | +1.31 | +1.34 |

## The weight of DC

With the middles and $\mu$ by the rule, DC unweighted (0) and weighted as AC (1): by quality, the median gain of the 8-bit PSNR over the standard decoder (dB), and the median iterations.

### Synthetic images, grey

| quality | PSNR, DC 0 | PSNR, DC 1 | iterations, DC 0 | iterations, DC 1 |
|---|---|---|---|---|
| 10 | +1.25 | +1.36 | 1090 | 1810 |
| 20 | +1.74 | +1.78 | 890 | 1825 |
| 30 | +2.60 | +2.75 | 985 | 1890 |
| 50 | +2.86 | +2.95 | 795 | 1900 |
| 70 | +3.30 | +3.41 | 1030 | 1640 |
| 90 | +3.93 | +3.95 | 1795 | 1625 |

### Photographs, grey

| quality | PSNR, DC 0 | PSNR, DC 1 | iterations, DC 0 | iterations, DC 1 |
|---|---|---|---|---|
| 10 | +0.19 | +0.48 | 410 | 530 |
| 20 | +0.29 | +0.48 | 255 | 320 |
| 30 | +0.34 | +0.49 | 205 | 225 |
| 50 | +0.49 | +0.59 | 175 | 165 |
| 70 | +0.53 | +0.59 | 135 | 130 |
| 90 | +1.42 | +1.45 | 80 | 80 |

### Synthetic images, 4:2:0

| quality | PSNR, DC 0 | PSNR, DC 1 | iterations, DC 0 | iterations, DC 1 |
|---|---|---|---|---|
| 10 | +0.62 | +0.97 | 2065 | 1200 |
| 20 | +1.30 | +1.46 | 1510 | 1140 |
| 30 | +1.73 | +1.75 | 1160 | 1010 |
| 50 | +2.29 | +2.36 | 1010 | 970 |
| 70 | +2.85 | +2.95 | 1010 | 840 |
| 90 | +3.98 | +4.16 | 1035 | 715 |

### Synthetic images, 4:4:4

| quality | PSNR, DC 0 | PSNR, DC 1 | iterations, DC 0 | iterations, DC 1 |
|---|---|---|---|---|
| 10 | +0.67 | +1.03 | 1715 | 640 |
| 20 | +0.97 | +1.32 | 860 | 600 |
| 30 | +2.20 | +2.35 | 690 | 570 |
| 50 | +2.79 | +3.05 | 525 | 520 |
| 70 | +3.27 | +3.61 | 565 | 465 |
| 90 | +3.40 | +3.53 | 580 | 325 |

## The proposed defaults

The middles, $\mu$ by the rule, DC weighted as AC, and in colour files the chroma's $\mu$ 0.3 times the rule's; of the greyscale files, against the former defaults (MMSE centres, $\mu = 10^{-3}$, DC unweighted). By quality, the median gain of the 8-bit PSNR and of the SSIM over the standard decoder, and the median iterations, of each.

### Synthetic images, grey, TV

| quality | PSNR | former | SSIM | former | iterations | former |
|---|---|---|---|---|---|---|
| 10 | +1.36 | +0.41 | +0.0369 | +0.0417 | 1810 | 2585 |
| 20 | +1.78 | +0.58 | +0.0327 | +0.0348 | 1825 | 1885 |
| 30 | +2.75 | +1.26 | +0.0287 | +0.0297 | 1890 | 1690 |
| 50 | +2.95 | +0.96 | +0.0217 | +0.0223 | 1900 | 1455 |
| 70 | +3.41 | +0.97 | +0.0150 | +0.0145 | 1640 | 1505 |
| 90 | +3.95 | +0.84 | +0.0035 | +0.0033 | 1625 | 1630 |

### Synthetic images, grey, TGV

| quality | PSNR | former | SSIM | former | iterations | former |
|---|---|---|---|---|---|---|
| 10 | +1.37 | +0.24 | +0.0383 | +0.0406 | 870 | 3905 |
| 20 | +1.79 | +0.68 | +0.0336 | +0.0347 | 830 | 3310 |
| 30 | +2.76 | +1.30 | +0.0290 | +0.0296 | 775 | 2990 |
| 50 | +2.95 | +0.86 | +0.0216 | +0.0222 | 765 | 2590 |
| 70 | +3.29 | +1.03 | +0.0148 | +0.0145 | 670 | 2290 |
| 90 | +3.97 | +0.80 | +0.0035 | +0.0033 | 805 | 1650 |

### Photographs, grey, TV

| quality | PSNR | former | SSIM | former | iterations | former |
|---|---|---|---|---|---|---|
| 10 | +0.48 | -0.46 | +0.0147 | -0.0201 | 530 | 1435 |
| 20 | +0.48 | -0.62 | +0.0082 | -0.0279 | 320 | 785 |
| 30 | +0.49 | -0.65 | +0.0061 | -0.0247 | 225 | 645 |
| 50 | +0.59 | -0.76 | +0.0044 | -0.0205 | 165 | 425 |
| 70 | +0.59 | -0.92 | +0.0039 | -0.0148 | 130 | 315 |
| 90 | +1.45 | -0.90 | +0.0060 | -0.0036 | 80 | 175 |

### Photographs, grey, TGV

| quality | PSNR | former | SSIM | former | iterations | former |
|---|---|---|---|---|---|---|
| 10 | +0.56 | -0.33 | +0.0172 | -0.0122 | 685 | 1475 |
| 20 | +0.52 | -0.50 | +0.0098 | -0.0199 | 745 | 1145 |
| 30 | +0.51 | -0.60 | +0.0067 | -0.0196 | 845 | 1085 |
| 50 | +0.60 | -0.69 | +0.0048 | -0.0175 | 875 | 1115 |
| 70 | +0.61 | -0.84 | +0.0041 | -0.0141 | 950 | 965 |
| 90 | +1.48 | -0.87 | +0.0060 | -0.0034 | 1025 | 975 |

### Synthetic images, 4:2:0, TV

| quality | PSNR | SSIM | iterations |
|---|---|---|---|
| 10 | +1.51 | +0.0512 | 1260 |
| 20 | +1.89 | +0.0470 | 1125 |
| 30 | +2.05 | +0.0395 | 1060 |
| 50 | +2.50 | +0.0284 | 1080 |
| 70 | +3.03 | +0.0190 | 1050 |
| 90 | +4.32 | +0.0085 | 980 |

### Synthetic images, 4:2:0, TGV

| quality | PSNR | SSIM | iterations |
|---|---|---|---|
| 10 | +1.48 | +0.0495 | 1055 |
| 20 | +2.01 | +0.0466 | 975 |
| 30 | +2.11 | +0.0391 | 925 |
| 50 | +2.67 | +0.0282 | 875 |
| 70 | +3.10 | +0.0189 | 795 |
| 90 | +4.48 | +0.0085 | 725 |

### Synthetic images, 4:4:4, TV

| quality | PSNR | SSIM | iterations |
|---|---|---|---|
| 10 | +1.17 | +0.0429 | 600 |
| 20 | +1.61 | +0.0378 | 555 |
| 30 | +2.66 | +0.0344 | 530 |
| 50 | +3.55 | +0.0265 | 515 |
| 70 | +3.95 | +0.0175 | 480 |
| 90 | +3.91 | +0.0047 | 365 |

### Synthetic images, 4:4:4, TGV

| quality | PSNR | SSIM | iterations |
|---|---|---|---|
| 10 | +1.27 | +0.0450 | 390 |
| 20 | +1.71 | +0.0382 | 390 |
| 30 | +2.92 | +0.0347 | 380 |
| 50 | +3.69 | +0.0265 | 365 |
| 70 | +4.09 | +0.0175 | 350 |
| 90 | +4.00 | +0.0047 | 395 |

### Photographs, 4:2:0, TV

| quality | PSNR | SSIM | iterations |
|---|---|---|---|
| 10 | +0.70 | +0.0209 | 955 |
| 20 | +0.63 | +0.0129 | 950 |
| 30 | +0.59 | +0.0089 | 930 |
| 50 | +0.58 | +0.0066 | 910 |
| 70 | +0.64 | +0.0056 | 940 |
| 90 | +1.13 | +0.0069 | 820 |

### Photographs, 4:2:0, TGV

| quality | PSNR | SSIM | iterations |
|---|---|---|---|
| 10 | +0.66 | +0.0235 | 990 |
| 20 | +0.62 | +0.0143 | 1020 |
| 30 | +0.59 | +0.0101 | 1095 |
| 50 | +0.60 | +0.0072 | 1115 |
| 70 | +0.64 | +0.0059 | 1130 |
| 90 | +1.15 | +0.0070 | 1150 |

### Photographs, 4:4:4, TV

| quality | PSNR | SSIM | iterations |
|---|---|---|---|
| 10 | +0.67 | +0.0234 | 215 |
| 20 | +0.62 | +0.0145 | 145 |
| 30 | +0.59 | +0.0105 | 105 |
| 50 | +0.61 | +0.0076 | 70 |
| 70 | +0.65 | +0.0059 | 50 |
| 90 | +1.40 | +0.0070 | 40 |

### Photographs, 4:4:4, TGV

| quality | PSNR | SSIM | iterations |
|---|---|---|---|
| 10 | +0.69 | +0.0280 | 300 |
| 20 | +0.64 | +0.0168 | 320 |
| 30 | +0.60 | +0.0119 | 330 |
| 50 | +0.62 | +0.0080 | 325 |
| 70 | +0.65 | +0.0060 | 370 |
| 90 | +1.42 | +0.0071 | 435 |

## The slack

With the middles and $\mu$ by the rule: by quality, the median gain of the 8-bit PSNR over the standard decoder (dB), without slack and with each slack (steps on each side) and price per step.

### mozjpeg

| quality | none | 0.5 at 0 | 0.5 at 0.3 | 0.5 at 3 | 0.5 at 30 | 1 at 0 | 1 at 0.3 | 1 at 3 | 1 at 30 | 2 at 0 | 2 at 0.3 | 2 at 3 | 2 at 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | -0.02 | -0.22 | -0.22 | -0.17 | -0.09 | -0.59 | -0.55 | -0.34 | -0.14 | -1.51 | -1.25 | -0.68 | -0.23 |
| 20 | +0.22 | +0.15 | +0.15 | +0.17 | +0.19 | -0.14 | -0.11 | +0.11 | +0.15 | -0.50 | -0.48 | -0.16 | +0.08 |
| 30 | +0.31 | +0.25 | +0.26 | +0.28 | +0.31 | +0.09 | +0.10 | +0.21 | +0.29 | -0.29 | -0.26 | +0.05 | +0.27 |
| 50 | +0.56 | +0.54 | +0.54 | +0.59 | +0.59 | +0.41 | +0.44 | +0.57 | +0.59 | +0.07 | +0.16 | +0.44 | +0.58 |
| 70 | +0.82 | +0.81 | +0.82 | +0.88 | +0.87 | +0.73 | +0.75 | +0.86 | +0.87 | +0.50 | +0.50 | +0.80 | +0.87 |
| 90 | +1.86 | +2.38 | +2.39 | +2.41 | +2.40 | +2.32 | +2.36 | +2.43 | +2.40 | +1.91 | +2.17 | +2.42 | +2.40 |

### libjpeg-turbo

| quality | none | 0.5 at 0 | 0.5 at 0.3 | 0.5 at 3 | 0.5 at 30 | 1 at 0 | 1 at 0.3 | 1 at 3 | 1 at 30 | 2 at 0 | 2 at 0.3 | 2 at 3 | 2 at 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | +1.25 | +0.43 | +0.43 | +0.45 | +0.84 | -0.01 | +0.11 | +0.22 | +0.75 | -1.45 | -0.85 | -0.28 | +0.67 |
| 20 | +1.74 | +1.03 | +1.04 | +1.07 | +1.26 | +0.66 | +0.71 | +0.95 | +1.20 | -0.44 | -0.14 | +0.71 | +1.17 |
| 30 | +2.60 | +1.56 | +1.57 | +1.65 | +2.03 | +1.04 | +1.08 | +1.30 | +2.03 | -0.11 | +0.17 | +1.11 | +2.02 |
| 50 | +2.86 | +1.97 | +2.07 | +2.29 | +2.60 | +1.47 | +1.53 | +2.05 | +2.59 | +0.68 | +0.95 | +1.85 | +2.59 |
| 70 | +3.30 | +2.26 | +2.43 | +2.82 | +3.08 | +1.89 | +2.01 | +2.70 | +3.07 | +1.14 | +1.36 | +2.64 | +3.07 |
| 90 | +3.93 | +3.49 | +3.53 | +3.65 | +3.86 | +3.01 | +3.24 | +3.65 | +3.86 | +2.10 | +2.75 | +3.65 | +3.86 |
