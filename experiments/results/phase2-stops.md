<!-- Written by experiments/phase2_stops.py; do not edit by hand. -->

# Phase 2: when the primal-dual method stops, under the data term of the defaults

- Files: the tuning images as libjpeg-turbo encoded them, of four sets: the synthetic greyscale images, the photographs in grey (BSDS500's train split), and the synthetic and the photographed colour images in 4:2:0 and 4:4:4.
- The model is the library's default (docs/math.md, 4.1); the method is the relaxed one of 5, with the ratio of the steps $\tau/\sigma$ and the relaxation $\rho$; the gap per sample is that of 6.4 (the partial gap of 6.6 where samples are free).
- The reference implementation, unround 0.1.0 (Rust, C interface); NumPy 2.5.3, scikit-image 0.26.0.

## The ratio of the steps

Two files of each set, with the relaxation 1.9, until the gap per sample is within the floor or for 6000 iterations: for each tolerance, the iterations the files took to reach it, in all and the most of one file; where a file did not reach it, how many did.

### TV

| gap per sample ≤ | $\tau/\sigma = 3$ | $\tau/\sigma = 10$ | $\tau/\sigma = 30$ | $\tau/\sigma = 100$ |
|---|---|---|---|---|
| 0.01 | 1970 (1000) | 1300 (640) | 1050 (480) | 1090 (420) |
| 0.005 | 2190 (1050) | 1500 (690) | 1290 (530) | 1470 (500) |
| 0.002 | 2600 (1100) | 1880 (750) | 1830 (630) | 2300 (600) |
| 0.001 | 3040 (1310) | 2390 (840) | 2500 (730) | 3440 (930) |
| 0.0005 | 3670 (1590) | 3160 (990) | 3620 (950) | 5350 (1600) |
| 0.0002 | 4940 (2000) | 5050 (1520) | 6540 (1810) | 10280 (3290) |
| 0.0001 | 6830 (2350) | 7500 (2330) | 10280 (3050) | 16950 (5550) |
| 5e-05 | 9070 (2870) | 11100 (3390) | 16100 (4970) | 6 of 8 |
| 2e-05 | 13490 (4060) | 18280 (5300) | 6 of 8 | 5 of 8 |
| 1e-05 | 18730 (5780) | 6 of 8 | 5 of 8 | 4 of 8 |

Milliseconds an iteration, the median of each set with 10 runs at a time: synthetic, grey 3.4, photographs, grey 2.3, synthetic, colour 11.1, photographs, colour 7.8.

### TGV

| gap per sample ≤ | $\tau/\sigma = 1$ | $\tau/\sigma = 3$ | $\tau/\sigma = 10$ | $\tau/\sigma = 30$ |
|---|---|---|---|---|
| 0.1 | 5680 (2120) | 3390 (1260) | 2260 (710) | 1920 (570) |
| 0.05 | 6220 (2180) | 3900 (1420) | 2850 (870) | 2480 (660) |
| 0.02 | 7160 (2380) | 4660 (1470) | 4040 (1110) | 3990 (890) |
| 0.01 | 8180 (2450) | 6070 (1610) | 5600 (1230) | 6080 (1070) |
| 0.005 | 10140 (2570) | 8410 (1830) | 8140 (1480) | 9660 (1730) |
| 0.002 | 15300 (3410) | 13860 (2460) | 15290 (3080) | 18720 (3910) |
| 0.001 | 21970 (4300) | 20940 (3810) | 24960 (5330) | 7 of 8 |
| 0.0005 | 7 of 8 | 7 of 8 | 6 of 8 | 5 of 8 |
| 0.0002 | 4 of 8 | 2 of 8 | 2 of 8 | 2 of 8 |
| 0.0001 | 2 of 8 | 2 of 8 | 2 of 8 | 0 of 8 |

Milliseconds an iteration, the median of each set with 10 runs at a time: synthetic, grey 9.1, photographs, grey 7.2, synthetic, colour 32.0, photographs, colour 34.9.

## Stopping at a tolerance

Of each set, four images (two for TGV) at the qualities 10, 50, 90. Stopping where the gap per sample first falls within a tolerance changes the result, against the reference's last point. A tolerance is acceptable when, over the files, the change of the PSNR of the binary64 result is at most 0.001 dB in the median and 0.005 dB in the 90th percentile, and that of the SSIM of its 8-bit samples at most 1e-05 and 5e-05; every file has to reach it, and the references have to tell it: their median last gap per sample at most 1/10 of it. For each variant, the largest acceptable tolerance whose smaller ones are acceptable too; the variant is the one that then stops the files in the fewest iterations in all, and the most iterations twice the most a file took, rounded up to 1, 2 or 5 times a power of ten.

### TV

References: $\tau/\sigma = 3$, $\rho = 1.5$, until the gap per sample is within 2e-06 or for 8000 iterations (13 of 48 files took them all); their last gap per sample 2.0e-06 in the median and 1.8e-05 at most: they tell the tolerances from 2.0e-05.

| variant | gap per sample ≤ | files that reached it | iterations: median (least to largest) | in all | PSNR change (dB): median / p90 / largest | SSIM change: median / p90 / largest | acceptable |
|---|---|---|---|---|---|---|---|
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.01 | 48 of 48 | 60 (10 to 850) | 8600 | 0.0119 / 0.0569 / 0.1030 | 3.3e-05 / 2.2e-04 / 6.6e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.005 | 48 of 48 | 65 (10 to 1030) | 10320 | 0.0078 / 0.0461 / 0.0814 | 1.9e-05 / 2.1e-04 / 5.4e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.002 | 48 of 48 | 90 (10 to 1190) | 13030 | 0.0022 / 0.0242 / 0.0827 | 9.9e-06 / 6.8e-05 / 1.6e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.001 | 48 of 48 | 130 (10 to 1400) | 15460 | 0.0017 / 0.0148 / 0.0743 | 4.7e-06 / 4.2e-05 / 1.3e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.0005 | 48 of 48 | 195 (20 to 1700) | 19090 | 0.0006 / 0.0076 / 0.0579 | 2.8e-06 / 2.0e-05 / 1.9e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.0002 | 48 of 48 | 370 (20 to 2090) | 26800 | 0.0002 / 0.0054 / 0.0468 | 1.3e-06 / 8.3e-06 / 1.5e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.0001 | 48 of 48 | 565 (20 to 2470) | 36750 | 0.0001 / 0.0028 / 0.0351 | 1.3e-06 / 9.9e-06 / 1.7e-04 | yes |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 5e-05 | 48 of 48 | 830 (30 to 3320) | 51850 | 0.0000 / 0.0017 / 0.0226 | 7.4e-07 / 7.0e-06 / 1.9e-04 | yes |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.01 | 48 of 48 | 55 (10 to 540) | 6370 | 0.0017 / 0.0398 / 0.0705 | 1.6e-05 / 1.4e-04 / 7.7e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.005 | 48 of 48 | 65 (10 to 620) | 7670 | 0.0014 / 0.0312 / 0.0610 | 1.1e-05 / 8.1e-05 / 6.6e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.002 | 48 of 48 | 105 (10 to 730) | 9980 | 0.0005 / 0.0140 / 0.0592 | 7.7e-06 / 5.7e-05 / 2.0e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.001 | 48 of 48 | 150 (20 to 840) | 12780 | 0.0003 / 0.0096 / 0.0576 | 3.6e-06 / 3.8e-05 / 1.5e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.0005 | 48 of 48 | 265 (20 to 1030) | 17290 | 0.0002 / 0.0057 / 0.0566 | 3.6e-06 / 2.0e-05 / 1.8e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.0002 | 48 of 48 | 575 (30 to 1660) | 28420 | 0.0001 / 0.0027 / 0.0485 | 1.3e-06 / 8.3e-06 / 1.7e-04 | yes |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.0001 | 48 of 48 | 970 (30 to 3590) | 43400 | 0.0001 / 0.0015 / 0.0352 | 1.1e-06 / 7.7e-06 / 1.9e-04 | yes |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 5e-05 | 48 of 48 | 1270 (40 to 6060) | 68360 | 0.0000 / 0.0005 / 0.0228 | 7.9e-07 / 5.4e-06 / 2.0e-04 | yes |

The choice (accepted): $\tau/\sigma = 10$, $\rho = 1.9$, a gap per sample of 0.0002, and at most 5000 iterations.

By set, at the choice:

| set | files | iterations: median | most | PSNR change (dB): median / p90 / largest | iterations at the former stop ($\tau/\sigma = 10$, $\rho = 1.9$ at 0.0005) |
|---|---|---|---|---|---|
| synthetic, grey | 12 | 800 | 1660 | 0.0003 / 0.0018 / 0.0030 | 370 |
| photographs, grey | 12 | 130 | 660 | 0.0001 / 0.0003 / 0.0005 | 80 |
| synthetic, colour | 12 | 665 | 1340 | 0.0002 / 0.0284 / 0.0485 | 445 |
| photographs, colour | 12 | 605 | 1270 | 0.0000 / 0.0024 / 0.0048 | 495 |

### TGV

References: $\tau/\sigma = 3$, $\rho = 1.5$, until the gap per sample is within 0.0001 or for 8000 iterations (13 of 24 files took them all); their last gap per sample 1.9e-04 in the median and 1.4e-03 at most: they tell the tolerances from 1.9e-03.

| variant | gap per sample ≤ | files that reached it | iterations: median (least to largest) | in all | PSNR change (dB): median / p90 / largest | SSIM change: median / p90 / largest | acceptable |
|---|---|---|---|---|---|---|---|
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.1 | 24 of 24 | 320 (70 to 1060) | 9240 | 0.0031 / 0.0444 / 0.2668 | 1.6e-05 / 1.2e-04 / 2.9e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.05 | 24 of 24 | 355 (100 to 1150) | 10720 | 0.0019 / 0.0261 / 0.2221 | 1.0e-05 / 9.1e-05 / 1.3e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.02 | 24 of 24 | 460 (120 to 1590) | 14740 | 0.0012 / 0.0168 / 0.1803 | 4.1e-06 / 7.2e-05 / 9.1e-05 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.01 | 24 of 24 | 625 (170 to 1650) | 18180 | 0.0008 / 0.0102 / 0.1512 | 5.8e-06 / 3.3e-05 / 5.8e-05 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.005 | 24 of 24 | 1075 (190 to 1900) | 25150 | 0.0003 / 0.0083 / 0.1268 | 6.8e-06 / 2.5e-05 / 4.1e-05 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.002 | 24 of 24 | 1965 (330 to 2780) | 40230 | 0.0002 / 0.0047 / 0.0926 | 5.8e-06 / 1.8e-05 / 3.4e-05 | yes |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.001 | 24 of 24 | 2610 (520 to 5610) | 60780 | 0.0001 / 0.0024 / 0.0504 | 2.5e-06 / 1.2e-05 / 2.5e-05 | not told |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.1 | 24 of 24 | 240 (60 to 660) | 7200 | 0.0057 / 0.0873 / 0.4643 | 2.1e-05 / 1.9e-04 / 3.0e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.05 | 24 of 24 | 285 (80 to 870) | 8750 | 0.0035 / 0.0619 / 0.3819 | 1.6e-05 / 1.3e-04 / 1.4e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.02 | 24 of 24 | 450 (130 to 1080) | 12110 | 0.0026 / 0.0358 / 0.3344 | 1.5e-05 / 5.9e-05 / 1.3e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.01 | 24 of 24 | 680 (190 to 1280) | 16870 | 0.0018 / 0.0199 / 0.2887 | 7.2e-06 / 5.0e-05 / 1.2e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.005 | 24 of 24 | 1055 (260 to 1790) | 24600 | 0.0008 / 0.0132 / 0.2480 | 6.2e-06 / 3.3e-05 / 9.1e-05 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.002 | 24 of 24 | 1850 (480 to 3510) | 42230 | 0.0003 / 0.0089 / 0.1547 | 6.4e-06 / 1.6e-05 / 5.1e-05 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.001 | 24 of 24 | 2805 (690 to 5310) | 66350 | 0.0002 / 0.0045 / 0.0677 | 5.2e-06 / 2.1e-05 / 4.3e-05 | not told |

The choice (accepted): $\tau/\sigma = 3$, $\rho = 1.9$, a gap per sample of 0.002, and at most 10000 iterations.

By set, at the choice:

| set | files | iterations: median | most | PSNR change (dB): median / p90 / largest | iterations at the former stop ($\tau/\sigma = 3$, $\rho = 1.9$ at 0.005) |
|---|---|---|---|---|---|
| synthetic, grey | 6 | 2135 | 2440 | 0.0003 / 0.0033 / 0.0053 | 1125 |
| photographs, grey | 6 | 1820 | 2780 | 0.0000 / 0.0001 / 0.0001 | 965 |
| synthetic, colour | 6 | 1375 | 2210 | 0.0024 / 0.0514 / 0.0926 | 915 |
| photographs, colour | 6 | 1360 | 2190 | 0.0000 / 0.0016 / 0.0030 | 1015 |

## TGV against TV

At their choices, on the files that TGV takes: the median iterations of each, the milliseconds an iteration of the ratios stage (10 runs at a time), and the time of TGV over that of TV.

| set | TGV: iterations | TV: iterations | TGV: ms an iteration | TV: ms an iteration | time, TGV over TV |
|---|---|---|---|---|---|
| synthetic, grey | 2135 | 1050 | 9.1 | 3.4 | 5.4 |
| photographs, grey | 1820 | 90 | 7.2 | 2.3 | 62.0 |
| synthetic, colour | 1375 | 780 | 32.0 | 11.1 | 5.1 |
| photographs, colour | 1360 | 545 | 34.9 | 7.8 | 11.1 |
