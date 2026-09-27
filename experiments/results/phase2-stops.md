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
| 0.01 | 1980 (1060) | 1280 (710) | 1010 (550) | 920 (400) |
| 0.005 | 2150 (1060) | 1450 (710) | 1300 (630) | 1480 (580) |
| 0.002 | 2690 (1200) | 1980 (790) | 2120 (760) | 2990 (940) |
| 0.001 | 3160 (1310) | 2660 (920) | 3200 (1020) | 4530 (1370) |
| 0.0005 | 3820 (1590) | 3620 (1080) | 4400 (1170) | 6930 (2140) |
| 0.0002 | 5250 (2000) | 5560 (1290) | 7650 (2110) | 12630 (3840) |
| 0.0001 | 7420 (2350) | 8750 (2220) | 12490 (3760) | 7 of 8 |
| 5e-05 | 10670 (3140) | 13970 (3730) | 7 of 8 | 5 of 8 |
| 2e-05 | 17550 (5060) | 6 of 8 | 5 of 8 | 5 of 8 |
| 1e-05 | 6 of 8 | 5 of 8 | 5 of 8 | 2 of 8 |

Milliseconds an iteration, the median of each set with 10 runs at a time: synthetic, grey 8.7, photographs, grey 6.9, synthetic, colour 20.3, photographs, colour 20.4.

### TGV

| gap per sample ≤ | $\tau/\sigma = 1$ | $\tau/\sigma = 3$ | $\tau/\sigma = 10$ | $\tau/\sigma = 30$ |
|---|---|---|---|---|
| 0.1 | 5700 (2130) | 3330 (1230) | 2180 (690) | 1740 (430) |
| 0.05 | 6190 (2150) | 3750 (1260) | 2690 (720) | 2360 (550) |
| 0.02 | 7050 (2300) | 4510 (1400) | 3920 (1050) | 3900 (850) |
| 0.01 | 8020 (2400) | 5840 (1620) | 5470 (1260) | 5820 (1170) |
| 0.005 | 10010 (2630) | 8230 (1850) | 8100 (1570) | 9400 (1580) |
| 0.002 | 14810 (3410) | 13450 (2390) | 15600 (2640) | 18420 (3270) |
| 0.001 | 20740 (4290) | 20570 (3640) | 24390 (4730) | 7 of 8 |
| 0.0005 | 7 of 8 | 7 of 8 | 6 of 8 | 4 of 8 |
| 0.0002 | 3 of 8 | 2 of 8 | 2 of 8 | 2 of 8 |
| 0.0001 | 2 of 8 | 2 of 8 | 2 of 8 | 1 of 8 |

Milliseconds an iteration, the median of each set with 10 runs at a time: synthetic, grey 18.2, photographs, grey 12.1, synthetic, colour 42.2, photographs, colour 38.3.

## Stopping at a tolerance

Of each set, four images (two for TGV) at the qualities 10, 50, 90. Stopping where the gap per sample first falls within a tolerance changes the result, against the reference's last point. A tolerance is acceptable when, over the files, the change of the PSNR of the binary64 result is at most 0.001 dB in the median and 0.005 dB in the 90th percentile, and that of the SSIM of its 8-bit samples at most 1e-05 and 5e-05; every file has to reach it, and the references have to tell it: their median last gap per sample at most 1/10 of it. For each variant, the largest acceptable tolerance whose smaller ones are acceptable too; the variant is the one that then stops the files in the fewest iterations in all, and the most iterations twice the most a file took, rounded up to 1, 2 or 5 times a power of ten.

### TV

References: $\tau/\sigma = 3$, $\rho = 1.5$, until the gap per sample is within 2e-06 or for 8000 iterations (22 of 48 files took them all); their last gap per sample 2.0e-06 in the median and 2.2e-05 at most: they tell the tolerances from 2.0e-05.

| variant | gap per sample ≤ | files that reached it | iterations: median (least to largest) | in all | PSNR change (dB): median / p90 / largest | SSIM change: median / p90 / largest | acceptable |
|---|---|---|---|---|---|---|---|
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.01 | 48 of 48 | 40 (10 to 850) | 7680 | 0.0091 / 0.0444 / 0.0688 | 2.1e-05 / 2.0e-04 / 3.8e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.005 | 48 of 48 | 50 (10 to 1000) | 9110 | 0.0064 / 0.0263 / 0.0521 | 1.7e-05 / 1.5e-04 / 2.9e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.002 | 48 of 48 | 70 (10 to 1170) | 11550 | 0.0028 / 0.0136 / 0.0491 | 4.9e-06 / 8.9e-05 / 1.7e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.001 | 48 of 48 | 105 (10 to 1390) | 13920 | 0.0016 / 0.0077 / 0.0384 | 4.1e-06 / 6.4e-05 / 1.6e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.0005 | 48 of 48 | 165 (20 to 1670) | 17630 | 0.0007 / 0.0049 / 0.0224 | 2.2e-06 / 2.7e-05 / 1.4e-04 | yes |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.0002 | 48 of 48 | 355 (20 to 2060) | 26510 | 0.0001 / 0.0023 / 0.0088 | 1.7e-06 / 1.8e-05 / 8.5e-05 | yes |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.0001 | 48 of 48 | 685 (20 to 2410) | 38760 | 0.0000 / 0.0010 / 0.0045 | 1.1e-06 / 1.1e-05 / 4.2e-05 | yes |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 5e-05 | 48 of 48 | 1285 (30 to 3540) | 58140 | 0.0000 / 0.0002 / 0.0030 | 4.8e-07 / 4.5e-06 / 3.3e-05 | yes |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.01 | 48 of 48 | 40 (10 to 570) | 5520 | 0.0034 / 0.0171 / 0.0997 | 1.9e-05 / 3.4e-04 / 7.0e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.005 | 48 of 48 | 60 (10 to 620) | 6690 | 0.0018 / 0.0131 / 0.0965 | 1.2e-05 / 2.2e-04 / 3.6e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.002 | 48 of 48 | 85 (10 to 690) | 8980 | 0.0008 / 0.0071 / 0.0745 | 6.4e-06 / 8.8e-05 / 2.8e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.001 | 48 of 48 | 150 (20 to 830) | 12140 | 0.0003 / 0.0059 / 0.0482 | 3.7e-06 / 4.5e-05 / 2.1e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.0005 | 48 of 48 | 275 (20 to 980) | 17360 | 0.0002 / 0.0047 / 0.0256 | 2.3e-06 / 2.3e-05 / 1.7e-04 | yes |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.0002 | 48 of 48 | 645 (30 to 1570) | 30710 | 0.0001 / 0.0024 / 0.0090 | 1.2e-06 / 2.3e-05 / 8.6e-05 | yes |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.0001 | 48 of 48 | 1010 (30 to 3530) | 50020 | 0.0000 / 0.0009 / 0.0044 | 1.1e-06 / 1.1e-05 / 4.1e-05 | yes |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 5e-05 | 48 of 48 | 1275 (40 to 6460) | 82510 | 0.0000 / 0.0002 / 0.0030 | 9.7e-07 / 4.7e-06 / 3.4e-05 | yes |

The choice (accepted): $\tau/\sigma = 10$, $\rho = 1.9$, a gap per sample of 0.0005, and at most 2000 iterations.

By set, at the choice:

| set | files | iterations: median | most | PSNR change (dB): median / p90 / largest |
|---|---|---|---|---|
| synthetic, grey | 12 | 395 | 790 | 0.0001 / 0.0044 / 0.0051 |
| photographs, grey | 12 | 90 | 560 | 0.0002 / 0.0016 / 0.0071 |
| synthetic, colour | 12 | 400 | 750 | 0.0003 / 0.0123 / 0.0256 |
| photographs, colour | 12 | 480 | 980 | 0.0001 / 0.0031 / 0.0047 |

### TGV

References: $\tau/\sigma = 3$, $\rho = 1.5$, until the gap per sample is within 0.0001 or for 8000 iterations (14 of 24 files took them all); their last gap per sample 2.4e-04 in the median and 9.2e-04 at most: they tell the tolerances from 2.4e-03.

| variant | gap per sample ≤ | files that reached it | iterations: median (least to largest) | in all | PSNR change (dB): median / p90 / largest | SSIM change: median / p90 / largest | acceptable |
|---|---|---|---|---|---|---|---|
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.1 | 24 of 24 | 305 (70 to 1060) | 8480 | 0.0005 / 0.0180 / 0.1341 | 7.9e-06 / 5.7e-05 / 3.9e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.05 | 24 of 24 | 345 (100 to 1170) | 9900 | 0.0002 / 0.0135 / 0.0938 | 7.1e-06 / 4.4e-05 / 2.8e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.02 | 24 of 24 | 425 (120 to 1590) | 13560 | 0.0001 / 0.0088 / 0.0626 | 6.0e-06 / 4.7e-05 / 1.5e-04 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.01 | 24 of 24 | 630 (150 to 1640) | 16710 | 0.0002 / 0.0058 / 0.0503 | 7.1e-06 / 2.0e-05 / 8.0e-05 | no |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.005 | 24 of 24 | 1000 (190 to 1900) | 23200 | 0.0001 / 0.0028 / 0.0344 | 5.3e-06 / 1.8e-05 / 8.9e-05 | yes |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.002 | 24 of 24 | 1890 (330 to 2780) | 38800 | 0.0001 / 0.0010 / 0.0162 | 2.0e-06 / 8.4e-06 / 7.3e-05 | not told |
| $\tau/\sigma = 3$, $\rho = 1.9$ | 0.001 | 24 of 24 | 2530 (500 to 5610) | 60570 | 0.0000 / 0.0006 / 0.0066 | 1.5e-06 / 4.9e-06 / 6.2e-05 | not told |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.1 | 24 of 24 | 220 (60 to 660) | 6480 | 0.0007 / 0.0289 / 0.2389 | 1.4e-05 / 2.8e-04 / 8.4e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.05 | 24 of 24 | 265 (80 to 870) | 7960 | 0.0006 / 0.0237 / 0.1930 | 1.9e-05 / 2.1e-04 / 7.3e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.02 | 24 of 24 | 390 (120 to 1060) | 11010 | 0.0004 / 0.0176 / 0.1376 | 9.3e-06 / 1.3e-04 / 3.2e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.01 | 24 of 24 | 645 (160 to 1300) | 15510 | 0.0002 / 0.0136 / 0.0979 | 6.7e-06 / 8.4e-05 / 1.8e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.005 | 24 of 24 | 1015 (260 to 1790) | 23200 | 0.0002 / 0.0078 / 0.0547 | 5.1e-06 / 4.8e-05 / 1.5e-04 | no |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.002 | 24 of 24 | 1845 (470 to 3510) | 41520 | 0.0001 / 0.0028 / 0.0209 | 2.3e-06 / 1.7e-05 / 1.1e-04 | not told |
| $\tau/\sigma = 10$, $\rho = 1.9$ | 0.001 | 24 of 24 | 2845 (730 to 5310) | 67140 | 0.0001 / 0.0010 / 0.0087 | 3.3e-06 / 8.6e-06 / 7.7e-05 | not told |

The choice (accepted): $\tau/\sigma = 3$, $\rho = 1.9$, a gap per sample of 0.005, and at most 5000 iterations.

By set, at the choice:

| set | files | iterations: median | most | PSNR change (dB): median / p90 / largest | iterations at the former stop ($\tau/\sigma = 10$, $\rho = 1.9$ at 0.01) |
|---|---|---|---|---|---|
| synthetic, grey | 6 | 1000 | 1160 | 0.0003 / 0.0004 / 0.0004 | 645 |
| photographs, grey | 6 | 1000 | 1490 | 0.0000 / 0.0001 / 0.0002 | 690 |
| synthetic, colour | 6 | 680 | 1320 | 0.0025 / 0.0229 / 0.0344 | 420 |
| photographs, colour | 6 | 1015 | 1900 | 0.0000 / 0.0001 / 0.0001 | 610 |

## TGV against TV

At their choices, on the files that TGV takes: the median iterations of each, the milliseconds an iteration of the ratios stage (10 runs at a time), and the time of TGV over that of TV.

| set | TGV: iterations | TV: iterations | TGV: ms an iteration | TV: ms an iteration | time, TGV over TV |
|---|---|---|---|---|---|
| synthetic, grey | 1000 | 550 | 18.2 | 8.7 | 3.8 |
| photographs, grey | 1000 | 50 | 12.1 | 6.9 | 35.2 |
| synthetic, colour | 680 | 400 | 42.2 | 20.3 | 3.5 |
| photographs, colour | 1015 | 435 | 38.3 | 20.4 | 4.4 |
