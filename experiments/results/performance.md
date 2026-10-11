# The speed of the solvers: threads, the sweep's loops, and what limits them

Measured on 2026-10-11 on a laptop: Intel Core i7-1360P (4 performance cores of 2
threads each, CPUs 0 to 7, and 8 efficiency cores, CPUs 8 to 15; 18 MB of L3, 2 MB of
L2 to a performance core; no AVX-512), 28 W sustained and 64 W at most, Ubuntu
26.04.1, Rust 1.99. The reference is built as it ships (x86-64-v3, AVX2 and FMA,
rust/.cargo/config.toml).

How it was measured: the throughput test of rust/unround (`cargo test --release --test
throughput -- --ignored --nocapture`, one thread per performance core with `taskset -c
0,2,4,6`) and the command line, each run twice, before and after alternately, with
nothing else running: a long experiment on the other cores lowers the clock of every
core, and was stopped while these ran. Numbers are ns per sample and iteration of the
solver, without its setup and its records; a sample is a pixel of a channel.

## Before and after

The solvers at the start of the day (commit 164c1ba) and now, on one thread, two runs
each:

| canvas | TGV before | TGV now | TV before | TV now |
|---|---|---|---|---|
| grey 512 x 512 | 12.72, 12.71 | 7.95, 7.94 | 4.62, 4.49 | 3.81, 3.84 |
| grey 2048 x 2048 | 17.48, 17.61 | 11.52, 11.40 | 7.27, 7.24 | 5.79, 5.90 |
| colour 1024 x 1024, 4:2:0 | 14.47, 14.76 | 10.60, 10.88 | 5.32, 5.45 | 5.02, 5.21 |

Now on 2 and 4 threads, one to a performance core:

| canvas | TGV, 2 | TGV, 4 | TV, 2 | TV, 4 |
|---|---|---|---|---|
| grey 512 x 512 | 4.87, 5.34 | 2.93, 3.03 | 2.31, 2.36 | 1.31, 1.29 |
| grey 2048 x 2048 | 8.04, 8.06 | 5.86, 5.93 | 3.99, 4.05 | 2.86, 2.90 |
| colour 1024 x 1024, 4:2:0 | 7.23, 7.21 | 5.46, 4.76 | 3.60, 3.48 | 2.62, 2.54 |

The command line, TGV to its tolerance, in seconds, two runs each (`--threads N`;
the files are the same to the byte as before, on every count of threads):

| file | before | 1 thread | 2 | 4 | 8 (CPUs 0-7) | 16 (all, the default) |
|---|---|---|---|---|---|---|
| chart-2, q30, 4:2:0 | 10.68, 10.68 | 8.32, 8.46 | 5.75, 5.60 | 4.10, 4.13 | 3.95, 4.02 | 4.38, 4.33 |
| Kodak kodim05, q30, 4:2:0 | 32.35, 31.66 | 25.01, 24.17 | 16.77, 16.85 | 11.94, 12.02 | 12.79, 12.78 | 12.67, 12.69 |
| Kodak kodim05, grey, q30 | 21.11, 21.10 | 13.62, 13.71 | 8.91, 8.70 | 6.05, 6.06 | 6.19, 6.17 | 7.14, 7.01 |

On this processor, the four performance cores, one thread each, are about as quick as
all sixteen threads, or quicker: the efficiency cores, and a core's second thread,
take power from the performance cores' clocks.

Left to the system's scheduler, as the command line runs (no `taskset`), in seconds, two
runs each, on 4, 6, 8 and 12 threads and on all 16 (the default):

| file | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| chart-2, q30, 4:2:0 | 3.96, 4.72 | 3.82, 4.08 | 3.29, 3.92 | 3.20, 3.80 | 3.45, 4.10 |
| Kodak kodim05, q30, 4:2:0 | 13.43, 13.26 | 12.73, 12.37 | 12.20, 11.88 | 12.89, 12.33 | 13.42, 12.80 |
| Kodak kodim05, grey, q30 | 8.59, 8.33 | 7.16, 7.07 | 6.34, 6.35 | 6.47, 6.46 | 7.26, 7.16 |

Eight threads are the quickest here, and the default of all sixteen 5 to 15% slower.

## What changed

- **Threads** (`--threads`): the canvas is cut into strips of whole bands, two to a
  thread, which threads take in order, an iteration of a strip at a time; a strip
  copies its neighbours' rows out as they were when the iteration began, so that every
  value is computed as by one sweep (docs/math.md, 5.1).
- **Records on the threads**: a record's rows are taken by the threads at once, in parts
  of whole bands, between the segments of iterations, each row's terms by one thread
  and the rows' sums added in the rows' order. A record taken on a thread of its own
  beside the next iterations took a core from the sweep: on four threads the records
  had been a quarter of the solver's time.
- **One loop for a step and its differences**: TGV's steps of w, p and r took the
  differences of r and of the extrapolated canvas and field into rows of their own, in
  four to six passes over a row; each is now one loop.
- **The projections' scales in the steps' loops**: their square roots and divisions
  overlap the steps' other arithmetic; coupled channels add their squares channel after
  channel, and the last channel's loop makes the scales.
- **The step of w row by row**: it was taken a band ahead into a ring of two bands,
  1.5 MB on a colour canvas 1024 wide, which pushed the rows of the point out of a
  core's L2 before the dual steps came back to them; it is now taken just before each
  row's dual steps, with its extrapolation in the same loop.
- **The relaxations where a row is read last**: x where its extrapolation is made, and
  w in its step, so that the settling of a row takes only the duals.
- **A channel's own projection in the loop of its step**: where a channel's projections
  are its own (one channel, or channels apart), the step is projected and the dual
  relaxed in the step's loop, and the step is not written out at all.

Every change keeps every value's operands and operations, so that the results are the
same to the last bit; the sweeps' tests hold each field to the definitions in the
natural layout, and the strips' tests the strips to one sweep.

## Where the time goes now

TGV on one performance core, by parts of the sweep, from timers around each part (the
settling of a channel's own projections is part of its steps):

| canvas | all | descent | proximal map | w, p and r | settling of coupled channels |
|---|---|---|---|---|---|
| grey 512 x 512 | 7.15 | 0.85 | 1.50 | 4.66 | -- |
| grey 2048 x 2048 | 9.64 | 1.40 | 1.57 | 6.63 | -- |
| colour 1024 x 1024, 4:2:0 | 10.16 | 1.37 | 1.17 | 5.99 | 1.61 |

The steps of w, p and r are more than half. On a grey canvas (or channels apart), each
sample takes two square roots and two divisions, for the projections onto the two
balls, which a core does at one in two to three cycles: about 2.3 ns of a sample at
4.4 GHz, whatever else the loop does. The rest of those loops reads the fields w, r and
p, which the iteration touches there first, from memory on a canvas larger than the L3.

## Why more threads give less

**The clock.** Eight independent chains of 256-bit FMAs per thread, nothing read from
memory, show the clock of a core: 4.40 GHz with one core working, 3.67 GHz with two,
3.01 GHz with four, and 2.8 GHz with four cores of two threads each. A computation that
scales perfectly is thus 1.67 times as quick on two cores and 2.74 times on four. TGV on
grey 2048 x 2048 is 1.42 and 1.94 times as quick (11.5, 8.1 and 5.9 ns), 85 and 71% of
that.

**Memory is not the limit.** An iteration reads and writes each of TGV's eight fields
once: 128 bytes a sample. A loop that reads and writes eight fields of 2048 x 2048 in
place, by rows, does so at 31 to 39 GB/s on one thread and 43 to 48 GB/s on two to
eight (2.7 ns a sample); the triad a = b + c/2 reaches 18.6 GB/s on one thread and 40
GB/s on eight. TGV on four threads moves 128 bytes a sample in 4.8 to 5.9 ns, 22 to 27
GB/s.

**Efficiency cores** are about 0.6 times as quick as a performance core here; a thread
there takes fewer strips, and adds little.

## The instruction sets

The reference built for each level of x86-64 (`-C target-cpu`), on one thread, the
means of two runs, and the command line on chart-2 (q30, 4:2:0) on 1 and 4 threads:

| target | TGV grey 512 | TGV grey 2048 | TGV colour 1024 | TV grey 512 | TV colour 1024 | command line, 1 and 4 threads |
|---|---|---|---|---|---|---|
| x86-64 (SSE2) | 11.4 | 14.0 | 12.0 | 5.6 | 6.1 | 9.58 s, 4.30 s |
| x86-64-v2 (SSE4.2) | 10.4 | 13.5 | 11.9 | 5.2 | 6.1 | 9.43 s, 4.24 s |
| x86-64-v3 (AVX2, FMA; shipped) | 7.9 | 11.1 | 10.5 | 3.8 | 5.1 | 8.30 s, 3.91 s |
| native (raptorlake) | 8.0 | 11.2 | 10.7 | 3.9 | 5.1 | 8.43 s, 3.87 s |

AVX2 and FMA are 12 to 32% quicker than SSE2 in the solvers, and 9 to 13% in the whole
command line; what this processor has beyond them (AVX-VNNI and the like) adds nothing
to doubles. The builds with FMA differ from those without in the last digits, as
docs/math.md allows; each build's files are the same on any count of threads. The
build for AVX-512 (ci/avx512.sh) is measured against the one shipped on CI's runners,
which have AVX-512, by hand (ci/avx512.sh measure, .github/workflows/throughput.yml).

## The C++ library

Its solver now takes the canvas on threads too, in parts of whole bands of rows (eight
times the least multiple of the components' rows of a cell, so that every block lies
in one part), each part's rows of every step at once, the threads meeting where a
difference reads a row of another part; its results are the same to the last bit on any
count of threads. Its command line, TGV to its tolerance, in seconds, on 1, 2, 4, 8 and
16 threads, one run each (its PNG files are the same on every count, and the same as
the reference's):

| file | 1 | 2 | 4 | 8 | 16 | the reference on 4 |
|---|---|---|---|---|---|---|
| chart-2, q30, 4:2:0 | 24.8 | 17.9 | 14.1 | 14.3 | 14.0 | 4.1 |
| Kodak kodim05, q30, 4:2:0 | 79.0 | 57.2 | 44.8 | 45.8 | 43.8 | 12.0 |
| Kodak kodim05, grey, q30 | 56.9 | 41.5 | 32.4 | 32.9 | 32.5 | 6.1 |

Its iteration was then some twenty passes over whole fields, which moved several times
the reference's 128 bytes a sample, so that the memory's bandwidth bounded it from two
threads on; and its records were the caller's alone, a quarter of its time on four
threads. Then:

- **A step takes a row at a time**: the differences that it reads, of the row, into a
  room of the part's own, and then its loop over the row; where a channel's balls are
  its own (apart, or one channel), its dual is projected, kept only for a record, and
  relaxed in the step's loop, and x and w are relaxed where they are extrapolated. An
  iteration of TGV on a grey picture reads and writes less than half the fields it did.
- **The records on the threads**: each part takes the terms of its rows (of G, the norms
  of the pixels, div_2 r~ and its largest norm, xi and G* there), and the caller adds
  the rows' sums in their order, as one thread added them.
- **The projections multiply** each entry by radius / |p|, as the reference's do, where
  they divided each by max(1, |p| / radius): on a grey canvas, five divisions a sample
  fewer. The binary64 pictures differ in their last digits, and the PNG files, the
  iterations and the stops of the runs below not at all.
- **Coupled channels a row at a time**: every channel's extrapolations first, into
  fields of each channel's own, and then for each row every channel's steps, their
  squares added channel after channel, the row's scales, and each channel's duals
  projected, kept only for a record, and relaxed, where they had been projected and
  relaxed in passes of their own over every channel's fields: a colour picture took some
  40% longer than its channels apart.
- **The extrapolations in rings**: x- and w- of a row are made where the dual steps read
  them, from x~ and x and from w~ and w, into rings of two rows of each channel's own,
  and x and w are relaxed once their row's are made; a part's first row's x- and last
  row's w-, which the parts beside read, are made first, and the parts meet once. The
  fields of the extrapolations, written and read back, are gone.

Kodak's kodim05 in grey, 500 iterations with a record every 10, in seconds, two runs
each, one thread to a performance core:

| | 1 thread | 4 threads |
|---|---|---|
| passes over whole fields | 8.39, 8.27 | 4.41, 4.78 |
| steps by rows | 4.96, 4.95 | 2.91, 2.95 |
| the records on the threads | 4.75, 4.67 | 2.62, 2.50 |
| their G and G* too | 4.60, 4.61 | 2.27, 2.37 |
| the projections multiplying | 4.26, 4.38 | 2.29, 2.31 |
| the extrapolations in rings | 3.99, 3.97 | 1.79, 1.94 |

Kodak's kodim05 of 4:2:0, 300 iterations on four threads: coupled 5.24 and 5.37 s with
the projections of their own, 3.90 and 3.90 a row at a time (its channels apart 3.75 and
3.72), 3.49 and 3.45 with the extrapolations in rings.

The command line, TGV to its tolerance, in seconds (on four threads two runs each):

| file | 1 thread, before | now | 4 threads, before | now | the reference on 4 |
|---|---|---|---|---|---|
| chart-2, q30, 4:2:0 | 24.9 | 13.4 | 13.87, 13.94 | 6.44, 6.62 | 3.80, 3.79 |
| Kodak kodim05, q30, 4:2:0 | 78.5 | 44.6 | 44.3, 44.1 | 20.92, 20.90 | 11.9, 12.0 |
| Kodak kodim05, grey, q30 | 56.2 | 27.8 | 30.8, 30.7 | 13.40, 13.84 | 6.0, 6.1 |

A picture is now about twice as quick, grey or colour, on one thread and on four. The
C++ library still takes 1.7 to 2.2 times the reference's time on four threads: its
primal steps and its dual steps each pass over a part's rows, where the reference's
sweep takes both of a band while its rows are in cache.

## What would go further

- **Several iterations a pass** (temporal blocking): every iteration reads and writes
  the whole point; taking two iterations of a band while its rows are in cache would
  halve the memory traffic, at the cost of rows exchanged between strips for two
  iterations. Here the sweep uses half the memory's bandwidth on four threads, and
  would gain little; a machine with more cores to a channel of memory would gain more.
- **The settling of coupled channels in the steps' loops**: their scales are known only
  after the last channel's step, so that a loop over all the channels of a row at once
  would be needed.
- The square roots and divisions of a grey canvas are its floor, as long as the
  projections are computed as they are.
- **The C++ library**: a sweep that takes a band's primal steps and its dual steps while
  its rows are in cache, as the reference's does, would move half the memory or less.
