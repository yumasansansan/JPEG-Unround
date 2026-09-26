// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Measures the speed of the solvers and of their parts, on one thread: run by hand,
//! `cargo test --release --test throughput -- --ignored --nocapture`, when a kernel
//! changes. Each figure is the least of several runs, in nanoseconds per sample of the
//! canvas (`C x H x W`): per iteration for the solvers, not counting their records,
//! and per call for the parts.

mod support;

use std::hint::black_box;
use std::time::Instant;

use jpeg_unround::dct;
use jpeg_unround::frames::{self, Frame, Vector};
use jpeg_unround::model::{DataTerm, Tgv, Tv};
use jpeg_unround::pdhg::{self, Options};
use support::synthetic;

const RUNS: usize = 5;

/// The least of `RUNS` measurements, each of which `work` returns in seconds.
fn least(mut work: impl FnMut() -> f64) -> f64 {
    (0..RUNS).map(|_| work()).fold(f64::INFINITY, f64::min)
}

/// The seconds that `work` takes.
fn timed(work: impl FnOnce()) -> f64 {
    let start = Instant::now();
    work();
    start.elapsed().as_secs_f64()
}

fn per_sample(seconds: f64, samples: usize) -> f64 {
    seconds * 1e9 / f64::from(u32::try_from(samples).expect("fits"))
}

fn fixed(iterations: u64) -> Options {
    Options {
        iterations: Some(iterations),
        tolerance: Some(0.0),
        record_every: 0,
        ..Options::default()
    }
}

/// The solver's own seconds per iteration, not counting its records.
fn solver(frame: &Frame, tgv: bool, iterations: u64) -> f64 {
    least(|| {
        let result = if tgv {
            pdhg::solve_tgv(frame, &Tgv::default(), &fixed(iterations), None, None)
        } else {
            pdhg::solve_tv(frame, &Tv::default(), &fixed(iterations), None, None)
        }
        .expect("a result");
        let seconds = result.history.seconds;
        seconds[seconds.len() - 1] / f64::from(u32::try_from(iterations).expect("fits"))
    })
}

/// Streams over memory: the least seconds of reading two arrays and writing a third,
/// per sample; what the bandwidth of one thread allows.
fn stream(samples: usize) -> f64 {
    let (a, b) = (vec![1.0; samples], vec![2.0; samples]);
    let mut c = vec![0.0; samples];
    least(|| {
        timed(|| {
            for ((out, &x), &y) in c.iter_mut().zip(black_box(&a)).zip(black_box(&b)) {
                *out = x + 0.5 * y;
            }
            black_box(&mut c);
        })
    })
}

/// The seconds of a solver's setup and first record: no iterations.
fn setup(frame: &Frame, tgv: bool) -> f64 {
    let options = Options {
        iterations: Some(0),
        ..Options::default()
    };
    least(|| {
        timed(|| {
            black_box(
                if tgv {
                    pdhg::solve_tgv(frame, &Tgv::default(), &options, None, None)
                } else {
                    pdhg::solve_tv(frame, &Tv::default(), &options, None, None)
                }
                .expect("a result"),
            );
        })
    })
}

/// The seconds of a record of TV's values, in the planes and in the natural layout.
fn records(frame: &Frame) -> (f64, f64) {
    use jpeg_unround::planar::Layout;
    use jpeg_unround::records::Values;
    use jpeg_unround::sweep::{TvOutputs, TvPoint, TvSweep};
    let layout = Layout::new(frame).expect("a layout");
    let gammas = vec![1.0; frame.channels().len()];
    let steps = frames::steps(frame, 0.1);
    let start = frames::start(frame, None).expect("a start");
    let size = frame.samples();
    let mut point = TvPoint {
        x: layout.to_planes(&start.canvas),
        px: vec![0.0; size],
        py: vec![0.0; size],
    };
    let mut sweep = TvSweep::new(frame, &layout, &steps, &gammas, 0.1, 0.1, 1.0, true, 1.9);
    let mut outputs = TvOutputs::new(&layout);
    sweep.iterate(&mut point, Some(&mut outputs));
    let mut values = Values::new(frame, &layout, &gammas);
    let planar = least(|| {
        timed(|| {
            black_box(values.tv(&Tv::default(), &outputs, 255.0));
        })
    });
    let canvas = layout.to_natural(&outputs.canvas);
    let field = Vector {
        x: layout.to_natural(&outputs.px),
        y: layout.to_natural(&outputs.py),
    };
    let coefficients: Vec<Vec<f64>> = layout
        .parts
        .iter()
        .zip(&outputs.coefficients)
        .map(|(part, planes)| {
            let mut own = vec![0.0; part.rows * part.columns * 64];
            layout.write_values_natural(part, planes, &mut own);
            own
        })
        .collect();
    let natural = least(|| {
        timed(|| {
            black_box(frames::tv_values(frame, &Tv::default(), &coefficients, &canvas, &field, 255.0).expect("values"));
        })
    });
    (planar, natural)
}

/// The parts of a solver's setup, each the least of several runs in nanoseconds per
/// sample: the layout, the start, the gradient, the sweeps and the first records.
fn setup_parts(name: &str, frame: &Frame) {
    use jpeg_unround::planar::Layout;
    use jpeg_unround::records::Values;
    use jpeg_unround::sweep::{self, TgvOutputs, TgvSweep, TvOutputs, TvSweep};
    let samples = frame.samples();
    let gammas = vec![1.0; frame.channels().len()];
    let layout = least(|| timed(|| drop(black_box(Layout::new(frame).expect("a layout")))));
    let own = Layout::new(frame).expect("a layout");
    let start = least(|| timed(|| drop(black_box(sweep::start(frame, &own, None).expect("a start")))));
    let (canvas, coefficients) = sweep::start(frame, &own, None).expect("a start");
    let gradient = least(|| timed(|| drop(black_box(sweep::gradient(&own, &canvas)))));
    let steps = frames::steps(frame, 0.1);
    // TV's and TGV's.
    let sweeps = (
        least(|| {
            timed(|| {
                drop(black_box(TvSweep::new(
                    frame, &own, &steps, &gammas, 0.1, 0.1, 1.0, true, 1.9,
                )));
            })
        }),
        least(|| {
            timed(|| {
                drop(black_box(TgvSweep::new(
                    frame,
                    &own,
                    &steps,
                    &gammas,
                    0.1,
                    0.1,
                    (1.0, 2.0),
                    true,
                    1.9,
                )));
            })
        }),
    );
    let values = least(|| timed(|| drop(black_box(Values::new(frame, &own, &gammas)))));
    let mut record = Values::new(frame, &own, &gammas);
    let (across, down) = sweep::gradient(&own, &canvas);
    let outputs = TgvOutputs {
        first: TvOutputs {
            canvas,
            coefficients,
            px: vec![0.0; samples],
            py: vec![0.0; samples],
        },
        wx: across,
        wy: down,
        rxx: vec![0.0; samples],
        ryy: vec![0.0; samples],
        rxy: vec![0.0; samples],
    };
    let records = (
        least(|| {
            timed(|| {
                black_box(record.tv(&Tv::default(), &outputs.first, 255.0));
            })
        }),
        least(|| {
            timed(|| {
                black_box(record.tgv(&Tgv::default(), &outputs, 255.0));
            })
        }),
    );
    eprintln!(
        "{name}: setup: layout {:.3}, start {:.2}, gradient {:.2}, TV sweep {:.2}, TGV sweep {:.2}, values {:.2}, \
         records TV {:.2}, TGV {:.2}",
        per_sample(layout, samples),
        per_sample(start, samples),
        per_sample(gradient, samples),
        per_sample(sweeps.0, samples),
        per_sample(sweeps.1, samples),
        per_sample(values, samples),
        per_sample(records.0, samples),
        per_sample(records.1, samples),
    );
}

/// The conversions between the layouts, into memory written before and into fresh
/// memory, and writing memory for the first time and again, each the least of several
/// runs in nanoseconds per sample.
fn conversion_parts(name: &str, frame: &Frame) {
    use jpeg_unround::planar::Layout;
    use jpeg_unround::sweep;
    let samples = frame.samples();
    let own = Layout::new(frame).expect("a layout");
    let (canvas, coefficients) = sweep::start(frame, &own, None).expect("a start");
    let mut natural = vec![0.0; samples];
    let to_natural = least(|| timed(|| own.write_natural(black_box(&canvas), &mut natural)));
    let to_fresh = least(|| timed(|| drop(black_box(own.to_natural(black_box(&canvas))))));
    let to_planes = least(|| timed(|| drop(black_box(own.to_planes(black_box(&natural))))));
    let first = frame.channels()[0].problem();
    let part = &own.parts[0];
    let levels = least(|| timed(|| drop(black_box(own.values_to_planes(part, black_box(first.levels()))))));
    let mut values_natural = vec![0.0; first.samples()];
    let coefficients_natural =
        least(|| timed(|| own.write_values_natural(part, black_box(&coefficients[0]), &mut values_natural)));
    let fresh = least(|| {
        timed(|| {
            let mut memory = vec![0.0; samples];
            memory.fill(1.0);
            black_box(&memory);
        })
    });
    let mut written = vec![0.0; samples];
    let again = least(|| {
        timed(|| {
            written.fill(1.0);
            black_box(&written);
        })
    });
    let first_samples = first.samples();
    eprintln!(
        "{name}: setup: to natural {:.2} (fresh {:.2}), to planes {:.2}, levels to planes {:.2}, coefficients to \
         natural {:.2} (first component); writing fresh memory {:.2}, again {:.2}",
        per_sample(to_natural, samples),
        per_sample(to_fresh, samples),
        per_sample(to_planes, samples),
        per_sample(levels, first_samples),
        per_sample(coefficients_natural, first_samples),
        per_sample(fresh, samples),
        per_sample(again, samples),
    );
}

fn parts(name: &str, frame: &Frame) {
    let samples = frame.samples();
    let (height, width) = (frame.height(), frame.width());
    let start = frames::start(frame, None).expect("a start");
    let canvas = start.canvas;
    let plane = frame.plane();
    // The DCT of the first channel's canvas, and back.
    let first = &canvas[..plane];
    let forward = least(|| {
        timed(|| {
            black_box(dct::forward(black_box(first), height, width));
        })
    });
    let coefficients = dct::forward(first, height, width);
    let inverse = least(|| {
        timed(|| {
            black_box(dct::inverse(black_box(&coefficients), height / 8, width / 8));
        })
    });
    // The proximal map of G, every channel, with a step of 0.1.
    let steps = frames::steps(frame, 0.1);
    let mut kept = start.coefficients.clone();
    let mut out = vec![0.0; samples];
    let prox = least(|| {
        timed(|| {
            frames::prox(frame, &steps, black_box(&canvas), &mut kept, &mut out);
        })
    });
    // The projection of a field onto the balls, most of whose vectors are outside them.
    let field = Vector {
        x: canvas.iter().map(|value| value * 0.01).collect(),
        y: canvas.iter().map(|value| value * -0.02).collect(),
    };
    let mut projected = field.clone();
    let coupled = least(|| {
        projected.clone_from(&field);
        timed(|| frames::project_vectors(frame, &mut projected, 1.0, true))
    });
    let apart = least(|| {
        projected.clone_from(&field);
        timed(|| frames::project_vectors(frame, &mut projected, 1.0, false))
    });
    // TV's primal and dual values: a record.
    let values = least(|| {
        timed(|| {
            black_box(
                frames::tv_values(frame, &Tv::default(), &start.coefficients, &canvas, &field, 255.0).expect("values"),
            );
        })
    });
    eprintln!("{name}: stream {:.2} (one channel)", per_sample(stream(plane), plane));
    eprintln!(
        "{name}: forward DCT {:.2}, inverse {:.2} (one channel); prox {:.2}; projection coupled {:.2}, apart {:.2}; \
         TV values {:.2} ns per sample",
        per_sample(forward, plane),
        per_sample(inverse, plane),
        per_sample(prox, samples),
        per_sample(coupled, samples),
        per_sample(apart, samples),
        per_sample(values, samples),
    );
}

#[test]
#[ignore = "a measurement, run by hand"]
fn throughput() {
    let frames = [
        (
            "grey 512 x 512",
            Frame::one(synthetic::problem(512, 512, 1, 2.0, &DataTerm::default()).0),
            40,
        ),
        (
            "grey 2048 x 2048",
            Frame::one(synthetic::problem(2048, 2048, 2, 2.0, &DataTerm::default()).0),
            10,
        ),
        (
            "colour 1024 x 1024, 4:2:0",
            synthetic::colour_frame(1024, 1024, 3, (2, 2), &DataTerm::default()).0,
            10,
        ),
    ];
    for (name, frame, iterations) in &frames {
        let samples = frame.samples();
        let tv = solver(frame, false, *iterations);
        let tgv = solver(frame, true, *iterations);
        let (planar, natural) = records(frame);
        eprintln!(
            "{name}: TV {:.2}, TGV {:.2} ns per sample and iteration; a record of TV {:.2} (natural layout {:.2}); \
             setup TV {:.2}, TGV {:.2}",
            per_sample(tv, samples),
            per_sample(tgv, samples),
            per_sample(planar, samples),
            per_sample(natural, samples),
            per_sample(setup(frame, false), samples),
            per_sample(setup(frame, true), samples),
        );
        setup_parts(name, frame);
        conversion_parts(name, frame);
        parts(name, frame);
    }
}
