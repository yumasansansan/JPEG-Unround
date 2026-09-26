// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The conformance cases (`conformance/cases/*.txt`, docs/math.md, 9): quantized pictures
//! and the options of their reconstruction, with what the Python implementation made of
//! them, which every implementation has to reproduce.
//!
//! What is rational is compared to the last bit: the intervals, the weights of the data
//! term and the steps of the primal-dual method. The centres are compared with their exact
//! values, the means of their bins in 60-digit decimals, within the bound that docs/math.md,
//! 9.2 asks of an implementation. The rest is compared within the tolerances that the file
//! holds, which bound the rounding of both implementations: the Euclidean distance of the
//! canvas, the primal value of every record, how far a dual value may lie above the other
//! implementation's primal value where the gap is certified, and the gaps; the records are
//! taken at the same iterations, and the method stops at the same one. The subgradient
//! method's case is a regression check, which its file says.
//!
//! The command line, run on each case's JPEG file (written by libjpeg's encoder from the
//! case's coefficients), writes the library's result, to the last bit.

mod support;

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use jpeg_unround::decode::{self, Component, Input, Method, Settings};
use jpeg_unround::frames::{Tensor, Vector};
use jpeg_unround::pdhg::{self, Initial, Weights};
use jpeg_unround::results::{History, Stop};
use jpeg_unround::{cli, tiff};
use jpegio_sys::ColorSpace;
use jpegio_sys::testing::{File, Layout};
use support::exact::{Dyadic, Exact};
use support::rounding::{U, gamma};

/// A component of a case: its sampling factors, blocks, table and levels.
#[derive(Debug, Default)]
struct Part {
    h: usize,
    v: usize,
    rows: usize,
    columns: usize,
    table: Vec<u16>,
    levels: Vec<i16>,
}

/// One record of the reference: its iteration, primal and dual values.
#[derive(Debug, Clone, Copy)]
struct Recorded {
    iteration: u64,
    primal: f64,
    dual: f64,
}

/// What a case's file holds, besides what only its bounds need.
#[derive(Debug, Default)]
struct Case {
    name: String,
    height: usize,
    width: usize,
    colour: String,
    parts: Vec<Part>,
    options: Vec<String>,
    steps: Option<(f64, f64)>,
    weights: BTreeMap<usize, Vec<f64>>,
    lower: BTreeMap<usize, Vec<f64>>,
    upper: BTreeMap<usize, Vec<f64>>,
    exact: BTreeMap<usize, Vec<Exact>>,
    first_coefficients: BTreeMap<usize, Vec<f64>>,
    first_fields: BTreeMap<(String, usize), Vec<f64>>,
    records: Vec<Recorded>,
    stop: (u64, String),
    canvas: Vec<f64>,
    method: String,
    tolerance_canvas: f64,
    tolerance_records: BTreeMap<u64, (f64, f64, f64)>,
}

fn numbers<T: std::str::FromStr>(fields: &[&str]) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    fields.iter().map(|field| field.parse().expect("a number")).collect()
}

fn index(field: &str) -> usize {
    field.parse().expect("an index")
}

impl Case {
    fn parse(text: &str) -> Self {
        let mut case = Self::default();
        for line in text.lines().filter(|line| !line.starts_with('#')) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let Some((&keyword, rest)) = fields.split_first() else {
                continue;
            };
            case.take(keyword, rest);
        }
        case
    }

    fn take(&mut self, keyword: &str, rest: &[&str]) {
        match keyword {
            "name" => rest[0].clone_into(&mut self.name),
            "picture" => {
                (self.height, self.width) = (index(rest[0]), index(rest[1]));
                rest[2].clone_into(&mut self.colour);
            }
            "component" => self.parts.push(Part {
                h: index(rest[1]),
                v: index(rest[2]),
                rows: index(rest[3]),
                columns: index(rest[4]),
                ..Part::default()
            }),
            "table" => self.parts[index(rest[0])].table = numbers(&rest[1..]),
            "levels" => self.parts[index(rest[0])].levels.extend(numbers::<i16>(&rest[2..])),
            "options" => self.options = rest.iter().map(|&option| option.to_owned()).collect(),
            "steps" => self.steps = Some((rest[0].parse().expect("tau"), rest[1].parse().expect("sigma"))),
            "weights" => {
                self.weights.insert(index(rest[0]), numbers(&rest[1..]));
            }
            "lower" => self
                .lower
                .entry(index(rest[0]))
                .or_default()
                .extend(numbers::<f64>(&rest[2..])),
            "upper" => self
                .upper
                .entry(index(rest[0]))
                .or_default()
                .extend(numbers::<f64>(&rest[2..])),
            "exact-centres" => self
                .exact
                .entry(index(rest[0]))
                .or_default()
                .extend(rest[2..].iter().map(|&pair| Exact::parse(pair))),
            "first-coefficients" => self
                .first_coefficients
                .entry(index(rest[0]))
                .or_default()
                .extend(numbers::<f64>(&rest[2..])),
            "first-p" | "first-w" | "first-r" => self
                .first_fields
                .entry((keyword.to_owned(), index(rest[0])))
                .or_default()
                .extend(numbers::<f64>(&rest[3..])),
            "record" => self.records.push(Recorded {
                iteration: rest[0].parse().expect("an iteration"),
                primal: rest[1].parse().expect("a primal value"),
                dual: rest[2].parse().expect("a dual value"),
            }),
            "stop" => self.stop = (rest[0].parse().expect("iterations"), rest[1].to_owned()),
            "canvas" => self.canvas.extend(numbers::<f64>(&rest[2..])),
            "constants" => {
                rest.iter()
                    .find_map(|field| field.strip_prefix("method="))
                    .expect("a method")
                    .clone_into(&mut self.method);
            }
            "tolerance" if rest[0] == "canvas" => self.tolerance_canvas = rest[1].parse().expect("a tolerance"),
            "tolerance" => {
                let [primal, bracket, gap] =
                    [rest[2], rest[3], rest[4]].map(|value| value.parse().expect("a tolerance"));
                self.tolerance_records
                    .insert(rest[1].parse().expect("an iteration"), (primal, bracket, gap));
            }
            _ => {}
        }
    }

    fn colour_space(&self) -> ColorSpace {
        match self.colour.as_str() {
            "grayscale" => ColorSpace::Grayscale,
            "ycbcr" => ColorSpace::YCbCr,
            "rgb" => ColorSpace::Rgb,
            other => panic!("a colour space, not {other}"),
        }
    }

    fn input(&self) -> Input {
        Input {
            height: self.height,
            width: self.width,
            color_space: self.colour_space(),
            components: self
                .parts
                .iter()
                .map(|part| Component {
                    levels: part.levels.clone(),
                    rows: part.rows,
                    columns: part.columns,
                    quant_table: part.table.clone().try_into().expect("64 steps"),
                    h_samp_factor: part.h,
                    v_samp_factor: part.v,
                })
                .collect(),
            icc_profile: None,
            exif_orientation: 0,
        }
    }

    fn settings(&self) -> Settings {
        cli::settings_of(self.options.iter().map(OsString::from).collect())
            .unwrap_or_else(|errors| panic!("{}: the options are refused: {errors:?}", self.name))
    }

    fn field(&self, name: &str, entry: usize) -> Vec<f64> {
        self.first_fields
            .get(&(name.to_owned(), entry))
            .cloned()
            .unwrap_or_else(|| panic!("{}: {name} {entry}", self.name))
    }

    fn initial(&self) -> Option<Initial> {
        if self.first_coefficients.is_empty() {
            return None;
        }
        let tgv = self.method == "tgv";
        Some(Initial {
            coefficients: Some(self.first_coefficients.values().cloned().collect()),
            p: Some(Vector {
                x: self.field("first-p", 0),
                y: self.field("first-p", 1),
            }),
            w: tgv.then(|| Vector {
                x: self.field("first-w", 0),
                y: self.field("first-w", 1),
            }),
            r: tgv.then(|| Tensor {
                xx: self.field("first-r", 0),
                yy: self.field("first-r", 1),
                xy: self.field("first-r", 2),
            }),
        })
    }
}

fn cases() -> Vec<Case> {
    let folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/cases");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&folder)
        .expect("the cases' folder")
        .map(|entry| entry.expect("an entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "txt"))
        .collect();
    paths.sort();
    assert!(paths.len() >= 17, "the cases of conformance/cases");
    paths
        .iter()
        .map(|path| Case::parse(&std::fs::read_to_string(path).expect("a case")))
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The Euclidean norm of the difference, and a factor that covers the rounding of computing it.
fn distance(one: &[f64], other: &[f64]) -> f64 {
    assert_eq!(one.len(), other.len());
    let squares: f64 = one.iter().zip(other).map(|(a, b)| (a - b) * (a - b)).sum();
    squares.sqrt() * (1.0 + gamma(one.len() + 3))
}

/// The intervals and the weights to the last bit, and the centres within the bound of 9.2.
fn check_model(case: &Case, input: &Input, settings: &Settings) {
    let frame = decode::frame_of(input, settings).expect("a frame");
    let midpoint = case.options.windows(2).any(|pair| pair == ["--centres", "midpoint"]);
    for (index, channel) in frame.channels().iter().enumerate() {
        let problem = channel.problem();
        assert_eq!(bits(problem.lower()), bits(&case.lower[&index]), "{}: lower", case.name);
        assert_eq!(bits(problem.upper()), bits(&case.upper[&index]), "{}: upper", case.name);
        assert_eq!(
            bits(problem.weights()),
            bits(&case.weights[&index]),
            "{}: weights",
            case.name
        );
        let part = &case.parts[index];
        for (k, (&centre, exact)) in problem.centres().iter().zip(&case.exact[&index]).enumerate() {
            let level = i128::from(part.levels[k]);
            let step = i128::from(part.table[k % 64]);
            let approximate = !midpoint && k % 64 != 0 && level != 0;
            let bound = if approximate {
                Dyadic::from_i128(2 * level.abs() + 20)
                    .mul(&Dyadic::from_f64(U))
                    .mul(&Dyadic::from_i128(step))
            } else {
                Dyadic::zero()
            };
            assert!(
                exact.within(centre, &bound),
                "{}: centre {k} of {index}: {centre}",
                case.name
            );
        }
    }
    if let Some((tau, sigma)) = case.steps {
        let weights = match settings.method {
            Method::Tgv => Weights::Tgv(&settings.tgv),
            _ => Weights::Tv(&settings.tv),
        };
        let plan = pdhg::plan(&settings.pdhg, weights).expect("a plan");
        let steps = pdhg::steps(plan.norm_squared, plan.step_ratio, plan.step_product).expect("the steps");
        assert_eq!(
            (steps.0.to_bits(), steps.1.to_bits()),
            (tau.to_bits(), sigma.to_bits()),
            "{}: steps",
            case.name
        );
    }
}

/// What an implementation made of a case: its canvas, records and stop.
struct Made {
    canvas: Vec<f64>,
    history: Option<History>,
    iterations: u64,
    stop: Option<Stop>,
}

fn solve(case: &Case, input: &Input, settings: &Settings) -> Made {
    if let Some(initial) = case.initial() {
        let frame = decode::frame_of(input, settings).expect("a frame");
        let result = match settings.method {
            Method::Tgv => pdhg::solve_tgv(&frame, &settings.tgv, &settings.pdhg, Some(&initial), None),
            _ => pdhg::solve_tv(&frame, &settings.tv, &settings.pdhg, Some(&initial), None),
        }
        .expect("a result");
        return Made {
            canvas: result.primal.canvas,
            history: Some(result.history),
            iterations: result.iterations,
            stop: Some(result.stop),
        };
    }
    let decoded = decode::solve(input, settings, None).expect("a reconstruction");
    let (history, iterations, stop) = match decoded.result {
        Some(result) => (Some(result.history), result.iterations, Some(result.stop)),
        None => (None, 0, None),
    };
    Made {
        canvas: decoded.canvas,
        history,
        iterations,
        stop,
    }
}

fn check_records(case: &Case, history: &History) {
    let iterations: Vec<u64> = case.records.iter().map(|record| record.iteration).collect();
    assert_eq!(history.iterations, iterations, "{}: the records' iterations", case.name);
    for (k, record) in case.records.iter().enumerate() {
        let (primal, dual) = (history.primal[k], history.dual[k]);
        let (tolerance, bracket, gap) = case.tolerance_records[&record.iteration];
        let n = record.iteration;
        assert!(
            (primal - record.primal).abs() <= tolerance,
            "{}: the primal value of record {n}, {primal}, and the reference's {}, beyond {tolerance}",
            case.name,
            record.primal
        );
        if bracket >= 0.0 {
            // Each dual value is at most the least value, and so at most the other's primal value.
            assert!(
                dual - record.primal <= bracket,
                "{}: record {n}'s dual value above the primal",
                case.name
            );
            assert!(
                record.dual - primal <= bracket,
                "{}: record {n}'s primal value below the dual",
                case.name
            );
        }
        if gap >= 0.0 {
            let (own, theirs) = (primal - dual, record.primal - record.dual);
            assert!(
                (own - theirs).abs() <= gap,
                "{}: record {n}'s gap {own} and {theirs}",
                case.name
            );
        }
    }
}

fn check_case(case: &Case) -> Made {
    let input = case.input();
    let settings = case.settings();
    if case.method != "subgradient" {
        check_model(case, &input, &settings);
    }
    let made = solve(case, &input, &settings);
    let far = distance(&made.canvas, &case.canvas);
    assert!(
        far <= case.tolerance_canvas,
        "{}: the canvas lies {far:e} from the reference's, beyond {:e}",
        case.name,
        case.tolerance_canvas
    );
    if let Some(history) = &made.history {
        check_records(case, history);
        assert_eq!(made.iterations, case.stop.0, "{}: the iterations", case.name);
        let converged = made.stop == Some(Stop::Converged);
        assert_eq!(
            converged,
            case.stop.1 == "converged",
            "{}: why the method stopped",
            case.name
        );
    }
    made
}

#[test]
fn every_case_is_reproduced_within_its_tolerances() {
    for case in cases() {
        check_case(&case);
    }
}

#[test]
#[ignore = "prints how far this implementation lies from every case, against the tolerances"]
fn distances_from_the_cases() {
    for case in cases() {
        let made = check_case(&case);
        let far = distance(&made.canvas, &case.canvas);
        let mut worst = 0.0f64;
        if let Some(history) = &made.history {
            for (record, &primal) in case.records.iter().zip(&history.primal) {
                let (tolerance, _, _) = case.tolerance_records[&record.iteration];
                worst = worst.max((primal - record.primal).abs() / tolerance);
            }
        }
        println!(
            "{:22} canvas {far:9.2e} of the {:9.2e} allowed ({:.1e}); primal values at {worst:.1e} of theirs",
            case.name,
            case.tolerance_canvas,
            far / case.tolerance_canvas
        );
    }
}

fn layout(case: &Case) -> Layout {
    match case.colour_space() {
        ColorSpace::Grayscale => Layout::Grayscale,
        ColorSpace::Rgb => Layout::Rgb,
        ColorSpace::YCbCr => {
            let factor = |value: usize| i32::try_from(value).expect("a factor");
            Layout::YCbCr([0, 1, 2].map(|index| (factor(case.parts[index].h), factor(case.parts[index].v))))
        }
    }
}

fn directory() -> PathBuf {
    let path = std::env::temp_dir().join(format!("unround-conformance-{}", std::process::id()));
    if path.exists() {
        std::fs::remove_dir_all(&path).expect("the old directory is removed");
    }
    std::fs::create_dir_all(&path).expect("a directory");
    path
}

#[test]
fn the_command_line_writes_the_library_s_result() {
    let place = directory();
    for case in cases().iter().filter(|case| case.initial().is_none()) {
        let file = File {
            width: u32::try_from(case.width).expect("fits"),
            height: u32::try_from(case.height).expect("fits"),
            layout: layout(case),
            quant_tables: case
                .parts
                .iter()
                .map(|part| part.table.clone().try_into().expect("64 steps"))
                .collect(),
            progressive: false,
            arithmetic: false,
            icc_profile: Vec::new(),
            exif: Vec::new(),
        };
        let levels: Vec<Vec<i16>> = case.parts.iter().map(|part| part.levels.clone()).collect();
        let data = file.write(&levels).expect("a JPEG file");
        let input = place.join(format!("{}.jpg", case.name));
        let output = place.join(format!("{}.tif", case.name));
        std::fs::write(&input, &data).expect("the file is written");
        let status = Command::new(env!("CARGO_BIN_EXE_unround"))
            .args(&case.options)
            .args(["--quiet", "--overwrite"])
            .arg(&input)
            .arg(&output)
            .status()
            .expect("the program runs");
        assert!(status.success(), "{}: {status}", case.name);
        let decoded = decode::decode(&data, &case.settings(), None).expect("the file decodes");
        let made = solve(case, &case.input(), &case.settings());
        assert_eq!(
            bits(&decoded.canvas),
            bits(&made.canvas),
            "{}: the file's coefficients",
            case.name
        );
        let expected = tiff::float64(
            &decoded.picture,
            decoded.height,
            decoded.width,
            decoded.channels,
            false,
            None,
        )
        .expect("a file");
        assert_eq!(
            std::fs::read(&output).expect("the result"),
            expected.bytes,
            "{}",
            case.name
        );
    }
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}
