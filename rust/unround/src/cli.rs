// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The command line, `unround [OPTIONS] INPUT [OUTPUT]`, as docs/cli.md states it.
//!
//! The options have the names of the library's settings, and every value of the
//! model and of the solvers can be given. The command line is checked as a whole:
//! every option that is refused is named at once.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::decode::{self, Method, Settings};
use crate::error::Error;
use crate::model::{Centres, DataTerm};
use crate::output;
use crate::pdhg::Record;
use crate::results::{History, Stop};
use crate::tiff;

/// The version of JPEG-Unround and the implementation that runs.
pub const VERSION: &str = concat!("unround ", env!("CARGO_PKG_VERSION"), " (Rust)");

const HELP: &str = "\
Usage: unround [OPTIONS] INPUT [OUTPUT]

Reconstructs the JPEG file INPUT within its quantization intervals and writes the
result to OUTPUT (by default next to INPUT). INPUT and OUTPUT may be - for the
standard input and output. docs/cli.md states every option and its default.

Output:      --format tiff|pnm  --bits 8|16  --ycbcr  --overwrite
Method:      --method mmse|tv|tgv|subgradient  --alpha A  --alpha1 A  --alpha0 A
             --channel-weights G,G,G  --channels coupled|apart
Data term:   --mu X|rule  --mu-scale S  --mu-power R  --weight-power P  --dc-weight W
             --centres mmse|midpoint  --slack S  --slack-cost B   (one value, or one per component)
Solver:      --iterations N  --tolerance X  --relative-tolerance X  --partial-tolerance X
             --partial-radius X  --step-ratio X  --relaxation X  --step-product X
             --norm-squared X  --no-weight-scaling  --record-every N  --free-radius R
Subgradient: --subgradient-iterations N  --subgradient-step X  --subgradient-decay X
             --no-momentum  --subgradient-record-every N
Reading:     --max-pixels N  --max-scans N  --warnings-are-errors
Reports:     -v, --verbose  -q, --quiet  --report PATH  --version  -h, --help
";

/// The options that take a value.
const VALUED: &[&str] = &[
    "format",
    "bits",
    "method",
    "alpha",
    "alpha1",
    "alpha0",
    "channel-weights",
    "channels",
    "mu",
    "mu-scale",
    "mu-power",
    "weight-power",
    "dc-weight",
    "centres",
    "slack",
    "slack-cost",
    "iterations",
    "tolerance",
    "relative-tolerance",
    "partial-tolerance",
    "partial-radius",
    "step-ratio",
    "relaxation",
    "step-product",
    "norm-squared",
    "record-every",
    "free-radius",
    "subgradient-iterations",
    "subgradient-step",
    "subgradient-decay",
    "subgradient-record-every",
    "max-pixels",
    "max-scans",
    "report",
];

/// The options that take none.
const FLAGS: &[&str] = &[
    "ycbcr",
    "overwrite",
    "no-weight-scaling",
    "no-momentum",
    "warnings-are-errors",
    "verbose",
    "quiet",
    "version",
    "help",
];

/// The format of the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// TIFF of binary64 samples, bit for bit.
    Tiff,
    /// PNM of 8 or 16 bits.
    Pnm,
}

/// The bits of a sample of PNM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bits {
    /// 8 bits, 0 to 255.
    Eight,
    /// 16 bits, 0 to 65535, where 255 is 65535.
    Sixteen,
}

/// What goes to standard error besides errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// Nothing.
    Quiet,
    /// The iterations the solver took, and why it stopped.
    Normal,
    /// Every record as well.
    Verbose,
}

/// What the command line asks for.
#[derive(Debug, Clone, PartialEq)]
pub struct Command {
    /// The file to read, or `-`.
    pub input: PathBuf,
    /// Where the result goes, or `-`.
    pub output: PathBuf,
    /// The format of the result.
    pub format: Format,
    /// The bits of PNM samples.
    pub bits: Bits,
    /// Whether TIFF holds the Y, Cb and Cr of the solution.
    pub ycbcr: bool,
    /// Whether a file that exists may be overwritten.
    pub overwrite: bool,
    /// The method, the model and the solvers' options.
    pub settings: Settings,
    /// Where the report goes, where one is asked for.
    pub report: Option<PathBuf>,
    /// What goes to standard error besides errors.
    pub verbosity: Verbosity,
}

/// What the arguments ask for: a command, or only the help or the version.
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    /// Reconstruct a file.
    Run(Box<Command>),
    /// Print the options.
    Help,
    /// Print the version.
    Version,
}

/// The options given, by name, and the positional arguments.
#[derive(Debug, Default)]
struct Given {
    values: BTreeMap<String, String>,
    flags: BTreeSet<String>,
    positional: Vec<OsString>,
}

fn split(arguments: Vec<OsString>, errors: &mut Vec<String>) -> Given {
    let mut given = Given::default();
    let mut rest = arguments.into_iter();
    let mut options_ended = false;
    while let Some(argument) = rest.next() {
        let text = argument.to_str().map(str::to_owned);
        let option = match &text {
            Some(text) if !options_ended && text == "--" => {
                options_ended = true;
                continue;
            }
            Some(text) if !options_ended && text.starts_with("--") => text[2..].to_owned(),
            Some(text) if !options_ended && text == "-v" => "verbose".to_owned(),
            Some(text) if !options_ended && text == "-q" => "quiet".to_owned(),
            Some(text) if !options_ended && text == "-h" => "help".to_owned(),
            Some(text) if !options_ended && text.starts_with('-') && text != "-" => {
                errors.push(format!("unknown option {text}"));
                continue;
            }
            _ => {
                given.positional.push(argument);
                continue;
            }
        };
        let (name, inline) = match option.split_once('=') {
            Some((name, value)) => (name.to_owned(), Some(value.to_owned())),
            None => (option, None),
        };
        if FLAGS.contains(&name.as_str()) {
            if inline.is_some() {
                errors.push(format!("--{name} takes no value"));
            } else if !given.flags.insert(name.clone()) {
                errors.push(format!("--{name} is given twice"));
            }
        } else if VALUED.contains(&name.as_str()) {
            let value = match inline {
                Some(value) => Some(value),
                None => rest.next().and_then(|value| value.into_string().ok()),
            };
            match value {
                None => errors.push(format!("--{name} needs a value")),
                Some(value) => {
                    if given.values.insert(name.clone(), value).is_some() {
                        errors.push(format!("--{name} is given twice"));
                    }
                }
            }
        } else {
            errors.push(format!("unknown option --{name}"));
        }
    }
    given
}

/// Reads the values of the options, and gathers every one that is refused.
struct Reader<'a> {
    given: &'a Given,
    errors: &'a mut Vec<String>,
}

impl Reader<'_> {
    fn text(&self, name: &str) -> Option<&str> {
        self.given.values.get(name).map(String::as_str)
    }

    fn flag(&self, name: &str) -> bool {
        self.given.flags.contains(name)
    }

    fn number(&mut self, name: &str, valid: impl Fn(f64) -> bool, range: &str) -> Option<f64> {
        let text = self.text(name)?;
        match text.parse::<f64>() {
            Ok(value) if valid(value) => Some(value),
            _ => {
                self.errors.push(format!("--{name} is {range}, not {text}"));
                None
            }
        }
    }

    fn count(&mut self, name: &str, least: u64) -> Option<u64> {
        let text = self.text(name)?;
        match text.parse::<u64>() {
            Ok(value) if value >= least => Some(value),
            _ => {
                self.errors
                    .push(format!("--{name} is a whole number of at least {least}, not {text}"));
                None
            }
        }
    }

    fn choice<T: Copy>(&mut self, name: &str, choices: &[(&str, T)]) -> Option<T> {
        let text = self.text(name)?;
        if let Some(&(_, value)) = choices.iter().find(|(word, _)| *word == text) {
            return Some(value);
        }
        let words: Vec<&str> = choices.iter().map(|&(word, _)| word).collect();
        self.errors
            .push(format!("--{name} is one of {}, not {text}", words.join(", ")));
        None
    }

    /// A value for every component, or a comma-separated list of one for each.
    fn list<T>(&mut self, name: &str, parse: impl Fn(&str) -> Option<T>, range: &str) -> Option<Vec<T>> {
        let text = self.text(name)?.to_owned();
        let mut values = Vec::new();
        for item in text.split(',') {
            if let Some(value) = parse(item.trim()) {
                values.push(value);
            } else {
                self.errors.push(format!("--{name} is {range}, not {item}"));
                return None;
            }
        }
        Some(values)
    }
}

fn at_least_zero(value: f64) -> bool {
    (0.0..f64::INFINITY).contains(&value)
}

fn positive(value: f64) -> bool {
    value > 0.0 && value.is_finite()
}

fn number_list(text: &str, valid: impl Fn(f64) -> bool) -> Option<f64> {
    text.parse::<f64>().ok().filter(|&value| valid(value))
}

/// The options of `G` for each component, from the lists of the data term.
fn data_terms(reader: &mut Reader<'_>) -> Vec<DataTerm> {
    let finite = |value: f64| value.is_finite();
    let mu = reader.list(
        "mu",
        |item| {
            if item == "rule" {
                Some(None)
            } else {
                number_list(item, at_least_zero).map(Some)
            }
        },
        "at least 0 and finite, or rule",
    );
    let mu_scale = reader.list(
        "mu-scale",
        |item| number_list(item, at_least_zero),
        "at least 0 and finite",
    );
    let mu_power = reader.list("mu-power", |item| number_list(item, finite), "finite");
    let power = reader.list(
        "weight-power",
        |item| number_list(item, at_least_zero),
        "at least 0 and finite",
    );
    let dc_weight = reader.list(
        "dc-weight",
        |item| number_list(item, at_least_zero),
        "at least 0 and finite",
    );
    let slack = reader.list(
        "slack",
        |item| number_list(item, at_least_zero),
        "at least 0 and finite",
    );
    let slack_cost = reader.list(
        "slack-cost",
        |item| number_list(item, at_least_zero),
        "at least 0 and finite",
    );
    let centres = reader.list(
        "centres",
        |item| match item {
            "mmse" => Some(Centres::Mmse),
            "midpoint" => Some(Centres::Midpoint),
            _ => None,
        },
        "mmse or midpoint",
    );
    let lengths: BTreeSet<usize> = [
        mu.as_ref().map(Vec::len),
        mu_scale.as_ref().map(Vec::len),
        mu_power.as_ref().map(Vec::len),
        power.as_ref().map(Vec::len),
        dc_weight.as_ref().map(Vec::len),
        slack.as_ref().map(Vec::len),
        slack_cost.as_ref().map(Vec::len),
        centres.as_ref().map(Vec::len),
    ]
    .into_iter()
    .flatten()
    .filter(|&length| length > 1)
    .collect();
    if lengths.len() > 1 {
        reader.errors.push(format!(
            "the lists of the data term have one value for each component, and so one length, not {lengths:?}"
        ));
    }
    let count = lengths.into_iter().next().unwrap_or(1);
    (0..count)
        .map(|index| {
            let pick = |values: &Option<Vec<f64>>, default: f64| {
                values
                    .as_ref()
                    .map_or(default, |values| values[if values.len() == 1 { 0 } else { index }])
            };
            let defaults = DataTerm::default();
            DataTerm {
                mu: mu
                    .as_ref()
                    .map_or(defaults.mu, |values| values[if values.len() == 1 { 0 } else { index }]),
                mu_scale: pick(&mu_scale, defaults.mu_scale),
                mu_power: pick(&mu_power, defaults.mu_power),
                slack: pick(&slack, defaults.slack),
                dc_weight: pick(&dc_weight, defaults.dc_weight),
                centres: centres.as_ref().map_or(defaults.centres, |values| {
                    values[if values.len() == 1 { 0 } else { index }]
                }),
                power: pick(&power, defaults.power),
                slack_cost: pick(&slack_cost, defaults.slack_cost),
            }
        })
        .collect()
}

fn settings(reader: &mut Reader<'_>) -> Settings {
    let mut settings = Settings::default();
    if let Some(method) = reader.choice(
        "method",
        &[
            ("mmse", Method::Mmse),
            ("tv", Method::Tv),
            ("tgv", Method::Tgv),
            ("subgradient", Method::Subgradient),
        ],
    ) {
        settings.method = method;
    }
    if let Some(alpha) = reader.number("alpha", positive, "positive and finite") {
        settings.tv.alpha = alpha;
    }
    if let Some(alpha1) = reader.number("alpha1", positive, "positive and finite") {
        settings.tgv.alpha1 = alpha1;
    }
    if let Some(alpha0) = reader.number("alpha0", positive, "positive and finite") {
        settings.tgv.alpha0 = alpha0;
    }
    let gammas = reader.list(
        "channel-weights",
        |item| number_list(item, positive),
        "a list of positive, finite weights",
    );
    settings.tv.channel_weights.clone_from(&gammas);
    settings.tgv.channel_weights = gammas;
    if let Some(coupled) = reader.choice("channels", &[("coupled", true), ("apart", false)]) {
        settings.tv.coupled = coupled;
        settings.tgv.coupled = coupled;
    }
    settings.data = data_terms(reader);
    let solver = &mut settings.pdhg;
    solver.iterations = reader.count("iterations", 0);
    solver.tolerance = reader.number("tolerance", at_least_zero, "at least 0 and finite");
    if let Some(value) = reader.number("relative-tolerance", at_least_zero, "at least 0 and finite") {
        solver.relative_tolerance = value;
    }
    if let Some(value) = reader.number("partial-tolerance", at_least_zero, "at least 0 and finite") {
        solver.partial_tolerance = value;
    }
    solver.partial_radius = reader.number("partial-radius", at_least_zero, "at least 0 and finite");
    solver.step_ratio = reader.number("step-ratio", positive, "positive and finite");
    solver.relaxation = reader.number("relaxation", |value| value > 0.0 && value < 2.0, "in (0, 2)");
    if let Some(value) = reader.number("step-product", |value| value > 0.0 && value < 1.0, "in (0, 1)") {
        solver.step_product = value;
    }
    solver.norm_squared = reader.number("norm-squared", positive, "positive and finite");
    solver.scale_with_weight = !reader.flag("no-weight-scaling");
    if let Some(value) = reader.count("record-every", 0) {
        solver.record_every = value;
    }
    if let Some(value) = reader.number("free-radius", at_least_zero, "at least 0 and finite") {
        solver.free_radius = value;
    }
    let subgradient = &mut settings.subgradient;
    if let Some(value) = reader.count("subgradient-iterations", 0) {
        subgradient.iterations = value;
    }
    if let Some(value) = reader.number("subgradient-step", positive, "positive and finite") {
        subgradient.step = value;
    }
    if let Some(value) = reader.number("subgradient-decay", at_least_zero, "at least 0 and finite") {
        subgradient.decay = value;
    }
    subgradient.momentum = !reader.flag("no-momentum");
    if let Some(value) = reader.count("subgradient-record-every", 0) {
        subgradient.record_every = value;
    }
    if let Some(value) = reader.count("max-pixels", 1) {
        settings.read.max_pixels = value;
    }
    if let Some(value) = reader.count("max-scans", 1) {
        settings.read.max_scans = u32::try_from(value).unwrap_or(u32::MAX);
    }
    settings.read.warnings_are_errors = reader.flag("warnings-are-errors");
    settings
}

fn format_of(path: &Path) -> Option<Format> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "tif" | "tiff" => Some(Format::Tiff),
        "pgm" | "ppm" | "pnm" => Some(Format::Pnm),
        _ => None,
    }
}

/// What the arguments (without the program's name) ask for.
///
/// # Errors
///
/// Every option that is refused, one message each.
pub fn parse(arguments: Vec<OsString>) -> Result<Request, Vec<String>> {
    let mut errors = Vec::new();
    let given = split(arguments, &mut errors);
    if given.flags.contains("help") {
        return Ok(Request::Help);
    }
    if given.flags.contains("version") {
        return Ok(Request::Version);
    }
    let mut reader = Reader {
        given: &given,
        errors: &mut errors,
    };
    let settings = settings(&mut reader);
    let format = reader.choice("format", &[("tiff", Format::Tiff), ("pnm", Format::Pnm)]);
    let bits = reader.choice("bits", &[("8", Bits::Eight), ("16", Bits::Sixteen)]);
    let (ycbcr, overwrite) = (reader.flag("ycbcr"), reader.flag("overwrite"));
    let (verbose, quiet) = (reader.flag("verbose"), reader.flag("quiet"));
    let report = reader.text("report").map(PathBuf::from);
    if verbose && quiet {
        errors.push("--verbose and --quiet do not go together".into());
    }
    let mut positional = given.positional.into_iter().map(PathBuf::from);
    let input = positional.next();
    let named = positional.next();
    if positional.next().is_some() {
        errors.push("one INPUT and at most one OUTPUT".into());
    }
    let Some(input) = input else {
        errors.push("an INPUT file to read".into());
        return Err(errors);
    };
    let format = format
        .or_else(|| named.as_deref().and_then(format_of))
        .unwrap_or(Format::Tiff);
    if ycbcr && format != Format::Tiff {
        errors.push("--ycbcr is for TIFF".into());
    }
    if bits.is_some() && format != Format::Pnm {
        errors.push("--bits is for PNM".into());
    }
    let output = named.unwrap_or_else(|| {
        let extension = match format {
            Format::Tiff => "tif",
            Format::Pnm => "pnm",
        };
        if input == Path::new("-") {
            PathBuf::from("-")
        } else {
            input.with_extension(extension)
        }
    });
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(Request::Run(Box::new(Command {
        input,
        output,
        format,
        bits: bits.unwrap_or(Bits::Eight),
        ycbcr,
        overwrite,
        settings,
        report,
        verbosity: match (verbose, quiet) {
            (true, _) => Verbosity::Verbose,
            (false, true) => Verbosity::Quiet,
            (false, false) => Verbosity::Normal,
        },
    })))
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:?}")
    } else {
        "null".to_owned()
    }
}

fn json_string(text: &str) -> String {
    let mut quoted = String::from("\"");
    for character in text.chars() {
        match character {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            character if u32::from(character) < 0x20 => {
                let _ = write!(quoted, "\\u{:04x}", u32::from(character));
            }
            character => quoted.push(character),
        }
    }
    quoted.push('"');
    quoted
}

fn json_list(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|&value| json_number(value)).collect();
    format!("[{}]", items.join(", "))
}

/// The report of a reconstruction, as a JSON object (docs/cli.md).
#[must_use]
pub fn report(command: &Command, decoded: &decode::Decoded) -> String {
    let method = match command.settings.method {
        Method::Mmse => "mmse",
        Method::Tv => "tv",
        Method::Tgv => "tgv",
        Method::Subgradient => "subgradient",
    };
    let space = match decoded.color_space {
        jpegio_sys::ColorSpace::Grayscale => "grayscale",
        jpegio_sys::ColorSpace::YCbCr => "ycbcr",
        jpegio_sys::ColorSpace::Rgb => "rgb",
    };
    let mut text = String::from("{\n");
    let _ = writeln!(text, "  \"version\": {},", json_string(VERSION));
    let _ = writeln!(text, "  \"input\": {},", json_string(&command.input.to_string_lossy()));
    let _ = writeln!(text, "  \"height\": {},", decoded.height);
    let _ = writeln!(text, "  \"width\": {},", decoded.width);
    let _ = writeln!(text, "  \"color_space\": {},", json_string(space));
    let _ = write!(text, "  \"method\": {}", json_string(method));
    if let Some(result) = &decoded.result {
        let stop = match result.stop {
            Stop::Converged => "converged",
            Stop::Iterations => "iterations",
            Stop::Observer => "observer",
            Stop::Stationary => "stationary",
        };
        let History {
            iterations,
            seconds,
            primal,
            dual,
            scaling,
            partial_gap,
        } = &result.history;
        let counts: Vec<String> = iterations.iter().map(u64::to_string).collect();
        let _ = write!(
            text,
            ",\n  \"iterations\": {},\n  \"stop\": {},\n  \"records\": {{\n    \"iterations\": [{}],\n    \
             \"seconds\": {},\n    \"primal\": {},\n    \"dual\": {},\n    \"scaling\": {},\n    \
             \"partial_gap\": {}\n  }}",
            result.iterations,
            json_string(stop),
            counts.join(", "),
            json_list(seconds),
            json_list(primal),
            json_list(dual),
            json_list(scaling),
            json_list(partial_gap),
        );
    }
    text.push_str("\n}\n");
    text
}

fn read_input(path: &Path) -> Result<Vec<u8>, Error> {
    if path == Path::new("-") {
        let mut data = Vec::new();
        std::io::stdin().read_to_end(&mut data)?;
        Ok(data)
    } else {
        Ok(std::fs::read(path)?)
    }
}

fn write_output(path: &Path, bytes: &[u8], overwrite: bool) -> Result<(), Error> {
    if path == Path::new("-") {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(bytes)?;
        stdout.flush()?;
        return Ok(());
    }
    if overwrite {
        std::fs::write(path, bytes)?;
    } else {
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(bytes)?;
    }
    Ok(())
}

/// Runs a command: reads, reconstructs, and writes.
///
/// # Errors
///
/// The status of docs/cli.md with its message: 1 where the file is not read or not
/// taken, 3 where the result is not written.
pub fn run(command: &Command) -> Result<(), (u8, String)> {
    let data = read_input(&command.input).map_err(|error| (1, format!("{}: {error}", command.input.display())))?;
    let mut verbose = |record: &Record<'_>| {
        eprintln!(
            "{:>6}  gap/sample {:.6e}  primal {:.10e}  dual {:.10e}",
            record.iteration, record.gap, record.primal, record.dual
        );
        ControlFlow::Continue(())
    };
    let observer: Option<&mut crate::pdhg::Observer<'_>> = if command.verbosity == Verbosity::Verbose {
        Some(&mut verbose)
    } else {
        None
    };
    let decoded = decode::decode(&data, &command.settings, observer).map_err(|error| {
        let status = match error {
            Error::Options(_) => 2,
            _ => 1,
        };
        (status, format!("{}: {error}", command.input.display()))
    })?;
    let bytes = match command.format {
        Format::Tiff => {
            let (samples, channels) = if command.ycbcr {
                if decoded.color_space != jpegio_sys::ColorSpace::YCbCr {
                    return Err((2, "--ycbcr is for files in YCbCr".into()));
                }
                // The planes, interleaved as a TIFF of chunky samples holds them.
                let size = decoded.height * decoded.width;
                let mut interleaved = Vec::with_capacity(3 * size);
                for pixel in 0..size {
                    for channel in 0..3 {
                        interleaved.push(decoded.planes[channel * size + pixel]);
                    }
                }
                (interleaved, 3)
            } else {
                (decoded.picture.clone(), decoded.channels)
            };
            tiff::float64(&samples, decoded.height, decoded.width, channels, command.ycbcr)
        }
        Format::Pnm => output::pnm(
            &decoded.picture,
            decoded.height,
            decoded.width,
            decoded.channels,
            command.bits == Bits::Sixteen,
        ),
    }
    .map_err(|error| (3, error.to_string()))?;
    write_output(&command.output, &bytes, command.overwrite)
        .map_err(|error| (3, format!("{}: {error}", command.output.display())))?;
    if let Some(path) = &command.report {
        write_output(path, report(command, &decoded).as_bytes(), true)
            .map_err(|error| (3, format!("{}: {error}", path.display())))?;
    }
    if command.verbosity != Verbosity::Quiet
        && command.output != Path::new("-")
        && let Some(result) = &decoded.result
    {
        eprintln!(
            "{}: {} iterations, {:?}",
            command.output.display(),
            result.iterations,
            result.stop
        );
    }
    Ok(())
}

/// The program: the arguments of the process, and its exit status.
#[must_use]
pub fn main() -> ExitCode {
    let arguments: Vec<OsString> = std::env::args_os().skip(1).collect();
    match parse(arguments) {
        Ok(Request::Help) => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        Ok(Request::Version) => {
            println!("{VERSION}");
            ExitCode::SUCCESS
        }
        Ok(Request::Run(command)) => match run(&command) {
            Ok(()) => ExitCode::SUCCESS,
            Err((status, message)) => {
                eprintln!("unround: {message}");
                ExitCode::from(status)
            }
        },
        Err(errors) => {
            for error in errors {
                eprintln!("unround: {error}");
            }
            eprintln!("unround: --help gives the options");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(text: &str) -> Vec<OsString> {
        text.split_whitespace().map(OsString::from).collect()
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "values are read as they are written")]
    fn the_options_reach_the_settings() {
        let Ok(Request::Run(command)) = parse(arguments(
            "--method tv --alpha 2 --mu 0.01,rule,0.5 --mu-scale 9 --centres midpoint --slack 0.5 \
             --iterations 300 --tolerance=1e-5 --relaxation 1.5 --record-every 5 --no-weight-scaling in.jpg out.tif",
        )) else {
            panic!("the arguments are taken");
        };
        assert_eq!(command.settings.method, Method::Tv);
        assert_eq!(command.settings.tv.alpha, 2.0);
        assert_eq!(command.settings.data.len(), 3);
        assert_eq!(command.settings.data[0].mu, Some(0.01));
        assert_eq!(command.settings.data[1].mu, None);
        assert_eq!(command.settings.data[1].mu_scale, 9.0);
        assert_eq!(command.settings.data[2].centres, Centres::Midpoint);
        assert_eq!(command.settings.data[2].slack, 0.5);
        assert_eq!(command.settings.pdhg.iterations, Some(300));
        assert_eq!(command.settings.pdhg.tolerance, Some(1e-5));
        assert_eq!(command.settings.pdhg.relaxation, Some(1.5));
        assert_eq!(command.settings.pdhg.record_every, 5);
        assert!(!command.settings.pdhg.scale_with_weight);
        assert_eq!(command.format, Format::Tiff);
        assert_eq!(command.output, PathBuf::from("out.tif"));
    }

    #[test]
    fn every_refused_option_is_named() {
        let Err(errors) = parse(arguments(
            "--alpha0 -1 --relaxation 2 --mu 1,2 --slack 0,0,0 --bogus --iterations x --alpha 1 --alpha 3 in.jpg",
        )) else {
            panic!("the arguments are refused");
        };
        let all = errors.join("\n");
        for fragment in [
            "--alpha0 is positive",
            "--relaxation is in (0, 2)",
            "one length",
            "unknown option --bogus",
            "--iterations is a whole number",
            "--alpha is given twice",
        ] {
            assert!(all.contains(fragment), "{fragment} in {all}");
        }
    }

    #[test]
    fn the_output_follows_the_input_and_the_format() {
        let Ok(Request::Run(command)) = parse(arguments("--format pnm --bits 16 picture.jpg")) else {
            panic!("the arguments are taken");
        };
        assert_eq!(command.output, PathBuf::from("picture.pnm"));
        assert_eq!(command.bits, Bits::Sixteen);
        assert_eq!(parse(arguments("--version")), Ok(Request::Version));
        assert!(parse(arguments("--ycbcr --format pnm in.jpg")).is_err());
    }
}
