// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The C interface of JPEG-Unround, `include/unround.h`: the library for any language
//! that calls C, the Python package among them.
//!
//! Settings are made from options by the names of the command line (docs/cli.md).
//! A file in memory -- a JPEG file, or a PNG file within the intervals of its rounded
//! samples (docs/math.md, 11) -- or components given as arrays, are reconstructed into
//! a result, whose arrays the caller reads until it frees the result; an observer can
//! follow the solver's records and stop it. A picture made elsewhere is measured
//! against a JPEG file's intervals and projected onto them. The writers of TIFF, PNG and PNM, the turning
//! of a picture upright, JFIF's conversion, the command line, and the C layer's
//! reading of JPEG files are there too.
//!
//! No panic crosses the interface: each function that computes catches one and
//! returns [`ERROR_INTERNAL`]. Errors are said in a message buffer that the caller
//! gives, NUL-terminated and cut at a character's boundary where it is too short.
//! This crate and jpegio-sys hold all the unsafe code of the Rust implementation.

use std::ffi::{CStr, CString, OsString, c_char, c_void};
use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;

use jpeg_unround::decode::{self, Component, Decoded, Input, Settings};
use jpeg_unround::pdhg::{Observer, Record as SolverRecord};
use jpeg_unround::picture::{self, Projection};
use jpeg_unround::results::{History, Stop};
use jpeg_unround::rounded::{self, Reconstructed};
use jpeg_unround::{Error, cli, colour, orientation, output, tiff};
use jpegio_sys::{ColorSpace, ffi};

/// The version of this interface: `UNROUND_ABI_VERSION` of the header it matches.
pub const ABI_VERSION: i32 = 5;

/// Success.
pub const OK: i32 = 0;
/// An argument the interface cannot take: a null pointer where one is needed, or a
/// size that does not fit.
pub const ERROR_ARGUMENT: i32 = 1;
/// A file that the C layer could not read, or would not.
pub const ERROR_READ: i32 = 2;
/// A file whose layout JPEG-Unround does not take.
pub const ERROR_UNSUPPORTED: i32 = 3;
/// Options, or arrays, out of their ranges.
pub const ERROR_OPTIONS: i32 = 4;
/// A file that could not be written or read.
pub const ERROR_IO: i32 = 5;
/// An error of JPEG-Unround itself: a panic, caught.
pub const ERROR_INTERNAL: i32 = 6;

const VERSION: &CStr = c"unround 0.1.0 (Rust, C interface)";

/// The version of this interface.
#[unsafe(no_mangle)]
pub extern "C" fn unround_abi_version() -> i32 {
    ABI_VERSION
}

/// The version of JPEG-Unround and the implementation, as a NUL-terminated string of
/// static storage.
#[unsafe(no_mangle)]
pub extern "C" fn unround_version() -> *const c_char {
    VERSION.as_ptr()
}

fn status_of(error: &Error) -> i32 {
    match error {
        Error::Read(_) => ERROR_READ,
        Error::Unsupported(_) => ERROR_UNSUPPORTED,
        Error::Options(_) => ERROR_OPTIONS,
        Error::Io(_) | Error::Write(_) => ERROR_IO,
    }
}

/// Writes `text` into a buffer of `size` bytes, NUL-terminated, cut at a character's
/// boundary where it is too long; nothing where there is no buffer.
///
/// # Safety
///
/// `buffer` is null, or valid for writes of `size` bytes.
unsafe fn write_message(buffer: *mut c_char, size: usize, text: &str) {
    if buffer.is_null() || size == 0 {
        return;
    }
    let mut end = text.len().min(size - 1);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    // SAFETY: the caller promised size writable bytes at buffer, and end + 1 <= size.
    let target = unsafe { slice::from_raw_parts_mut(buffer.cast::<u8>(), end + 1) };
    target[..end].copy_from_slice(&text.as_bytes()[..end]);
    target[end] = 0;
}

/// Runs `body`, and turns a panic into [`ERROR_INTERNAL`], with its message.
fn guarded(message: *mut c_char, size: usize, body: impl FnOnce() -> Result<(), (i32, String)>) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(body));
    let (status, text) = match outcome {
        Ok(Ok(())) => return OK,
        Ok(Err((status, text))) => (status, text),
        Err(panic) => {
            let text = panic
                .downcast_ref::<&str>()
                .map(|text| (*text).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic without a message".to_owned());
            (ERROR_INTERNAL, format!("an internal error of JPEG-Unround: {text}"))
        }
    };
    // SAFETY: every function that calls this promises a message buffer of size bytes,
    // or a null one.
    unsafe { write_message(message, size, &text) };
    status
}

fn refuse(text: &str) -> (i32, String) {
    (ERROR_ARGUMENT, text.to_owned())
}

/// Settings of a reconstruction: the method, the model and the solvers' options.
#[derive(Debug)]
pub struct UnroundSettings {
    settings: Settings,
}

/// The strings of an array of `count` NUL-terminated UTF-8 strings.
///
/// # Safety
///
/// `strings` is valid for reads of `count` pointers, each to a NUL-terminated string,
/// or null where `count` is 0.
unsafe fn strings(count: usize, strings: *const *const c_char) -> Result<Vec<OsString>, (i32, String)> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if strings.is_null() {
        return Err(refuse("the strings are a null pointer"));
    }
    // SAFETY: the caller promised count pointers at strings.
    let pointers = unsafe { slice::from_raw_parts(strings, count) };
    pointers
        .iter()
        .map(|&pointer| {
            if pointer.is_null() {
                return Err(refuse("a string is a null pointer"));
            }
            // SAFETY: the caller promised a NUL-terminated string at each pointer.
            let text = unsafe { CStr::from_ptr(pointer) };
            text.to_str()
                .map(|text| OsString::from(text.to_owned()))
                .map_err(|_| refuse("a string is not UTF-8"))
        })
        .collect()
}

/// Settings from `count` options, each a NUL-terminated UTF-8 string, as the command
/// line takes them without its files and its options of the output: `"--mu"`,
/// `"0.01"`, `"--method=tv"`. Null where they are refused, with every refusal in the
/// message.
///
/// # Safety
///
/// `options` is valid for reads of `count` pointers to NUL-terminated strings (or null
/// where `count` is 0), and `message` valid for writes of `message_size` bytes, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_settings_new(
    count: usize,
    options: *const *const c_char,
    message: *mut c_char,
    message_size: usize,
) -> *mut UnroundSettings {
    let mut made = ptr::null_mut();
    guarded(message, message_size, || {
        // SAFETY: the caller's promise of options is passed on.
        let options = unsafe { strings(count, options) }?;
        let settings = cli::settings_of(options).map_err(|errors| (ERROR_OPTIONS, errors.join("; ")))?;
        made = Box::into_raw(Box::new(UnroundSettings { settings }));
        Ok(())
    });
    made
}

/// Frees settings, where they are not null.
///
/// # Safety
///
/// `settings` is null, or settings that `unround_settings_new` made and that are not
/// freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_settings_free(settings: *mut UnroundSettings) {
    if !settings.is_null() {
        // SAFETY: the caller promised settings made by Box::into_raw, freed once.
        drop(unsafe { Box::from_raw(settings) });
    }
}

/// Writes the options that give the settings, as `unround_settings_new` takes them,
/// into a buffer of `size` bytes: `--name=value` or `--flag` one after another, a
/// space between, each number as the shortest decimal that reads back as the same
/// double; NUL-terminated, and cut where the buffer is too short. Returns the length
/// of the whole text, without the NUL; 0 for null settings.
///
/// # Safety
///
/// `settings` is null or live settings, and `buffer` valid for writes of `size` bytes,
/// or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_settings_options(
    settings: *const UnroundSettings,
    buffer: *mut c_char,
    size: usize,
) -> usize {
    // SAFETY: the caller promised null or live settings.
    let Some(settings) = (unsafe { settings.as_ref() }) else {
        return 0;
    };
    catch_unwind(AssertUnwindSafe(|| cli::options_of(&settings.settings).join(" "))).map_or(0, |text| {
        // SAFETY: the caller promised a buffer of size bytes, or a null one.
        unsafe { write_message(buffer, size, &text) };
        text.len()
    })
}

/// A record of the solver, as an observer sees it: the iteration, its gap per
/// sample, the primal and dual values, and the canvas of `channels x height x width`
/// samples, valid for the length of the call.
#[repr(C)]
#[derive(Debug)]
pub struct UnroundRecord {
    /// The iteration.
    pub iteration: u64,
    /// The gap per sample, which the tolerance is compared with.
    pub gap: f64,
    /// The primal value.
    pub primal: f64,
    /// The dual value.
    pub dual: f64,
    /// The channels of the canvas.
    pub channels: u64,
    /// Its rows.
    pub height: u64,
    /// Its columns.
    pub width: u64,
    /// Its samples, channel by channel, each row by row.
    pub canvas: *const f64,
}

const _: () = assert!(size_of::<UnroundRecord>() == 64);

/// What the solver calls with each record after the first: nonzero stops it.
pub type UnroundObserver = Option<unsafe extern "C" fn(user: *mut c_void, record: *const UnroundRecord) -> i32>;

fn wide(value: usize) -> u64 {
    // Lossless on the 64-bit systems JPEG-Unround supports.
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn narrow(value: u64) -> Result<usize, (i32, String)> {
    usize::try_from(value).map_err(|_| refuse("a size that does not fit"))
}

/// A result of a JPEG file, or of components given as arrays (`unround_result_format`).
pub const FORMAT_JPEG: i32 = 1;
/// A result of a PNG file, reconstructed within the intervals of its rounded samples.
pub const FORMAT_PNG: i32 = 2;

/// A reconstruction: what the solver made, and what the caller reads of it.
#[derive(Debug)]
pub struct UnroundResult {
    made: Made,
    /// The first warning of the reading, or an empty string, NUL-terminated.
    warning: CString,
    /// Of a projection, how far each coefficient of the picture given lay outside its
    /// interval, component by component; none for a reconstruction.
    excess: Vec<Vec<f64>>,
}

/// A JPEG file's reconstruction, or a PNG file's.
#[derive(Debug)]
enum Made {
    /// A JPEG file's, or its components'.
    Jpeg(Box<Decoded>),
    /// A PNG file's.
    Png(Box<Rounded>),
}

/// A PNG file's reconstruction, and its picture's planes: the colours', then the
/// alpha's.
#[derive(Debug)]
struct Rounded {
    reconstructed: Reconstructed,
    planes: Vec<f64>,
}

/// What a solver did: its iterations, why it stopped, and its records.
#[derive(Debug, Clone, Copy)]
struct Solved<'a> {
    iterations: u64,
    stop: Stop,
    history: &'a History,
}

/// A warning as a C string: what lies before a NUL, which no message of the library
/// holds.
fn c_string(text: &str) -> CString {
    let end = text.find('\0').unwrap_or(text.len());
    CString::new(&text[..end]).unwrap_or_default()
}

impl UnroundResult {
    fn jpeg(decoded: Decoded) -> Self {
        let warning = c_string(&decoded.warning);
        Self {
            made: Made::Jpeg(Box::new(decoded)),
            warning,
            excess: Vec::new(),
        }
    }

    fn projection(projection: Projection) -> Self {
        let Projection { decoded, excess } = projection;
        Self {
            excess,
            ..Self::jpeg(decoded)
        }
    }

    fn png(reconstructed: Reconstructed) -> Self {
        let mut planes = reconstructed.colours.picture.clone();
        if let Some(alpha) = &reconstructed.alpha {
            planes.extend_from_slice(&alpha.picture);
        }
        let warning = c_string(&reconstructed.warning);
        Self {
            made: Made::Png(Box::new(Rounded { reconstructed, planes })),
            warning,
            excess: Vec::new(),
        }
    }

    /// The picture, interleaved, and its rows, columns and channels.
    fn picture(&self) -> (&[f64], usize, usize, usize) {
        match &self.made {
            Made::Jpeg(decoded) => (&decoded.picture, decoded.height, decoded.width, decoded.channels),
            Made::Png(rounded) => (
                &rounded.reconstructed.picture,
                rounded.reconstructed.height,
                rounded.reconstructed.width,
                rounded.reconstructed.channels,
            ),
        }
    }

    /// The picture channel by channel: a JPEG file's canvas cut to it, a PNG file's
    /// colours and alpha.
    fn planes(&self) -> &[f64] {
        match &self.made {
            Made::Jpeg(decoded) => &decoded.planes,
            Made::Png(rounded) => &rounded.planes,
        }
    }

    fn color_space(&self) -> i32 {
        match &self.made {
            Made::Jpeg(decoded) => match decoded.color_space {
                ColorSpace::Grayscale => ffi::GRAYSCALE,
                ColorSpace::YCbCr => ffi::YCBCR,
                ColorSpace::Rgb => ffi::RGB,
            },
            Made::Png(rounded) => {
                if rounded.reconstructed.channels <= 2 {
                    ffi::GRAYSCALE
                } else {
                    ffi::RGB
                }
            }
        }
    }

    /// The whole canvas, `channels x height x width`, as the solver left it: of a PNG
    /// file, the picture's planes, since nothing lies beyond the picture.
    fn canvas(&self) -> (&[f64], usize, usize, usize) {
        match &self.made {
            Made::Jpeg(decoded) => (
                &decoded.canvas,
                decoded.frame.channels().len(),
                decoded.frame.height(),
                decoded.frame.width(),
            ),
            Made::Png(rounded) => (
                &rounded.planes,
                rounded.reconstructed.channels,
                rounded.reconstructed.height,
                rounded.reconstructed.width,
            ),
        }
    }

    /// The reconstruction of a JPEG file, whose components a PNG file has none of.
    fn decoded(&self) -> Option<&Decoded> {
        match &self.made {
            Made::Jpeg(decoded) => Some(decoded),
            Made::Png(_) => None,
        }
    }

    /// The solver of the picture, or of a PNG file's colours: none for the MMSE decoder.
    fn solved(&self) -> Option<Solved<'_>> {
        match &self.made {
            Made::Jpeg(decoded) => decoded.result.as_ref().map(|result| Solved {
                iterations: result.iterations,
                stop: result.stop,
                history: &result.history,
            }),
            Made::Png(rounded) => Some(Solved {
                iterations: rounded.reconstructed.colours.iterations,
                stop: rounded.reconstructed.colours.stop,
                history: &rounded.reconstructed.colours.history,
            }),
        }
    }

    /// The solver of a PNG file's alpha, reconstructed alone.
    fn alpha(&self) -> Option<Solved<'_>> {
        match &self.made {
            Made::Jpeg(_) => None,
            Made::Png(rounded) => rounded.reconstructed.alpha.as_ref().map(|alpha| Solved {
                iterations: alpha.iterations,
                stop: alpha.stop,
                history: &alpha.history,
            }),
        }
    }

    fn icc_profile(&self) -> Option<&[u8]> {
        match &self.made {
            Made::Jpeg(decoded) => decoded.icc_profile.as_deref(),
            Made::Png(rounded) => rounded.reconstructed.icc_profile.as_deref(),
        }
    }

    fn exif_orientation(&self) -> i32 {
        match &self.made {
            Made::Jpeg(decoded) => decoded.exif_orientation,
            Made::Png(_) => 0,
        }
    }

    fn format(&self) -> i32 {
        match &self.made {
            Made::Jpeg(_) => FORMAT_JPEG,
            Made::Png(_) => FORMAT_PNG,
        }
    }

    /// The bits of a sample of the file: a PNG file's 1 to 16, a JPEG file's 8, which
    /// are the only ones the C layer reads.
    fn bits(&self) -> i32 {
        match &self.made {
            Made::Jpeg(_) => 8,
            Made::Png(rounded) => i32::try_from(rounded.reconstructed.bits).unwrap_or(0),
        }
    }
}

/// Runs `body` with the caller's observer, where one is given: each record goes to
/// the callback with `user`, and a nonzero return stops the solver.
///
/// # Safety
///
/// `callback` is null, or callable with `user` and a record for the length of the call.
unsafe fn observed<T>(
    callback: UnroundObserver,
    user: *mut c_void,
    body: impl FnOnce(Option<&mut Observer<'_>>) -> Result<T, Error>,
) -> Result<T, (i32, String)> {
    let mut observe = |record: &SolverRecord<'_>| {
        let Some(callback) = callback else {
            return ControlFlow::Continue(());
        };
        let raw = UnroundRecord {
            iteration: record.iteration,
            gap: record.gap,
            primal: record.primal,
            dual: record.dual,
            channels: wide(record.canvas.len() / (record.height * record.width).max(1)),
            height: wide(record.height),
            width: wide(record.width),
            canvas: record.canvas.as_ptr(),
        };
        // SAFETY: the caller promised a callback callable with user and a record, which
        // lives until the call returns.
        let status = unsafe { callback(user, &raw const raw) };
        if status == 0 {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    };
    let observer: Option<&mut Observer<'_>> = if callback.is_some() { Some(&mut observe) } else { None };
    body(observer).map_err(|error| (status_of(&error), error.to_string()))
}

/// Hands a result over to the caller, who frees it with `unround_result_free`.
///
/// # Safety
///
/// `result` is valid for a write.
unsafe fn deliver(made: UnroundResult, result: *mut *mut UnroundResult) {
    let made = Box::into_raw(Box::new(made));
    // SAFETY: the caller promised result valid for a write.
    unsafe { result.write(made) };
}

/// Reconstructs a file of `size` bytes at `data` with the settings, into a result that
/// `unround_result_free` frees: a PNG file, known by its signature, within the
/// intervals of its rounded samples (docs/math.md, 11), and any other as a JPEG file.
/// `observer`, where it is not null, is called with `user` and each record after the
/// first -- of a PNG file with alpha, the colours' and then the alpha's -- and stops
/// the solver where it returns nonzero.
///
/// # Safety
///
/// `data` is valid for reads of `size` bytes; `settings` are settings of
/// `unround_settings_new`; `observer` is null, or callable with `user` and a record;
/// `result` is valid for a write; `message` valid for writes of `message_size` bytes,
/// or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_decode(
    data: *const u8,
    size: usize,
    settings: *const UnroundSettings,
    observer: UnroundObserver,
    user: *mut c_void,
    result: *mut *mut UnroundResult,
    message: *mut c_char,
    message_size: usize,
) -> i32 {
    guarded(message, message_size, || {
        if data.is_null() || settings.is_null() || result.is_null() {
            return Err(refuse("the data, the settings and the result are not null pointers"));
        }
        // SAFETY: the caller promised size readable bytes at data.
        let bytes = unsafe { slice::from_raw_parts(data, size) };
        // SAFETY: the caller promised settings of unround_settings_new.
        let settings = unsafe { &(*settings).settings };
        let made = if rounded::is_png(bytes) {
            // SAFETY: the caller's promises of observer and user are passed on.
            let reconstructed =
                unsafe { observed(observer, user, |watch| rounded::reconstruct(bytes, settings, watch)) }?;
            UnroundResult::png(reconstructed)
        } else {
            let input = Input::read(bytes, &settings.read).map_err(|error| (status_of(&error), error.to_string()))?;
            // SAFETY: as above.
            let decoded = unsafe { observed(observer, user, |watch| decode::solve(&input, settings, watch)) }?;
            UnroundResult::jpeg(decoded)
        };
        // SAFETY: the caller promised result valid for a write.
        unsafe { deliver(made, result) };
        Ok(())
    })
}

/// A component given as arrays: `rows x columns` blocks of 64 levels in natural
/// order, block rows from the top, its sampling factors and its quantization table.
#[repr(C)]
#[derive(Debug)]
pub struct UnroundComponent {
    /// The levels.
    pub levels: *const i16,
    /// Block rows.
    pub rows: u32,
    /// Blocks across.
    pub columns: u32,
    /// The horizontal sampling factor.
    pub h_samp_factor: i32,
    /// The vertical sampling factor.
    pub v_samp_factor: i32,
    /// The quantization table, in natural order.
    pub quant_table: [u16; 64],
}

const _: () = assert!(size_of::<UnroundComponent>() == 152);

/// Components given as arrays: the picture's rows and columns, its color space (1
/// greyscale, 2 YCbCr, 3 RGB, as the C layer says it), and `count` components.
#[repr(C)]
#[derive(Debug)]
pub struct UnroundInput {
    /// Rows of the picture.
    pub height: u32,
    /// Columns of the picture.
    pub width: u32,
    /// The color space.
    pub color_space: i32,
    /// The number of components.
    pub count: u32,
    /// The components.
    pub components: *const UnroundComponent,
}

const _: () = assert!(size_of::<UnroundInput>() == 24);

fn usize_of(value: u32) -> usize {
    // Lossless on the 64-bit systems JPEG-Unround supports.
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// The input of components given as arrays, copied.
///
/// # Safety
///
/// `given` points to components whose arrays are valid for their sizes.
unsafe fn input_of(given: &UnroundInput) -> Result<Input, (i32, String)> {
    let color_space = match given.color_space {
        ffi::GRAYSCALE => ColorSpace::Grayscale,
        ffi::YCBCR => ColorSpace::YCbCr,
        ffi::RGB => ColorSpace::Rgb,
        other => return Err(refuse(&format!("a color space of 1, 2 or 3, not {other}"))),
    };
    if given.components.is_null() || given.count == 0 {
        return Err(refuse("components are given"));
    }
    // SAFETY: the caller promised count components at the pointer.
    let components = unsafe { slice::from_raw_parts(given.components, usize_of(given.count)) };
    let mut copied = Vec::with_capacity(components.len());
    for component in components {
        let (rows, columns) = (usize_of(component.rows), usize_of(component.columns));
        let count = rows
            .checked_mul(columns)
            .and_then(|blocks| blocks.checked_mul(64))
            .ok_or_else(|| refuse("a component too large"))?;
        if component.levels.is_null() {
            return Err(refuse("a component's levels are a null pointer"));
        }
        let factor = |value: i32| usize::try_from(value).map_err(|_| refuse("a sampling factor below 0"));
        // SAFETY: the caller promised rows x columns blocks of 64 levels at the pointer.
        let levels = unsafe { slice::from_raw_parts(component.levels, count) };
        copied.push(Component {
            levels: levels.to_vec(),
            rows,
            columns,
            quant_table: component.quant_table,
            h_samp_factor: factor(component.h_samp_factor)?,
            v_samp_factor: factor(component.v_samp_factor)?,
        });
    }
    Ok(Input {
        height: usize_of(given.height),
        width: usize_of(given.width),
        color_space,
        components: copied,
        icc_profile: None,
        exif_orientation: 0,
        warning: String::new(),
    })
}

/// Reconstructs components given as arrays, as `unround_decode` does a file.
///
/// # Safety
///
/// `input` is valid, with arrays valid for their sizes; the rest as for
/// `unround_decode`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_solve(
    input: *const UnroundInput,
    settings: *const UnroundSettings,
    observer: UnroundObserver,
    user: *mut c_void,
    result: *mut *mut UnroundResult,
    message: *mut c_char,
    message_size: usize,
) -> i32 {
    guarded(message, message_size, || {
        if input.is_null() || settings.is_null() || result.is_null() {
            return Err(refuse("the input, the settings and the result are not null pointers"));
        }
        // SAFETY: the caller promised a valid input.
        let given = unsafe { &*input };
        // SAFETY: the caller promised its arrays valid for their sizes.
        let input = unsafe { input_of(given) }?;
        // SAFETY: the caller promised settings of unround_settings_new.
        let settings = unsafe { &(*settings).settings };
        // SAFETY: the caller's promises of observer and user are passed on.
        let decoded = unsafe { observed(observer, user, |watch| decode::solve(&input, settings, watch)) }?;
        // SAFETY: the caller promised result valid for a write.
        unsafe { deliver(UnroundResult::jpeg(decoded), result) };
        Ok(())
    })
}

/// Measures a picture against a JPEG file's intervals, with the settings' slack, and
/// projects it onto them (docs/math.md, 12), into a result that `unround_result_free`
/// frees. The picture, `height x width x channels` binary64 samples interleaved, is of
/// the file's size as the file stores it, before its EXIF orientation: greyscale, or
/// of a file in colour RGB. The result is the projection, as a reconstruction without
/// a solver, and holds the picture's excess (`unround_result_excess`).
///
/// # Safety
///
/// `picture` is valid for reads of `height x width x channels` doubles; the rest as
/// for `unround_decode`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_project(
    data: *const u8,
    size: usize,
    settings: *const UnroundSettings,
    samples: *const f64,
    height: u64,
    width: u64,
    channels: u64,
    result: *mut *mut UnroundResult,
    message: *mut c_char,
    message_size: usize,
) -> i32 {
    guarded(message, message_size, || {
        if data.is_null() || settings.is_null() || result.is_null() {
            return Err(refuse("the data, the settings and the result are not null pointers"));
        }
        // SAFETY: the caller promised size readable bytes at data.
        let bytes = unsafe { slice::from_raw_parts(data, size) };
        if rounded::is_png(bytes) {
            return Err((
                ERROR_OPTIONS,
                "a picture is projected onto the intervals of a JPEG file, not of a PNG file".to_owned(),
            ));
        }
        // SAFETY: the caller promised settings of unround_settings_new.
        let settings = unsafe { &(*settings).settings };
        // SAFETY: the caller promised that many doubles at samples.
        let given = unsafe { picture(samples, height, width, channels) }?;
        let input = Input::read(bytes, &settings.read).map_err(|error| (status_of(&error), error.to_string()))?;
        if (wide(input.height), wide(input.width)) != (height, width) {
            return Err((
                ERROR_OPTIONS,
                format!(
                    "a picture of the file is {} x {}, not {height} x {width}",
                    input.height, input.width
                ),
            ));
        }
        let projection = picture::project(&input, settings, given, narrow(channels)?)
            .map_err(|error| (status_of(&error), error.to_string()))?;
        // SAFETY: the caller promised result valid for a write.
        unsafe { deliver(UnroundResult::projection(projection), result) };
        Ok(())
    })
}

/// Frees a result, where it is not null.
///
/// # Safety
///
/// `result` is null, or a result of `unround_decode` or `unround_solve` not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_free(result: *mut UnroundResult) {
    if !result.is_null() {
        // SAFETY: the caller promised a result made by Box::into_raw, freed once.
        drop(unsafe { Box::from_raw(result) });
    }
}

/// A result, where the pointer is not null.
///
/// # Safety
///
/// `result` is null, or a result not freed yet, which lives for `'a`.
unsafe fn result_of<'a>(result: *const UnroundResult) -> Option<&'a UnroundResult> {
    // SAFETY: the caller promised null or a live result.
    unsafe { result.as_ref() }
}

/// Writes a value where the pointer is not null.
///
/// # Safety
///
/// `target` is null, or valid for a write.
unsafe fn put<T>(target: *mut T, value: T) {
    if !target.is_null() {
        // SAFETY: the caller promised a pointer valid for a write.
        unsafe { target.write(value) };
    }
}

/// The picture of a result, `height x width x channels` binary64 samples, interleaved:
/// greyscale or RGB, and of a PNG file with alpha, the alpha last. Its sizes are
/// written where the pointers are not null.
///
/// # Safety
///
/// `result` is a live result; each pointer is null or valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_picture(
    result: *const UnroundResult,
    height: *mut u64,
    width: *mut u64,
    channels: *mut u64,
) -> *const f64 {
    // SAFETY: the caller promised a live result.
    let Some(made) = (unsafe { result_of(result) }) else {
        return ptr::null();
    };
    let (picture, rows, columns, count) = made.picture();
    // SAFETY: the caller promised each pointer null or valid for a write.
    unsafe { put(height, wide(rows)) };
    // SAFETY: as above.
    unsafe { put(width, wide(columns)) };
    // SAFETY: as above.
    unsafe { put(channels, wide(count)) };
    picture.as_ptr()
}

/// The planes of a result, channel by channel, `count` samples in all: of a JPEG file,
/// the canvas cut to the picture (the Y, Cb and Cr of a file in YCbCr); of a PNG file,
/// the picture's colours and its alpha.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_planes(result: *const UnroundResult, count: *mut u64) -> *const f64 {
    // SAFETY: the caller promised a live result.
    let Some(made) = (unsafe { result_of(result) }) else {
        return ptr::null();
    };
    let planes = made.planes();
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(count, wide(planes.len())) };
    planes.as_ptr()
}

/// The color space of a result's file: 1 greyscale, 2 YCbCr, 3 RGB (a PNG file's
/// colours are greyscale or RGB, with alpha or without); 0 for a null result.
///
/// # Safety
///
/// `result` is null, or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_color_space(result: *const UnroundResult) -> i32 {
    // SAFETY: the caller promised null or a live result.
    unsafe { result_of(result) }.map_or(0, UnroundResult::color_space)
}

/// The whole canvas of a result, `channels x height x width` samples, as the solver
/// left it: the picture and what lies beyond it, of a PNG file the planes, since
/// nothing lies beyond its picture.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_canvas(
    result: *const UnroundResult,
    channels: *mut u64,
    height: *mut u64,
    width: *mut u64,
) -> *const f64 {
    // SAFETY: the caller promised a live result.
    let Some(made) = (unsafe { result_of(result) }) else {
        return ptr::null();
    };
    let (canvas, count, rows, columns) = made.canvas();
    // SAFETY: the caller promised each pointer null or valid for a write.
    unsafe { put(channels, wide(count)) };
    // SAFETY: as above.
    unsafe { put(height, wide(rows)) };
    // SAFETY: as above.
    unsafe { put(width, wide(columns)) };
    canvas.as_ptr()
}

/// The number of components of a result: those of a JPEG file, none of a PNG file's;
/// 0 for a null result.
///
/// # Safety
///
/// `result` is null, or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_components(result: *const UnroundResult) -> u64 {
    // SAFETY: the caller promised null or a live result.
    unsafe { result_of(result) }
        .and_then(UnroundResult::decoded)
        .map_or(0, |decoded| wide(decoded.coefficients.len()))
}

/// The coefficients of component `component` of a result, each within its interval,
/// `count` of them, 64 to a block in natural order; null for a component that is not
/// there.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_coefficients(
    result: *const UnroundResult,
    component: u64,
    count: *mut u64,
) -> *const f64 {
    // SAFETY: the caller promised a live result.
    let Some(decoded) = (unsafe { result_of(result) }).and_then(UnroundResult::decoded) else {
        return ptr::null();
    };
    let Some(values) = usize::try_from(component)
        .ok()
        .and_then(|index| decoded.coefficients.get(index))
    else {
        return ptr::null();
    };
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(count, wide(values.len())) };
    values.as_ptr()
}

/// What the model of component `component` holds, `count` values: `which` 0, 1 and 2
/// the lower and upper ends of the intervals and the centres of the data term, one for
/// each coefficient; 3 and 4 the quantization steps and the weights of the data term,
/// one for each frequency (64). Null for a component or a `which` that is not there.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_problem(
    result: *const UnroundResult,
    component: u64,
    which: i32,
    count: *mut u64,
) -> *const f64 {
    // SAFETY: the caller promised a live result.
    let Some(decoded) = (unsafe { result_of(result) }).and_then(UnroundResult::decoded) else {
        return ptr::null();
    };
    let Some(channel) = usize::try_from(component)
        .ok()
        .and_then(|index| decoded.frame.channels().get(index))
    else {
        return ptr::null();
    };
    let problem = channel.problem();
    let values: &[f64] = match which {
        0 => problem.lower(),
        1 => problem.upper(),
        2 => problem.centres(),
        3 => problem.steps(),
        4 => problem.weights(),
        _ => return ptr::null(),
    };
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(count, wide(values.len())) };
    values.as_ptr()
}

/// The shape of component `component` of a result: its block rows and columns, and
/// the canvas's samples per sample of the component down and across. `ERROR_ARGUMENT`
/// for a component that is not there.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_component(
    result: *const UnroundResult,
    component: u64,
    rows: *mut u64,
    columns: *mut u64,
    down: *mut u64,
    across: *mut u64,
) -> i32 {
    // SAFETY: the caller promised a live result.
    let Some(decoded) = (unsafe { result_of(result) }).and_then(UnroundResult::decoded) else {
        return ERROR_ARGUMENT;
    };
    let Some(channel) = usize::try_from(component)
        .ok()
        .and_then(|index| decoded.frame.channels().get(index))
    else {
        return ERROR_ARGUMENT;
    };
    let (ratio_down, ratio_across) = channel.ratio();
    // SAFETY: the caller promised each pointer null or valid for a write.
    unsafe { put(rows, wide(channel.problem().rows())) };
    // SAFETY: as above.
    unsafe { put(columns, wide(channel.problem().columns())) };
    // SAFETY: as above.
    unsafe { put(down, wide(ratio_down)) };
    // SAFETY: as above.
    unsafe { put(across, wide(ratio_across)) };
    OK
}

/// Writes a solver's iterations and why it stopped -- 1 a tolerance, 2 the most
/// iterations, 3 the observer, 4 a subgradient of 0 -- and says whether there was
/// one (1) or not (0).
///
/// # Safety
///
/// Each pointer is null or valid for a write.
unsafe fn solver(solved: Option<Solved<'_>>, iterations: *mut u64, stop: *mut i32) -> i32 {
    let Some(solved) = solved else {
        return 0;
    };
    let reason = match solved.stop {
        Stop::Converged => 1,
        Stop::Iterations => 2,
        Stop::Observer => 3,
        Stop::Stationary => 4,
    };
    // SAFETY: the caller promised each pointer null or valid for a write.
    unsafe { put(iterations, solved.iterations) };
    // SAFETY: as above.
    unsafe { put(stop, reason) };
    1
}

/// The iterations of a solver's records; null where there was no solver.
///
/// # Safety
///
/// `count` is null or valid for a write.
unsafe fn record_iterations(solved: Option<Solved<'_>>, count: *mut u64) -> *const u64 {
    let Some(solved) = solved else {
        return ptr::null();
    };
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(count, wide(solved.history.iterations.len())) };
    solved.history.iterations.as_ptr()
}

/// A value of a solver's records; null where there was no solver, or for a `which`
/// that is not there.
///
/// # Safety
///
/// `count` is null or valid for a write.
unsafe fn record_values(solved: Option<Solved<'_>>, which: i32, count: *mut u64) -> *const f64 {
    let Some(solved) = solved else {
        return ptr::null();
    };
    let history = solved.history;
    let values = match which {
        0 => &history.seconds,
        1 => &history.primal,
        2 => &history.dual,
        3 => &history.scaling,
        4 => &history.partial_gap,
        _ => return ptr::null(),
    };
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(count, wide(values.len())) };
    values.as_ptr()
}

/// Whether a solver made a result (1) or not (0: the decoder of the centres), with
/// the iterations it took and why it stopped: 1 a tolerance, 2 the most iterations,
/// 3 the observer, 4 a subgradient of 0. Of a PNG file, the solver of its colours.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_solver(
    result: *const UnroundResult,
    iterations: *mut u64,
    stop: *mut i32,
) -> i32 {
    // SAFETY: the caller promised a live result.
    let solved = unsafe { result_of(result) }.and_then(UnroundResult::solved);
    // SAFETY: the caller promised each pointer null or valid for a write.
    unsafe { solver(solved, iterations, stop) }
}

/// The iterations of the solver's records, `count` of them; null where no solver ran.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_record_iterations(result: *const UnroundResult, count: *mut u64) -> *const u64 {
    // SAFETY: the caller promised a live result.
    let solved = unsafe { result_of(result) }.and_then(UnroundResult::solved);
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { record_iterations(solved, count) }
}

/// A value of the solver's records, one for each: `which` 0 the solver's seconds, 1 the
/// primal value, 2 the dual value, 3 TGV's scaling of the dual, 4 TGV's partial gap
/// (NaN where not taken). Null where no solver ran, or for a `which` that is not there.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_record_values(
    result: *const UnroundResult,
    which: i32,
    count: *mut u64,
) -> *const f64 {
    // SAFETY: the caller promised a live result.
    let solved = unsafe { result_of(result) }.and_then(UnroundResult::solved);
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { record_values(solved, which, count) }
}

/// Whether a result has an alpha reconstructed alone (1: a PNG file's with alpha) or
/// not (0), with its solver's iterations and why it stopped, as
/// `unround_result_solver` says them.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_alpha_solver(
    result: *const UnroundResult,
    iterations: *mut u64,
    stop: *mut i32,
) -> i32 {
    // SAFETY: the caller promised a live result.
    let solved = unsafe { result_of(result) }.and_then(UnroundResult::alpha);
    // SAFETY: the caller promised each pointer null or valid for a write.
    unsafe { solver(solved, iterations, stop) }
}

/// The iterations of the records of the alpha's solver, `count` of them; null where
/// the result has no alpha.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_alpha_record_iterations(
    result: *const UnroundResult,
    count: *mut u64,
) -> *const u64 {
    // SAFETY: the caller promised a live result.
    let solved = unsafe { result_of(result) }.and_then(UnroundResult::alpha);
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { record_iterations(solved, count) }
}

/// A value of the records of the alpha's solver, as `unround_result_record_values`
/// says them; null where the result has no alpha, or for a `which` that is not there.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_alpha_record_values(
    result: *const UnroundResult,
    which: i32,
    count: *mut u64,
) -> *const f64 {
    // SAFETY: the caller promised a live result.
    let solved = unsafe { result_of(result) }.and_then(UnroundResult::alpha);
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { record_values(solved, which, count) }
}

/// The ICC profile of a result's file, `size` bytes; null where it has none.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_icc_profile(result: *const UnroundResult, size: *mut u64) -> *const u8 {
    // SAFETY: the caller promised a live result.
    let Some(profile) = (unsafe { result_of(result) }).and_then(UnroundResult::icc_profile) else {
        return ptr::null();
    };
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(size, wide(profile.len())) };
    profile.as_ptr()
}

/// The EXIF orientation of a result's file, 1 to 8, or 0 where it has none (and for a
/// PNG file).
///
/// # Safety
///
/// `result` is null, or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_exif_orientation(result: *const UnroundResult) -> i32 {
    // SAFETY: the caller promised null or a live result.
    unsafe { result_of(result) }.map_or(0, UnroundResult::exif_orientation)
}

/// The format of a result's file: `FORMAT_JPEG` (and for components given as arrays)
/// or `FORMAT_PNG`; 0 for a null result.
///
/// # Safety
///
/// `result` is null, or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_format(result: *const UnroundResult) -> i32 {
    // SAFETY: the caller promised null or a live result.
    unsafe { result_of(result) }.map_or(0, UnroundResult::format)
}

/// The bits of a sample of a result's file: a PNG file's, 1 to 16; a JPEG file's, 8;
/// 0 for a null result.
///
/// # Safety
///
/// `result` is null, or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_bits(result: *const UnroundResult) -> i32 {
    // SAFETY: the caller promised null or a live result.
    unsafe { result_of(result) }.map_or(0, UnroundResult::bits)
}

/// The first warning of the reading of a result's file, NUL-terminated and valid
/// while the result lives: a corrupt-data warning of a JPEG file, or a PNG file's
/// transparent colour that is not taken; an empty string where there was none. Null
/// for a null result.
///
/// # Safety
///
/// `result` is null, or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_warning(result: *const UnroundResult) -> *const c_char {
    // SAFETY: the caller promised null or a live result.
    unsafe { result_of(result) }.map_or(ptr::null(), |made| made.warning.as_ptr())
}

/// How far each coefficient of component `component` of the picture given to
/// `unround_project` lay outside its interval, in steps (0 within it), `count` of
/// them, 64 to a block in natural order; null for a result of a reconstruction, or a
/// component that is not there.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_excess(
    result: *const UnroundResult,
    component: u64,
    count: *mut u64,
) -> *const f64 {
    // SAFETY: the caller promised a live result.
    let Some(made) = (unsafe { result_of(result) }) else {
        return ptr::null();
    };
    let Some(values) = usize::try_from(component).ok().and_then(|index| made.excess.get(index)) else {
        return ptr::null();
    };
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(count, wide(values.len())) };
    values.as_ptr()
}

/// Samples of `height x width x channels`, from a pointer.
///
/// # Safety
///
/// `samples` is valid for reads of that many doubles.
unsafe fn picture<'a>(samples: *const f64, height: u64, width: u64, channels: u64) -> Result<&'a [f64], (i32, String)> {
    let count = narrow(height)?
        .checked_mul(narrow(width)?)
        .and_then(|size| size.checked_mul(usize::try_from(channels).ok()?))
        .ok_or_else(|| refuse("a picture too large"))?;
    if samples.is_null() {
        return Err(refuse("the samples are a null pointer"));
    }
    // SAFETY: the caller promised count doubles at samples.
    Ok(unsafe { slice::from_raw_parts(samples, count) })
}

/// Hands bytes over to the caller, who frees them with `unround_bytes_free`.
///
/// # Safety
///
/// `bytes` and `size` are valid for writes.
unsafe fn hand_over(made: Vec<u8>, bytes: *mut *mut u8, size: *mut u64) {
    let boxed = made.into_boxed_slice();
    let length = wide(boxed.len());
    let pointer = Box::into_raw(boxed).cast::<u8>();
    // SAFETY: the caller promised bytes valid for a write.
    unsafe { bytes.write(pointer) };
    // SAFETY: the caller promised size valid for a write.
    unsafe { size.write(length) };
}

/// An ICC profile of `size` bytes at `icc`, or none where `icc` is null.
///
/// # Safety
///
/// `icc` is null, or valid for reads of `size` bytes.
unsafe fn profile<'a>(icc: *const u8, size: u64) -> Result<Option<&'a [u8]>, (i32, String)> {
    if icc.is_null() {
        return Ok(None);
    }
    let length = narrow(size)?;
    // SAFETY: the caller promised size bytes at icc.
    Ok(Some(unsafe { slice::from_raw_parts(icc, length) }))
}

/// Hands a file over to the caller, and its warning into the message.
///
/// # Safety
///
/// `bytes` and `size` are valid for writes; `message` for writes of `message_size`
/// bytes, or null.
unsafe fn hand_over_written(
    written: output::Written,
    bytes: *mut *mut u8,
    size: *mut u64,
    message: *mut c_char,
    message_size: usize,
) {
    // SAFETY: the caller's promises are passed on.
    unsafe { hand_over(written.bytes, bytes, size) };
    // SAFETY: as above.
    unsafe { write_message(message, message_size, &written.warning) };
}

/// The bytes of a TIFF of binary64 samples, `height x width x channels` (1 to 4
/// interleaved: grey or RGB, and with 2 or 4 the alpha last, which the file declares),
/// exactly as they are; with `ycbcr` nonzero, the three samples of a
/// pixel are JFIF's Y, Cb and Cr, and the file says so. The ICC profile of
/// `icc_size` bytes at `icc` (none where `icc` is null) is embedded where it goes
/// with the picture; where it does not, it is left out, and the message says why.
/// On success the message holds that warning, or an empty string.
/// `unround_bytes_free` frees the bytes.
///
/// # Safety
///
/// `samples` is valid for reads of `height x width x channels` doubles; `icc` is
/// null or valid for reads of `icc_size` bytes; `bytes` and `size` for writes;
/// `message` for writes of `message_size` bytes, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_tiff(
    samples: *const f64,
    height: u64,
    width: u64,
    channels: u64,
    ycbcr: i32,
    icc: *const u8,
    icc_size: u64,
    bytes: *mut *mut u8,
    size: *mut u64,
    message: *mut c_char,
    message_size: usize,
) -> i32 {
    guarded(message, message_size, || {
        if bytes.is_null() || size.is_null() {
            return Err(refuse("the bytes and the size are not null pointers"));
        }
        // SAFETY: the caller promised the samples.
        let values = unsafe { picture(samples, height, width, channels) }?;
        // SAFETY: the caller promised the profile.
        let icc_profile = unsafe { profile(icc, icc_size) }?;
        let written = tiff::float64(
            values,
            narrow(height)?,
            narrow(width)?,
            narrow(channels)?,
            ycbcr != 0,
            icc_profile,
        )
        .map_err(|error| (status_of(&error), error.to_string()))?;
        // SAFETY: the caller promised bytes, size and message valid for writes.
        unsafe { hand_over_written(written, bytes, size, message, message_size) };
        Ok(())
    })
}

/// How the samples of PNG and PNM are made: [`DITHER_NONE`], [`DITHER_BLUE_NOISE`] or
/// [`DITHER_BLUE_NOISE_TOGETHER`] (docs/math.md, 10).
fn dither_of(dither: i32) -> Result<output::Dither, (i32, String)> {
    match dither {
        DITHER_NONE => Ok(output::Dither::None),
        DITHER_BLUE_NOISE => Ok(output::Dither::BlueNoise),
        DITHER_BLUE_NOISE_TOGETHER => Ok(output::Dither::BlueNoiseTogether),
        other => Err((ERROR_OPTIONS, format!("the dither is 0, 1 or 2, not {other}"))),
    }
}

/// Each sample rounded to the nearest level, halves away from 0.
pub const DITHER_NONE: i32 = 0;
/// The ordered dither of the blue-noise matrix, the channels of a pixel apart.
pub const DITHER_BLUE_NOISE: i32 = 1;
/// The ordered dither of the blue-noise matrix, the channels of a pixel together.
pub const DITHER_BLUE_NOISE_TOGETHER: i32 = 2;

/// The bytes of a PNG file of `height x width x channels` samples (1 to 4: grey or
/// RGB, and with 2 or 4 the alpha last), of 8
/// bits, or with `sixteen` nonzero of 16 bits, made as `dither` says
/// ([`DITHER_NONE`], [`DITHER_BLUE_NOISE`] or [`DITHER_BLUE_NOISE_TOGETHER`];
/// docs/cli.md), compressed at zlib's level `compression`, 0 to 9, or -1 for zlib's
/// default; with the ICC profile as `unround_tiff` takes it. `unround_bytes_free` frees
/// the bytes.
///
/// # Safety
///
/// As for `unround_tiff`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_png(
    samples: *const f64,
    height: u64,
    width: u64,
    channels: u64,
    sixteen: i32,
    dither: i32,
    compression: i32,
    icc: *const u8,
    icc_size: u64,
    bytes: *mut *mut u8,
    size: *mut u64,
    message: *mut c_char,
    message_size: usize,
) -> i32 {
    guarded(message, message_size, || {
        if bytes.is_null() || size.is_null() {
            return Err(refuse("the bytes and the size are not null pointers"));
        }
        let level = match compression {
            -1 => None,
            level => Some(
                u8::try_from(level)
                    .ok()
                    .filter(|&level| level <= 9)
                    .ok_or_else(|| (ERROR_OPTIONS, format!("zlib's level is 0 to 9, or -1, not {level}")))?,
            ),
        };
        let depth = output::Depth {
            sixteen: sixteen != 0,
            dither: dither_of(dither)?,
        };
        // SAFETY: the caller promised the samples.
        let values = unsafe { picture(samples, height, width, channels) }?;
        // SAFETY: the caller promised the profile.
        let icc_profile = unsafe { profile(icc, icc_size) }?;
        let written = output::png(
            values,
            narrow(height)?,
            narrow(width)?,
            narrow(channels)?,
            depth,
            level,
            icc_profile,
        )
        .map_err(|error| (status_of(&error), error.to_string()))?;
        // SAFETY: the caller promised bytes, size and message valid for writes.
        unsafe { hand_over_written(written, bytes, size, message, message_size) };
        Ok(())
    })
}

/// The bytes of a PNM file of `height x width x channels` samples (1 or 3), of 8 bits,
/// or with `sixteen` nonzero of 16 bits, made as `dither` says, as for `unround_png`
/// (docs/cli.md). `unround_bytes_free` frees them.
///
/// # Safety
///
/// As for `unround_tiff`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_pnm(
    samples: *const f64,
    height: u64,
    width: u64,
    channels: u64,
    sixteen: i32,
    dither: i32,
    bytes: *mut *mut u8,
    size: *mut u64,
    message: *mut c_char,
    message_size: usize,
) -> i32 {
    guarded(message, message_size, || {
        if bytes.is_null() || size.is_null() {
            return Err(refuse("the bytes and the size are not null pointers"));
        }
        let depth = output::Depth {
            sixteen: sixteen != 0,
            dither: dither_of(dither)?,
        };
        // SAFETY: the caller promised the samples.
        let values = unsafe { picture(samples, height, width, channels) }?;
        let made = output::pnm(values, narrow(height)?, narrow(width)?, narrow(channels)?, depth)
            .map_err(|error| (status_of(&error), error.to_string()))?;
        // SAFETY: the caller promised bytes and size valid for writes.
        unsafe { hand_over(made, bytes, size) };
        Ok(())
    })
}

/// Frees bytes that the interface handed over, where they are not null.
///
/// # Safety
///
/// `bytes` is null, or bytes of `size` that a writer of the interface handed over,
/// not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_bytes_free(bytes: *mut u8, size: u64) {
    if bytes.is_null() {
        return;
    }
    let Ok(length) = usize::try_from(size) else {
        return;
    };
    // SAFETY: the caller promised bytes of this length, made by Box::into_raw, freed once.
    drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(bytes, length)) });
}

/// The picture of `height x width` pixels of `channels` interleaved samples, turned
/// upright as EXIF's `orientation` says (1 to 8, or 0 for none, which is 1), into
/// `turned`; its rows and columns into `turned_height` and `turned_width`, where they
/// are not null. Every sample of the result is one of the picture's, bit for bit.
///
/// # Safety
///
/// `samples` is valid for reads, and `turned` for writes, of
/// `height x width x channels` doubles; `turned_height` and `turned_width` are null
/// or valid for writes; `message` for writes of `message_size` bytes, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_orient(
    samples: *const f64,
    height: u64,
    width: u64,
    channels: u64,
    orientation_value: i32,
    turned: *mut f64,
    turned_height: *mut u64,
    turned_width: *mut u64,
    message: *mut c_char,
    message_size: usize,
) -> i32 {
    guarded(message, message_size, || {
        // SAFETY: the caller promised the samples.
        let values = unsafe { picture(samples, height, width, channels) }?;
        if turned.is_null() {
            return Err(refuse("the turned picture is a null pointer"));
        }
        let result = orientation::oriented(
            values,
            narrow(height)?,
            narrow(width)?,
            narrow(channels)?,
            orientation_value,
        )
        .map_err(|error| (status_of(&error), error.to_string()))?;
        // SAFETY: the caller promised as many writable doubles at turned.
        let target = unsafe { slice::from_raw_parts_mut(turned, result.samples.len()) };
        target.copy_from_slice(&result.samples);
        if !turned_height.is_null() {
            // SAFETY: the caller promised turned_height valid for a write.
            unsafe { turned_height.write(wide(result.height)) };
        }
        if !turned_width.is_null() {
            // SAFETY: the caller promised turned_width valid for a write.
            unsafe { turned_width.write(wide(result.width)) };
        }
        Ok(())
    })
}

/// JFIF's RGB, `height x width x 3` interleaved, of Y, Cb and Cr planes of
/// `height x width` each, one after another (docs/math.md, 8), into `picture`.
///
/// # Safety
///
/// `planes` is valid for reads, and `picture` for writes, of `3 x height x width`
/// doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_to_rgb(planes: *const f64, height: u64, width: u64, picture_out: *mut f64) -> i32 {
    guarded(ptr::null_mut(), 0, || {
        // SAFETY: the caller promised the planes.
        let values = unsafe { picture(planes, height, width, 3) }?;
        if picture_out.is_null() {
            return Err(refuse("the picture is a null pointer"));
        }
        let converted = colour::to_rgb(values, narrow(height)?, narrow(width)?)
            .map_err(|error| (status_of(&error), error.to_string()))?;
        // SAFETY: the caller promised as many writable doubles at picture_out.
        let target = unsafe { slice::from_raw_parts_mut(picture_out, converted.len()) };
        target.copy_from_slice(&converted);
        Ok(())
    })
}

/// JFIF's Y, Cb and Cr planes, one after another, of an RGB picture of
/// `height x width x 3` interleaved, into `planes`.
///
/// # Safety
///
/// `picture` is valid for reads, and `planes` for writes, of `3 x height x width`
/// doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_to_ycbcr(picture_in: *const f64, height: u64, width: u64, planes: *mut f64) -> i32 {
    guarded(ptr::null_mut(), 0, || {
        // SAFETY: the caller promised the picture.
        let values = unsafe { picture(picture_in, height, width, 3) }?;
        if planes.is_null() {
            return Err(refuse("the planes are a null pointer"));
        }
        let converted = colour::to_ycbcr(values, narrow(height)?, narrow(width)?)
            .map_err(|error| (status_of(&error), error.to_string()))?;
        // SAFETY: the caller promised as many writable doubles at planes.
        let target = unsafe { slice::from_raw_parts_mut(planes, converted.len()) };
        target.copy_from_slice(&converted);
        Ok(())
    })
}

/// The command line on `count` arguments, NUL-terminated UTF-8 strings without the
/// program's name: its exit status (docs/cli.md), or 2 for arguments that are not
/// strings, and 70 for an internal error.
///
/// # Safety
///
/// `arguments` is valid for reads of `count` pointers to NUL-terminated strings, or
/// null where `count` is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_main(count: usize, arguments: *const *const c_char) -> i32 {
    let mut status = 70;
    let outcome = guarded(ptr::null_mut(), 0, || {
        // SAFETY: the caller's promise of arguments is passed on.
        let arguments = unsafe { strings(count, arguments) }?;
        status = i32::from(cli::main_with(arguments));
        Ok(())
    });
    match outcome {
        OK => status,
        ERROR_ARGUMENT => 2,
        _ => 70,
    }
}

/// The C layer's `unround_jpegio_abi_version`, through this library.
#[unsafe(no_mangle)]
pub extern "C" fn unround_jpeg_abi_version() -> i32 {
    ffi::unround_jpegio_abi_version()
}

/// The C layer's `unround_jpegio_libjpeg_version`, through this library.
#[unsafe(no_mangle)]
pub extern "C" fn unround_jpeg_libjpeg_version() -> *const c_char {
    ffi::unround_jpegio_libjpeg_version()
}

/// The C layer's `unround_jpegio_libpng_version`, through this library.
#[unsafe(no_mangle)]
pub extern "C" fn unround_jpeg_libpng_version() -> *const c_char {
    ffi::unround_jpegio_libpng_version()
}

/// The C layer's `unround_jpegio_check_icc`, through this library.
///
/// # Safety
///
/// As `unround/jpegio.h` says of `unround_jpegio_check_icc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_jpeg_check_icc(
    profile: *const u8,
    size: u64,
    channels: i32,
    reason: *mut c_char,
    reason_size: usize,
) -> i32 {
    // SAFETY: the caller's promises are those the C layer asks for.
    unsafe { ffi::unround_jpegio_check_icc(profile, size, channels, reason, reason_size) }
}

/// The C layer's `unround_jpegio_read`, through this library.
///
/// # Safety
///
/// As `unround/jpegio.h` says of `unround_jpegio_read`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_jpeg_read(
    data: *const u8,
    size: usize,
    options: *const ffi::Options,
    image: *mut ffi::Image,
    message: *mut c_char,
    message_size: usize,
) -> ffi::Status {
    // SAFETY: the caller's promises are those the C layer asks for.
    unsafe { ffi::unround_jpegio_read(data, size, options, image, message, message_size) }
}

/// The C layer's `unround_jpegio_image_free`, through this library.
///
/// # Safety
///
/// As `unround/jpegio.h` says of `unround_jpegio_image_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_jpeg_image_free(image: *mut ffi::Image) {
    // SAFETY: the caller's promise is the one the C layer asks for.
    unsafe { ffi::unround_jpegio_image_free(image) };
}

/// The C layer's `unround_jpegio_decode_planes`, through this library.
///
/// # Safety
///
/// As `unround/jpegio.h` says of `unround_jpegio_decode_planes`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_jpeg_decode_planes(
    data: *const u8,
    size: usize,
    options: *const ffi::Options,
    planes: *mut ffi::Planes,
    message: *mut c_char,
    message_size: usize,
) -> ffi::Status {
    // SAFETY: the caller's promises are those the C layer asks for.
    unsafe { ffi::unround_jpegio_decode_planes(data, size, options, planes, message, message_size) }
}

/// The C layer's `unround_jpegio_planes_free`, through this library.
///
/// # Safety
///
/// As `unround/jpegio.h` says of `unround_jpegio_planes_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_jpeg_planes_free(planes: *mut ffi::Planes) {
    // SAFETY: the caller's promise is the one the C layer asks for.
    unsafe { ffi::unround_jpegio_planes_free(planes) };
}
