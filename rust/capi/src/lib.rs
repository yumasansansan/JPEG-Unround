// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The C interface of JPEG-Unround, `include/unround.h`: the library for any language
//! that calls C, the Python package among them.
//!
//! Settings are made from options by the names of the command line (docs/cli.md).
//! A file in memory, or components given as arrays, are reconstructed into a result,
//! whose arrays the caller reads until it frees the result; an observer can follow
//! the solver's records and stop it. The writers of TIFF, PNG and PNM, the turning
//! of a picture upright, JFIF's conversion, the command line, and the C layer's
//! reading of JPEG files are there too.
//!
//! No panic crosses the interface: each function that computes catches one and
//! returns [`ERROR_INTERNAL`]. Errors are said in a message buffer that the caller
//! gives, NUL-terminated and cut at a character's boundary where it is too short.
//! This crate and jpegio-sys hold all the unsafe code of the Rust implementation.

use std::ffi::{CStr, OsString, c_char, c_void};
use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;

use jpeg_unround::decode::{self, Component, Decoded, Input, Settings};
use jpeg_unround::pdhg::Record as SolverRecord;
use jpeg_unround::results::Stop;
use jpeg_unround::{Error, cli, colour, orientation, output, tiff};
use jpegio_sys::{ColorSpace, ffi};

/// The version of this interface: `UNROUND_ABI_VERSION` of the header it matches.
pub const ABI_VERSION: i32 = 2;

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

/// A reconstruction: what the solver made, and what the caller reads of it.
#[derive(Debug)]
pub struct UnroundResult {
    decoded: Decoded,
}

/// Solves an input with the settings and the observer, into `result`.
///
/// # Safety
///
/// `callback` is null, or callable with `user` and a record for the length of the call;
/// `result` is valid for a write.
unsafe fn reconstruct(
    input: &Input,
    settings: &Settings,
    callback: UnroundObserver,
    user: *mut c_void,
    result: *mut *mut UnroundResult,
) -> Result<(), (i32, String)> {
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
    let observer: Option<&mut jpeg_unround::pdhg::Observer<'_>> =
        if callback.is_some() { Some(&mut observe) } else { None };
    let decoded = decode::solve(input, settings, observer).map_err(|error| (status_of(&error), error.to_string()))?;
    let made = Box::into_raw(Box::new(UnroundResult { decoded }));
    // SAFETY: the caller promised result valid for a write.
    unsafe { result.write(made) };
    Ok(())
}

/// Reconstructs a JPEG file of `size` bytes at `data` with the settings, into a
/// result that `unround_result_free` frees. `observer`, where it is not null, is
/// called with `user` and each record after the first, and stops the solver where it
/// returns nonzero.
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
        let input = Input::read(bytes, &settings.read).map_err(|error| (status_of(&error), error.to_string()))?;
        // SAFETY: the caller's promises of observer, user and result are passed on.
        unsafe { reconstruct(&input, settings, observer, user, result) }
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
        // SAFETY: the caller's promises of observer, user and result are passed on.
        unsafe { reconstruct(&input, settings, observer, user, result) }
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

/// The reconstruction of a result, where the pointer is not null.
///
/// # Safety
///
/// `result` is null, or a result not freed yet, which lives for `'a`.
unsafe fn decoded<'a>(result: *const UnroundResult) -> Option<&'a Decoded> {
    // SAFETY: the caller promised null or a live result.
    unsafe { result.as_ref() }.map(|result| &result.decoded)
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
/// greyscale, or RGB. Its sizes are written where the pointers are not null.
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
    let Some(decoded) = (unsafe { decoded(result) }) else {
        return ptr::null();
    };
    // SAFETY: the caller promised each pointer null or valid for a write.
    unsafe { put(height, wide(decoded.height)) };
    // SAFETY: as above.
    unsafe { put(width, wide(decoded.width)) };
    // SAFETY: as above.
    unsafe { put(channels, wide(decoded.channels)) };
    decoded.picture.as_ptr()
}

/// The planes of a result: the canvas cut to the picture, channel by channel (the Y,
/// Cb and Cr of a file in YCbCr), `count` samples in all.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_planes(result: *const UnroundResult, count: *mut u64) -> *const f64 {
    // SAFETY: the caller promised a live result.
    let Some(decoded) = (unsafe { decoded(result) }) else {
        return ptr::null();
    };
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(count, wide(decoded.planes.len())) };
    decoded.planes.as_ptr()
}

/// The color space of a result's file: 1 greyscale, 2 YCbCr, 3 RGB; 0 for a null result.
///
/// # Safety
///
/// `result` is null, or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_color_space(result: *const UnroundResult) -> i32 {
    // SAFETY: the caller promised null or a live result.
    match unsafe { decoded(result) }.map(|decoded| decoded.color_space) {
        Some(ColorSpace::Grayscale) => ffi::GRAYSCALE,
        Some(ColorSpace::YCbCr) => ffi::YCBCR,
        Some(ColorSpace::Rgb) => ffi::RGB,
        None => 0,
    }
}

/// The whole canvas of a result, `channels x height x width` samples, as the solver
/// left it: the picture and what lies beyond it.
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
    let Some(decoded) = (unsafe { decoded(result) }) else {
        return ptr::null();
    };
    // SAFETY: the caller promised each pointer null or valid for a write.
    unsafe { put(channels, wide(decoded.frame.channels().len())) };
    // SAFETY: as above.
    unsafe { put(height, wide(decoded.frame.height())) };
    // SAFETY: as above.
    unsafe { put(width, wide(decoded.frame.width())) };
    decoded.canvas.as_ptr()
}

/// The number of components of a result; 0 for a null result.
///
/// # Safety
///
/// `result` is null, or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_components(result: *const UnroundResult) -> u64 {
    // SAFETY: the caller promised null or a live result.
    unsafe { decoded(result) }.map_or(0, |decoded| wide(decoded.coefficients.len()))
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
    let Some(decoded) = (unsafe { decoded(result) }) else {
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
    let Some(decoded) = (unsafe { decoded(result) }) else {
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
    let Some(decoded) = (unsafe { decoded(result) }) else {
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

/// Whether a solver made a result (1) or not (0: the decoder of the centres), with
/// the iterations it took and why it stopped: 1 a tolerance, 2 the most iterations,
/// 3 the observer, 4 a subgradient of 0.
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
    let Some(solved) = (unsafe { decoded(result) }).and_then(|decoded| decoded.result.as_ref()) else {
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

/// The iterations of the solver's records, `count` of them; null where no solver ran.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_record_iterations(result: *const UnroundResult, count: *mut u64) -> *const u64 {
    // SAFETY: the caller promised a live result.
    let Some(solved) = (unsafe { decoded(result) }).and_then(|decoded| decoded.result.as_ref()) else {
        return ptr::null();
    };
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(count, wide(solved.history.iterations.len())) };
    solved.history.iterations.as_ptr()
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
    let Some(solved) = (unsafe { decoded(result) }).and_then(|decoded| decoded.result.as_ref()) else {
        return ptr::null();
    };
    let history = &solved.history;
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

/// The ICC profile of a result's file, `size` bytes; null where it has none.
///
/// # Safety
///
/// As for `unround_result_picture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_icc_profile(result: *const UnroundResult, size: *mut u64) -> *const u8 {
    // SAFETY: the caller promised a live result.
    let Some(profile) = (unsafe { decoded(result) }).and_then(|decoded| decoded.icc_profile.as_ref()) else {
        return ptr::null();
    };
    // SAFETY: the caller promised a pointer null or valid for a write.
    unsafe { put(size, wide(profile.len())) };
    profile.as_ptr()
}

/// The EXIF orientation of a result's file, 1 to 8, or 0 where it has none.
///
/// # Safety
///
/// `result` is null, or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn unround_result_exif_orientation(result: *const UnroundResult) -> i32 {
    // SAFETY: the caller promised null or a live result.
    unsafe { decoded(result) }.map_or(0, |decoded| decoded.exif_orientation)
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

/// The bytes of a TIFF of binary64 samples, `height x width x channels` (1, or 3
/// interleaved), exactly as they are; with `ycbcr` nonzero, the three samples of a
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

/// The bytes of a PNG file of `height x width x channels` samples (1 or 3), of 8
/// bits, or with `sixteen` nonzero of 16 bits (docs/cli.md), compressed at zlib's
/// level `compression`, 0 to 9, or -1 for zlib's default; with the ICC profile as
/// `unround_tiff` takes it. `unround_bytes_free` frees the bytes.
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
        // SAFETY: the caller promised the samples.
        let values = unsafe { picture(samples, height, width, channels) }?;
        // SAFETY: the caller promised the profile.
        let icc_profile = unsafe { profile(icc, icc_size) }?;
        let written = output::png(
            values,
            narrow(height)?,
            narrow(width)?,
            narrow(channels)?,
            sixteen != 0,
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
/// or with `sixteen` nonzero of 16 bits (docs/cli.md). `unround_bytes_free` frees them.
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
        let made = output::pnm(values, narrow(height)?, narrow(width)?, narrow(channels)?, sixteen != 0)
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
