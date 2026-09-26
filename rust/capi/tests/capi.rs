// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The C interface, called as C calls it: its results are the library's to the last
//! bit, its observer stops the solver, and what it refuses it says.

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::slice;

use jpeg_unround::decode::{self, Method, Settings};
use jpeg_unround::{colour, output, pdhg, tiff};
use jpegio_sys::testing::{File, Layout};
use unround_capi as capi;
use unround_capi::{UnroundInput, UnroundRecord, UnroundResult, UnroundSettings};

/// A greyscale file of 20 x 30 samples, and its levels and table.
fn grey_file() -> (Vec<u8>, Vec<i16>, [u16; 64]) {
    let mut table = [0u16; 64];
    for (index, step) in table.iter_mut().enumerate() {
        *step = u16::try_from(8 + index % 8 + index / 8).expect("fits");
    }
    let mut levels = vec![0i16; 3 * 4 * 64];
    let mut state: u32 = 7;
    for (index, level) in levels.iter_mut().enumerate() {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let draw = i16::try_from((state >> 16) % 21).expect("fits") - 10;
        *level = if index % 64 == 0 {
            3 * draw
        } else if index % 64 < 10 {
            draw / 3
        } else {
            0
        };
    }
    let file = File {
        width: 30,
        height: 20,
        layout: Layout::Grayscale,
        quant_tables: vec![table],
        progressive: false,
        arithmetic: false,
    };
    let data = file.write(slice::from_ref(&levels)).expect("a file");
    (data, levels, table)
}

/// Settings of the interface from options, or the message that refuses them.
fn settings(options: &[&str]) -> Result<*mut UnroundSettings, String> {
    let owned: Vec<CString> = options
        .iter()
        .map(|option| CString::new(*option).expect("no NUL"))
        .collect();
    let pointers: Vec<*const c_char> = owned.iter().map(|option| option.as_ptr()).collect();
    let mut message = [c_char::default(); 512];
    // SAFETY: the pointers lead to NUL-terminated strings that live for the call, and
    // the message buffer has 512 bytes.
    let made = unsafe { capi::unround_settings_new(pointers.len(), pointers.as_ptr(), message.as_mut_ptr(), 512) };
    if made.is_null() {
        // SAFETY: the interface wrote a NUL-terminated message.
        let text = unsafe { CStr::from_ptr(message.as_ptr()) };
        Err(text.to_string_lossy().into_owned())
    } else {
        Ok(made)
    }
}

fn free_settings(made: *mut UnroundSettings) {
    // SAFETY: made came from unround_settings_new and is freed once.
    unsafe { capi::unround_settings_free(made) };
}

/// A decoding through the interface: its status, its result, and its message.
fn decode_with(
    data: &[u8],
    made: *const UnroundSettings,
    observer: capi::UnroundObserver,
    user: *mut c_void,
) -> (i32, *mut UnroundResult, String) {
    let mut result: *mut UnroundResult = ptr::null_mut();
    let mut message = [c_char::default(); 512];
    // SAFETY: data is valid for its length, made are live settings, the observer is null
    // or callable with user, and result and message are valid for writes.
    let status = unsafe {
        capi::unround_decode(
            data.as_ptr(),
            data.len(),
            made,
            observer,
            user,
            &raw mut result,
            message.as_mut_ptr(),
            512,
        )
    };
    // SAFETY: the interface wrote a NUL-terminated message, or left the buffer zeroed.
    let text = unsafe { CStr::from_ptr(message.as_ptr()) };
    (status, result, text.to_string_lossy().into_owned())
}

fn free_result(result: *mut UnroundResult) {
    // SAFETY: result came from the interface and is freed once.
    unsafe { capi::unround_result_free(result) };
}

/// The picture of a result, as a slice with its sizes.
fn picture_of(result: *const UnroundResult) -> (Vec<f64>, u64, u64, u64) {
    let (mut height, mut width, mut channels) = (0u64, 0u64, 0u64);
    // SAFETY: result is a live result, and the sizes are valid for writes.
    let samples = unsafe { capi::unround_result_picture(result, &raw mut height, &raw mut width, &raw mut channels) };
    let count = usize::try_from(height * width * channels).expect("fits");
    // SAFETY: the interface promised count samples, valid while the result lives.
    let values = unsafe { slice::from_raw_parts(samples, count) }.to_vec();
    (values, height, width, channels)
}

#[test]
fn the_versions_are_those_of_the_header() {
    assert_eq!(capi::unround_abi_version(), 1);
    // SAFETY: the version is a NUL-terminated string of static storage.
    let version = unsafe { CStr::from_ptr(capi::unround_version()) };
    assert!(version.to_string_lossy().starts_with("unround "));
    assert_eq!(capi::unround_jpeg_abi_version(), 1);
}

#[test]
fn settings_are_made_from_the_options_of_the_command_line() {
    let made = settings(&["--method", "tv", "--mu=0.5", "--iterations", "7"]).expect("settings");
    // The options that give the settings, every value of them: those given, and the
    // defaults of the rest.
    // SAFETY: made are live settings; a null buffer of no bytes is allowed.
    let length = unsafe { capi::unround_settings_options(made, ptr::null_mut(), 0) };
    let mut buffer = vec![c_char::default(); length + 1];
    // SAFETY: the buffer has length + 1 bytes.
    let written = unsafe { capi::unround_settings_options(made, buffer.as_mut_ptr(), buffer.len()) };
    assert_eq!(written, length);
    // SAFETY: the interface wrote a NUL-terminated text.
    let text = unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_str()
        .expect("ASCII")
        .to_owned();
    assert_eq!(text.len(), length);
    let options: Vec<&str> = text.split(' ').collect();
    for option in [
        "--method=tv",
        "--mu=0.5",
        "--iterations=7",
        "--alpha=1.0",
        "--free-radius=255.0",
    ] {
        assert!(options.contains(&option), "{option} in {text}");
    }
    let again = settings(&options).expect("the options give settings");
    // SAFETY: as above, with a buffer too short, which is cut.
    let cut = unsafe { capi::unround_settings_options(again, buffer.as_mut_ptr(), 9) };
    assert_eq!(cut, length);
    // SAFETY: as above.
    assert_eq!(unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_bytes(), b"--method");
    // SAFETY: null settings are allowed.
    let none = unsafe { capi::unround_settings_options(ptr::null(), buffer.as_mut_ptr(), 9) };
    assert_eq!(none, 0);
    free_settings(again);
    free_settings(made);
    let refused = settings(&["--alpha", "0", "--format", "pnm", "--bogus"]).expect_err("refused");
    for fragment in [
        "--alpha is positive",
        "--format is an option of the command line",
        "unknown option --bogus",
    ] {
        assert!(refused.contains(fragment), "{fragment} in {refused}");
    }
    let defaults = settings(&[]).expect("settings");
    free_settings(defaults);
}

#[test]
fn a_file_decodes_as_the_library_decodes_it() {
    let (data, _, _) = grey_file();
    for (method, options) in [
        (Method::Mmse, vec!["--method", "mmse"]),
        (Method::Tv, vec!["--method", "tv", "--iterations", "30"]),
        (Method::Tgv, vec!["--method", "tgv", "--iterations", "30"]),
    ] {
        let made = settings(&options).expect("settings");
        let (status, result, message) = decode_with(&data, made, None, ptr::null_mut());
        assert_eq!(status, capi::OK, "{message}");
        let (values, height, width, channels) = picture_of(result);
        assert_eq!((height, width, channels), (20, 30, 1));
        let library = Settings {
            method,
            pdhg: pdhg::Options {
                iterations: Some(30),
                ..pdhg::Options::default()
            },
            ..Settings::default()
        };
        let expected = decode::decode(&data, &library, None).expect("the file decodes");
        assert_eq!(values, expected.picture);
        // SAFETY: result is a live result.
        assert_eq!(unsafe { capi::unround_result_components(result) }, 1);
        let mut count = 0u64;
        // SAFETY: as above, and count is valid for a write.
        let coefficients = unsafe { capi::unround_result_coefficients(result, 0, &raw mut count) };
        // SAFETY: the interface promised count coefficients.
        let coefficients = unsafe { slice::from_raw_parts(coefficients, usize::try_from(count).expect("fits")) };
        assert_eq!(coefficients, expected.coefficients[0].as_slice());
        // SAFETY: as above.
        let beyond = unsafe { capi::unround_result_coefficients(result, 1, &raw mut count) };
        assert!(beyond.is_null());
        let (mut iterations, mut stop) = (0u64, 0i32);
        // SAFETY: as above, and the pointers are valid for writes.
        let solved = unsafe { capi::unround_result_solver(result, &raw mut iterations, &raw mut stop) };
        assert_eq!(solved, i32::from(method != Method::Mmse));
        if method != Method::Mmse {
            assert_eq!((iterations, stop), (30, 2));
        }
        free_result(result);
        free_settings(made);
    }
}

#[test]
fn components_given_as_arrays_solve_as_their_file() {
    let (data, levels, table) = grey_file();
    let made = settings(&["--method", "tv", "--iterations", "20"]).expect("settings");
    let component = capi::UnroundComponent {
        levels: levels.as_ptr(),
        rows: 3,
        columns: 4,
        h_samp_factor: 1,
        v_samp_factor: 1,
        quant_table: table,
    };
    let input = UnroundInput {
        height: 20,
        width: 30,
        color_space: 1,
        count: 1,
        components: &raw const component,
    };
    let mut result: *mut UnroundResult = ptr::null_mut();
    // SAFETY: the input's arrays are valid for their sizes, made are live settings, and
    // result is valid for a write.
    let status = unsafe {
        capi::unround_solve(
            &raw const input,
            made,
            None,
            ptr::null_mut(),
            &raw mut result,
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(status, capi::OK);
    let (from_arrays, ..) = picture_of(result);
    let (status, from_file, message) = decode_with(&data, made, None, ptr::null_mut());
    assert_eq!(status, capi::OK, "{message}");
    assert_eq!(from_arrays, picture_of(from_file).0);
    free_result(result);
    free_result(from_file);
    free_settings(made);
}

/// Counts the records it is given, and stops the solver at the second.
unsafe extern "C" fn stop_at_the_second(user: *mut c_void, record: *const UnroundRecord) -> i32 {
    // SAFETY: the test gives a pointer to a u64 as user.
    let seen = unsafe { &mut *user.cast::<u64>() };
    // SAFETY: the interface gives a record valid for the call.
    let record = unsafe { &*record };
    assert_eq!((record.channels, record.height, record.width), (1, 24, 32));
    // SAFETY: the canvas has channels x height x width samples, valid for the call.
    let canvas = unsafe { slice::from_raw_parts(record.canvas, 24 * 32) };
    assert!(canvas.iter().all(|sample| sample.is_finite()));
    *seen += 1;
    i32::from(*seen == 2)
}

#[test]
fn an_observer_follows_the_records_and_stops_the_solver() {
    let (data, _, _) = grey_file();
    let made = settings(&["--method", "tgv", "--iterations", "100", "--record-every", "5"]).expect("settings");
    let mut seen = 0u64;
    let (status, result, message) = decode_with(&data, made, Some(stop_at_the_second), (&raw mut seen).cast());
    assert_eq!(status, capi::OK, "{message}");
    assert_eq!(seen, 2);
    let (mut iterations, mut stop) = (0u64, 0i32);
    // SAFETY: result is a live result, and the pointers are valid for writes.
    unsafe { capi::unround_result_solver(result, &raw mut iterations, &raw mut stop) };
    assert_eq!((iterations, stop), (10, 3));
    let mut count = 0u64;
    // SAFETY: as above.
    let recorded = unsafe { capi::unround_result_record_iterations(result, &raw mut count) };
    // SAFETY: the interface promised count iterations.
    let recorded = unsafe { slice::from_raw_parts(recorded, usize::try_from(count).expect("fits")) };
    assert_eq!(recorded, [0, 5, 10]);
    // SAFETY: as above.
    let scaling = unsafe { capi::unround_result_record_values(result, 3, &raw mut count) };
    assert!(!scaling.is_null() && count == 3);
    // SAFETY: as above; a which that is not there gives null.
    assert!(unsafe { capi::unround_result_record_values(result, 9, &raw mut count) }.is_null());
    free_result(result);
    free_settings(made);
}

#[test]
fn what_is_refused_is_said() {
    let made = settings(&[]).expect("settings");
    let (status, result, message) = decode_with(b"not a JPEG file", made, None, ptr::null_mut());
    assert_eq!(status, capi::ERROR_READ);
    assert!(result.is_null());
    assert!(message.contains("Not a JPEG file"), "{message}");
    // SAFETY: null pointers are what is being refused; nothing is read or written.
    let status = unsafe {
        capi::unround_decode(
            ptr::null(),
            0,
            made,
            None,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(status, capi::ERROR_ARGUMENT);
    // A message cut short keeps whole characters, and its NUL.
    let mut short = [c_char::try_from(b'x').expect("an ASCII letter fits"); 5];
    // SAFETY: the buffer has 5 bytes.
    let status = unsafe {
        capi::unround_decode(
            ptr::null(),
            0,
            made,
            None,
            ptr::null_mut(),
            ptr::null_mut(),
            short.as_mut_ptr(),
            5,
        )
    };
    assert_eq!(status, capi::ERROR_ARGUMENT);
    assert_eq!(short[4], 0);
    free_settings(made);
    // SAFETY: freeing null does nothing.
    unsafe { capi::unround_result_free(ptr::null_mut()) };
}

#[test]
fn the_writers_and_the_conversion_are_the_library_s() {
    let samples: Vec<f64> = (0..2 * 3 * 3).map(|index| f64::from(index) * 17.25 - 20.0).collect();
    for ycbcr in [0, 1] {
        let (mut bytes, mut size) = (ptr::null_mut::<u8>(), 0u64);
        // SAFETY: samples has 2 x 3 x 3 doubles, and bytes and size are valid for writes.
        let status = unsafe {
            capi::unround_tiff(
                samples.as_ptr(),
                2,
                3,
                3,
                ycbcr,
                &raw mut bytes,
                &raw mut size,
                ptr::null_mut(),
                0,
            )
        };
        assert_eq!(status, capi::OK);
        // SAFETY: the interface handed over size bytes.
        let written = unsafe { slice::from_raw_parts(bytes, usize::try_from(size).expect("fits")) }.to_vec();
        assert_eq!(written, tiff::float64(&samples, 2, 3, 3, ycbcr == 1).expect("a file"));
        // SAFETY: the bytes are freed once, with their size.
        unsafe { capi::unround_bytes_free(bytes, size) };
    }
    let (mut bytes, mut size) = (ptr::null_mut::<u8>(), 0u64);
    // SAFETY: as above.
    let status = unsafe {
        capi::unround_pnm(
            samples.as_ptr(),
            2,
            3,
            3,
            1,
            &raw mut bytes,
            &raw mut size,
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(status, capi::OK);
    // SAFETY: as above.
    let written = unsafe { slice::from_raw_parts(bytes, usize::try_from(size).expect("fits")) }.to_vec();
    assert_eq!(written, output::pnm(&samples, 2, 3, 3, true).expect("a file"));
    // SAFETY: as above.
    unsafe { capi::unround_bytes_free(bytes, size) };
    let mut rgb = vec![0.0; samples.len()];
    // SAFETY: both arrays have 3 x 2 x 3 doubles.
    let status = unsafe { capi::unround_to_rgb(samples.as_ptr(), 2, 3, rgb.as_mut_ptr()) };
    assert_eq!(status, capi::OK);
    assert_eq!(rgb, colour::to_rgb(&samples, 2, 3).expect("planes"));
    let mut back = vec![0.0; samples.len()];
    // SAFETY: as above.
    let status = unsafe { capi::unround_to_ycbcr(rgb.as_ptr(), 2, 3, back.as_mut_ptr()) };
    assert_eq!(status, capi::OK);
    assert_eq!(back, colour::to_ycbcr(&rgb, 2, 3).expect("a picture"));
}

#[test]
fn the_command_line_runs_through_the_interface() {
    let arguments = [
        CString::new("--alpha").expect("no NUL"),
        CString::new("0").expect("no NUL"),
    ];
    let pointers: Vec<*const c_char> = arguments.iter().map(|argument| argument.as_ptr()).collect();
    // SAFETY: the pointers lead to NUL-terminated strings that live for the call.
    assert_eq!(unsafe { capi::unround_main(pointers.len(), pointers.as_ptr()) }, 2);
}
