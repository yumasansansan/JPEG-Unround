// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Tests of the C layer as Rust reads it. Files are written by libjpeg's encoder
//! from coefficients chosen here (`unround_test_jpeg`, the tests' support in C),
//! and what a read returns has to be exactly what went in; the planes have to
//! have the sizes of the components. Then the errors, and threads that read at
//! once.

use std::ffi::{c_char, c_int, c_void};
use std::ptr;
use std::slice;
use std::thread;

use jpegio_sys::{ColorSpace, ErrorKind, Image, Options, Planes};

const GRAYSCALE: i32 = 1;
const YCBCR: i32 = 2;
const RGB: i32 = 3;

/// `test_jpeg` of the tests' support in C.
#[repr(C)]
struct TestJpeg {
    width: u32,
    height: u32,
    color_space: i32,
    h_samp_factor: [i32; 4],
    v_samp_factor: [i32; 4],
    quant_table_slot: [i32; 4],
    quant_tables: [[u16; 64]; 4],
    coefficients: [*const i16; 4],
    data_precision: i32,
    progressive: i32,
    sequential_scans: i32,
    arithmetic: i32,
    optimize_coding: i32,
    restart_interval: i32,
    icc_profile: *const u8,
    icc_profile_size: u32,
    app1: *const u8,
    app1_size: u32,
}

unsafe extern "C" {
    fn test_jpeg_components(spec: *const TestJpeg) -> i32;
    fn test_jpeg_blocks_wide(spec: *const TestJpeg, component: i32) -> u32;
    fn test_jpeg_blocks_high(spec: *const TestJpeg, component: i32) -> u32;
    fn test_jpeg_write(
        spec: *const TestJpeg,
        data: *mut *mut u8,
        size: *mut usize,
        message: *mut c_char,
        message_size: usize,
    ) -> c_int;
    fn free(memory: *mut c_void);
}

/// A deterministic stream of numbers, the `SplitMix64` generator.
struct Numbers(u64);

impl Numbers {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number from `low` to `high`, both included.
    fn range(&mut self, low: i32, high: i32) -> i32 {
        let span = u64::try_from(i64::from(high) - i64::from(low) + 1).expect("low <= high");
        let offset = i64::try_from(self.next() % span).expect("the offset fits");
        i32::try_from(i64::from(low) + offset).expect("the number is within the range")
    }
}

/// A file the tests encoded, with the tables and the coefficients that went into it.
struct File {
    quant_table_slot: [i32; 4],
    quant_tables: [[u16; 64]; 4],
    coefficients: Vec<Vec<i16>>,
    data: Vec<u8>,
}

struct Layout {
    color_space: i32,
    h: [i32; 3],
    v: [i32; 3],
}

const LAYOUTS: [Layout; 4] = [
    Layout {
        color_space: GRAYSCALE,
        h: [1, 0, 0],
        v: [1, 0, 0],
    },
    Layout {
        color_space: YCBCR,
        h: [2, 1, 1],
        v: [2, 1, 1],
    },
    Layout {
        color_space: YCBCR,
        h: [2, 1, 1],
        v: [1, 1, 1],
    },
    Layout {
        color_space: RGB,
        h: [1, 1, 1],
        v: [1, 1, 1],
    },
];

fn component_count(spec: &TestJpeg) -> i32 {
    // SAFETY: spec is a valid test_jpeg for the length of the call.
    unsafe { test_jpeg_components(spec) }
}

fn blocks(spec: &TestJpeg, component: i32) -> usize {
    // SAFETY: as above.
    let wide = unsafe { test_jpeg_blocks_wide(spec, component) };
    // SAFETY: as above.
    let high = unsafe { test_jpeg_blocks_high(spec, component) };
    usize::try_from(wide).expect("fits") * usize::try_from(high).expect("fits")
}

fn make_file(layout: &Layout, width: u32, height: u32, seed: u64, progressive: bool, arithmetic: bool) -> File {
    let mut numbers = Numbers(seed);
    let mut spec = TestJpeg {
        width,
        height,
        color_space: layout.color_space,
        h_samp_factor: [layout.h[0], layout.h[1], layout.h[2], 0],
        v_samp_factor: [layout.v[0], layout.v[1], layout.v[2], 0],
        quant_table_slot: [0, 1, 1, 0],
        quant_tables: [[0; 64]; 4],
        coefficients: [ptr::null(); 4],
        data_precision: 0,
        progressive: i32::from(progressive),
        sequential_scans: 0,
        arithmetic: i32::from(arithmetic),
        optimize_coding: 0,
        restart_interval: 0,
        icc_profile: ptr::null(),
        icc_profile_size: 0,
        app1: ptr::null(),
        app1_size: 0,
    };
    for table in &mut spec.quant_tables[..2] {
        for entry in table.iter_mut() {
            *entry = u16::try_from(numbers.range(1, 99)).expect("fits");
        }
    }
    let components = component_count(&spec);
    let mut coefficients = Vec::new();
    for component in 0..components {
        let count = blocks(&spec, component) * 64;
        let values: Vec<i16> = (0..count)
            .map(|index| {
                let value = if index % 64 == 0 {
                    numbers.range(-500, 500)
                } else if index % 64 < 12 && numbers.range(0, 2) == 0 {
                    numbers.range(-60, 60)
                } else {
                    0
                };
                i16::try_from(value).expect("fits")
            })
            .collect();
        coefficients.push(values);
    }
    for (slot, values) in spec.coefficients.iter_mut().zip(&coefficients) {
        *slot = values.as_ptr();
    }
    let mut data: *mut u8 = ptr::null_mut();
    let mut size = 0usize;
    let mut message: [c_char; 256] = [0; 256];
    // SAFETY: spec points to coefficients that live until the call returns, and
    // data, size and message are valid for writes.
    let result = unsafe { test_jpeg_write(&raw const spec, &raw mut data, &raw mut size, message.as_mut_ptr(), 256) };
    assert_eq!(result, 0, "the test encoder failed");
    // SAFETY: test_jpeg_write allocated size bytes at data with malloc().
    let bytes = unsafe { slice::from_raw_parts(data, size) }.to_vec();
    // SAFETY: data came from malloc() and is freed once.
    unsafe { free(data.cast()) };
    File {
        quant_table_slot: spec.quant_table_slot,
        quant_tables: spec.quant_tables,
        coefficients,
        data: bytes,
    }
}

#[test]
fn coefficients_read_back_as_written() {
    let sizes = [(1, 1), (17, 9), (33, 47), (64, 40)];
    let mut seed = 1;
    for layout in &LAYOUTS {
        for (width, height) in sizes {
            for (progressive, arithmetic) in [(false, false), (true, false), (false, true), (true, true)] {
                seed += 1;
                let file = make_file(layout, width, height, seed, progressive, arithmetic);
                let image = Image::read(&file.data, &Options::default()).expect("the file reads");
                assert_eq!((image.width(), image.height()), (width, height));
                assert_eq!(image.progressive(), progressive);
                assert_eq!(image.arithmetic(), arithmetic);
                assert_eq!(image.warnings(), 0);
                assert_eq!(image.warning(), "");
                assert_eq!(image.icc_profile(), None);
                let expected_space = match layout.color_space {
                    GRAYSCALE => ColorSpace::Grayscale,
                    RGB => ColorSpace::Rgb,
                    _ => ColorSpace::YCbCr,
                };
                assert_eq!(image.color_space(), expected_space);
                assert_eq!(image.components().len(), file.coefficients.len());
                for (index, (component, expected)) in image.components().zip(&file.coefficients).enumerate() {
                    let slot = file.quant_table_slot[index];
                    assert_eq!(component.quant_table_slot(), slot);
                    let table = &file.quant_tables[usize::try_from(slot).expect("fits")];
                    assert_eq!(component.quant_table(), table);
                    assert_eq!(component.h_samp_factor(), layout.h[index].max(1));
                    assert_eq!(component.v_samp_factor(), layout.v[index].max(1));
                    assert_eq!(component.coefficients(), expected.as_slice());
                    let last_x = component.width_in_blocks() - 1;
                    let last_y = component.height_in_blocks() - 1;
                    let last = component.block(last_x, last_y).expect("the last block is there");
                    assert_eq!(&last[..], &expected[expected.len() - 64..]);
                    assert!(component.block(last_x + 1, last_y).is_none());
                    assert!(component.block(last_x, last_y + 1).is_none());
                }

                let planes = Planes::decode(&file.data, &Options::default()).expect("the planes decode");
                assert_eq!(planes.warning(), "");
                assert_eq!(planes.planes().len(), image.components().len());
                for (plane, component) in planes.planes().zip(image.components()) {
                    assert_eq!(plane.width(), component.width_in_blocks() * 8);
                    assert_eq!(plane.height(), component.height_in_blocks() * 8);
                    assert!(plane.stride() >= plane.width());
                    let rows = usize::try_from(plane.height()).expect("fits");
                    assert_eq!(
                        plane.samples().len(),
                        usize::try_from(plane.stride()).expect("fits") * rows
                    );
                    assert_eq!(
                        plane.row(plane.height() - 1).map(<[u8]>::len),
                        Some(usize::try_from(plane.width()).expect("fits"))
                    );
                    assert!(plane.row(plane.height()).is_none());
                }
            }
        }
    }
}

#[test]
fn errors_say_what_went_wrong() {
    let garbage = Image::read(b"this is not a JPEG file", &Options::default()).expect_err("not a JPEG file");
    assert_eq!(garbage.kind(), ErrorKind::Decode);
    assert!(garbage.message().contains("Not a JPEG file"), "{garbage}");
    let empty = Planes::decode(&[], &Options::default()).expect_err("nothing to read");
    assert_eq!(empty.kind(), ErrorKind::Decode);

    let file = make_file(&LAYOUTS[1], 40, 30, 99, true, false);
    let limited = Options {
        max_pixels: 40 * 30 - 1,
        ..Options::default()
    };
    assert_eq!(
        Image::read(&file.data, &limited).expect_err("too large").kind(),
        ErrorKind::Limit
    );
    let allowed = Options {
        max_pixels: 40 * 30,
        ..Options::default()
    };
    assert!(Image::read(&file.data, &allowed).is_ok());
    let few_scans = Options {
        max_scans: 1,
        ..Options::default()
    };
    assert_eq!(
        Image::read(&file.data, &few_scans).expect_err("too many scans").kind(),
        ErrorKind::Limit
    );

    let cut = &file.data[..file.data.len() * 3 / 4];
    let warned = Image::read(cut, &Options::default()).expect("a cut file reads with a warning");
    assert!(warned.warnings() > 0);
    assert!(warned.warning().contains("Premature end"), "{}", warned.warning());
    let strict = Options {
        warnings_are_errors: true,
        ..Options::default()
    };
    assert_eq!(
        Image::read(cut, &strict).expect_err("a warning is an error").kind(),
        ErrorKind::Decode
    );
}

#[test]
fn threads_read_at_once() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Image>();
    assert_send_sync::<Planes>();

    let files: Vec<File> = (0..4u64)
        .map(|index| {
            let layout = &LAYOUTS[usize::try_from(index).expect("fits")];
            make_file(
                layout,
                24 + 8 * u32::try_from(index).expect("fits"),
                19,
                1000 + index,
                index % 2 == 1,
                false,
            )
        })
        .collect();
    thread::scope(|scope| {
        for thread_index in 0..8 {
            let files = &files;
            scope.spawn(move || {
                for round in 0..20 {
                    let file = &files[(thread_index + round) % files.len()];
                    let image = Image::read(&file.data, &Options::default()).expect("the file reads");
                    for (component, expected) in image.components().zip(&file.coefficients) {
                        assert_eq!(component.coefficients(), expected.as_slice());
                    }
                    let planes = Planes::decode(&file.data, &Options::default()).expect("the planes decode");
                    assert_eq!(planes.planes().len(), file.coefficients.len());
                }
            });
        }
    });
}

#[test]
fn the_linked_layer_is_the_one_declared() {
    assert_eq!(jpegio_sys::abi_version(), jpegio_sys::ffi::ABI_VERSION);
    assert_eq!(jpegio_sys::libjpeg_version(), "libjpeg-turbo 3.2.0 (libjpeg API 62)");
}
