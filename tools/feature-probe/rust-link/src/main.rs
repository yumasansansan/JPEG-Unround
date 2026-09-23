// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Calls the C23 static library, once where it succeeds and once where it
//! reports an error through longjmp inside C.

unsafe extern "C" {
    fn unround_probe_checked_add(a: i32, b: i32, out: *mut i32) -> i32;
}

fn main() {
    let mut sum = 0i32;
    // SAFETY: the function writes one i32 through a pointer to a live local.
    let ok = unsafe { unround_probe_checked_add(2, 3, &mut sum) };
    assert_eq!((ok, sum), (0, 5));
    // SAFETY: as above; the overflow is reported, and the jump stays in C.
    let overflow = unsafe { unround_probe_checked_add(i32::MAX, 1, &mut sum) };
    assert_eq!(overflow, -1);
    println!("rust-link: pass (C23 static library, linked by LLD)");
}
