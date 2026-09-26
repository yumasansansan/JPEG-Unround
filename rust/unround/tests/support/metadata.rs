// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! What a JPEG file carries besides its picture, for tests: ICC profiles and EXIF
//! segments with an orientation.

/// A well-formed ICC profile of version 2 and the size given (at least 148 bytes),
/// a monitor's, of the data color space given (`b"RGB "` or `b"GRAY"`): its
/// header, one tag, and the tag's data, bytes that count up from `seed`.
#[must_use]
pub fn profile(size: usize, space: [u8; 4], seed: u8) -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..size)
        .map(|index| seed.wrapping_add(u8::try_from(index % 251).expect("fits")))
        .collect();
    bytes[..132].fill(0);
    let put = |bytes: &mut Vec<u8>, at: usize, value: u32| bytes[at..at + 4].copy_from_slice(&value.to_be_bytes());
    put(&mut bytes, 0, u32::try_from(size).expect("a small profile"));
    bytes[8] = 2;
    bytes[12..16].copy_from_slice(b"mntr");
    bytes[16..20].copy_from_slice(&space);
    bytes[20..24].copy_from_slice(b"XYZ ");
    bytes[36..40].copy_from_slice(b"acsp");
    put(&mut bytes, 68, 0x0000_F6D6); // the illuminant D50
    put(&mut bytes, 72, 0x0001_0000);
    put(&mut bytes, 76, 0x0000_D32D);
    put(&mut bytes, 128, 1);
    bytes[132..136].copy_from_slice(b"desc");
    put(&mut bytes, 136, 144);
    put(&mut bytes, 140, u32::try_from(size - 144).expect("a small profile"));
    bytes
}

/// The payload of an EXIF APP1 marker whose first IFD holds the orientation given,
/// little-endian or big-endian.
#[must_use]
pub fn exif(orientation: u16, little: bool) -> Vec<u8> {
    let short = |value: u16| {
        if little {
            value.to_le_bytes().to_vec()
        } else {
            value.to_be_bytes().to_vec()
        }
    };
    let long = |value: u32| {
        if little {
            value.to_le_bytes().to_vec()
        } else {
            value.to_be_bytes().to_vec()
        }
    };
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(if little { b"II" } else { b"MM" });
    payload.extend(short(42));
    payload.extend(long(8));
    payload.extend(short(1));
    payload.extend(short(0x0112)); // Orientation
    payload.extend(short(3)); // SHORT
    payload.extend(long(1));
    payload.extend(short(orientation));
    payload.extend(short(0));
    payload.extend(long(0)); // no next IFD
    payload
}
