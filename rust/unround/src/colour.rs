// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The YCbCr of JFIF and its inverse, in binary64 (docs/math.md, 8).
//!
//! The coefficients are the rationals that `K_R = 0.299` and `K_B = 0.114` make,
//! each the double nearest to it (the quotient of two integers that binary64 holds
//! exactly, which one division rounds correctly), and not the six-digit decimals
//! that T.871 prints: those would move a converted block's coefficients by more
//! than 1e-4 of a small step. The operations are in the order docs/math.md gives;
//! each product is fused with the sum that takes it where the processor can.

use crate::dct::multiply_add;
use crate::error::Error;

const CENTRE: f64 = 128.0;

/// `2 (1 - K_R) = 1.402 = 701 / 500`.
pub const RED_FROM_CR: f64 = 701.0 / 500.0;
/// `2 (1 - K_B) = 1.772 = 443 / 250`.
pub const BLUE_FROM_CB: f64 = 443.0 / 250.0;
/// `2 K_B (1 - K_B) / K_G = 25251 / 73375`.
pub const GREEN_FROM_CB: f64 = 25_251.0 / 73_375.0;
/// `2 K_R (1 - K_R) / K_G = 209599 / 293500`.
pub const GREEN_FROM_CR: f64 = 209_599.0 / 293_500.0;
/// `K_R = 299 / 1000`.
pub const Y_FROM_RED: f64 = 299.0 / 1000.0;
/// `K_G = 587 / 1000`.
pub const Y_FROM_GREEN: f64 = 587.0 / 1000.0;
/// `K_B = 114 / 1000`.
pub const Y_FROM_BLUE: f64 = 114.0 / 1000.0;
/// `1 / 1.772 = 250 / 443`.
pub const CB_FROM_BLUE: f64 = 250.0 / 443.0;
/// `1 / 1.402 = 500 / 701`.
pub const CR_FROM_RED: f64 = 500.0 / 701.0;

/// The RGB of JFIF of one pixel's Y, Cb and Cr:
/// `R = Y + c_R (Cr - 128)`, `B = Y + c_B (Cb - 128)`,
/// `G = (Y - c_GB (Cb - 128)) - c_GR (Cr - 128)`.
#[inline]
#[must_use]
pub fn rgb_of(luma: f64, blue_difference: f64, red_difference: f64) -> [f64; 3] {
    let cb = blue_difference - CENTRE;
    let cr = red_difference - CENTRE;
    let red = multiply_add(RED_FROM_CR, cr, luma);
    let blue = multiply_add(BLUE_FROM_CB, cb, luma);
    let green = multiply_add(-GREEN_FROM_CR, cr, multiply_add(-GREEN_FROM_CB, cb, luma));
    [red, green, blue]
}

/// The Y, Cb and Cr of JFIF of one RGB pixel: `Y = (k_R R + k_G G) + k_B B`,
/// `Cb = 128 + k_Cb (B - Y)`, `Cr = 128 + k_Cr (R - Y)`.
#[inline]
#[must_use]
pub fn ycbcr_of(red: f64, green: f64, blue: f64) -> [f64; 3] {
    let luma = multiply_add(Y_FROM_BLUE, blue, multiply_add(Y_FROM_RED, red, Y_FROM_GREEN * green));
    let blue_difference = multiply_add(CB_FROM_BLUE, blue - luma, CENTRE);
    let red_difference = multiply_add(CR_FROM_RED, red - luma, CENTRE);
    [luma, blue_difference, red_difference]
}

/// The RGB of JFIF, `height x width x 3` interleaved, of Y, Cb and Cr planes of
/// `height x width` each, one after another.
///
/// # Errors
///
/// [`Error::Options`] unless there are three planes of that size.
pub fn to_rgb(planes: &[f64], height: usize, width: usize) -> Result<Vec<f64>, Error> {
    let size = height * width;
    if planes.len() != 3 * size {
        return Err(Error::Options(format!(
            "Y, Cb and Cr planes of {height} x {width} are {} samples, not {}",
            3 * size,
            planes.len()
        )));
    }
    let (luma, rest) = planes.split_at(size);
    let (blue, red) = rest.split_at(size);
    let mut picture = Vec::with_capacity(3 * size);
    for ((&y, &cb), &cr) in luma.iter().zip(blue).zip(red) {
        picture.extend_from_slice(&rgb_of(y, cb, cr));
    }
    Ok(picture)
}

/// The Y, Cb and Cr planes of JFIF, one after another, of an RGB picture of
/// `height x width x 3` interleaved.
///
/// # Errors
///
/// [`Error::Options`] unless the picture has that size.
pub fn to_ycbcr(picture: &[f64], height: usize, width: usize) -> Result<Vec<f64>, Error> {
    let size = height * width;
    if picture.len() != 3 * size {
        return Err(Error::Options(format!(
            "an RGB picture of {height} x {width} is {} samples, not {}",
            3 * size,
            picture.len()
        )));
    }
    let mut planes = vec![0.0; 3 * size];
    for (index, &[red, green, blue]) in picture.as_chunks::<3>().0.iter().enumerate() {
        let [y, cb, cr] = ycbcr_of(red, green, blue);
        planes[index] = y;
        planes[size + index] = cb;
        planes[2 * size + index] = cr;
    }
    Ok(planes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::quotient;

    #[test]
    #[expect(clippy::float_cmp, reason = "constants are compared to the last bit")]
    fn the_constants_are_the_nearest_doubles_to_their_rationals() {
        assert_eq!(RED_FROM_CR, quotient(701, 500));
        assert_eq!(BLUE_FROM_CB, quotient(443, 250));
        assert_eq!(GREEN_FROM_CB, quotient(25_251, 73_375));
        assert_eq!(GREEN_FROM_CR, quotient(209_599, 293_500));
        assert_eq!(Y_FROM_RED, 0.299);
        assert_eq!(Y_FROM_GREEN, 0.587);
        assert_eq!(Y_FROM_BLUE, 0.114);
        assert_eq!(CB_FROM_BLUE, quotient(250, 443));
        assert_eq!(CR_FROM_RED, quotient(500, 701));
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "a grey converts exactly")]
    fn a_grey_is_its_luma() {
        // With Cb and Cr at 128, every product is 0 and R = G = B = Y exactly.
        for luma in [0.0, 17.25, 128.0, 254.5, -3.0, 300.0] {
            assert_eq!(rgb_of(luma, 128.0, 128.0), [luma; 3]);
        }
    }
}
