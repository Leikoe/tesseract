//! Exact host-side reference semantics for checkpoint low-precision formats.
//!
//! CUDA kernels have independent implementations and are differentially
//! checked against these functions. Keeping the oracle free of CUDA and model
//! code makes it usable by loaders, property tests, and future backends.

/// Decodes one finite E2M1 FP4 nibble.
pub(crate) fn decode_e2m1(nibble: u8) -> f32 {
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = MAGNITUDES[(nibble & 0x07) as usize];
    if nibble & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Decodes NVIDIA's finite-only E4M3 FP8 representation.
///
/// Exponent 15 remains finite except for mantissa 7, which is NaN. There are
/// no infinities. Signed zero is preserved.
pub(crate) fn decode_e4m3fn(bits: u8) -> f32 {
    let sign = bits & 0x80 != 0;
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    let magnitude = match (exponent, mantissa) {
        (0, mantissa) => f32::from(mantissa) * 2.0f32.powi(-9),
        (15, 7) => f32::NAN,
        (exponent, mantissa) => {
            (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
        }
    };
    if sign { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn decodes_all_fp4_values_exactly() {
        let expected: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for (bits, expected) in expected.into_iter().enumerate() {
            assert_eq!(decode_e2m1(bits as u8).to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn decodes_e4m3fn_boundaries() {
        assert_eq!(decode_e4m3fn(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(decode_e4m3fn(0x80).to_bits(), (-0.0f32).to_bits());
        assert_eq!(decode_e4m3fn(0x01), 2.0f32.powi(-9));
        assert_eq!(decode_e4m3fn(0x38), 1.0);
        assert_eq!(decode_e4m3fn(0x7e), 448.0);
        assert!(decode_e4m3fn(0x7f).is_nan());
        assert!(decode_e4m3fn(0xff).is_nan());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn fp4_sign_bit_only_changes_sign(bits in 0u8..8) {
            let positive = decode_e2m1(bits);
            let negative = decode_e2m1(bits | 8);
            prop_assert_eq!(negative.to_bits(), (-positive).to_bits());
        }

        #[test]
        fn finite_fp8_sign_bit_only_changes_sign(bits in 0u8..0x7fu8) {
            let positive = decode_e4m3fn(bits);
            let negative = decode_e4m3fn(bits | 0x80);
            prop_assert!(positive.is_finite());
            prop_assert_eq!(negative.to_bits(), (-positive).to_bits());
        }
    }
}
