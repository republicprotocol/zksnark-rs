use super::*;

extern crate itertools;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;

pub struct KeccakInternal {
    pub a: [Word64; 25],
    pub offset: usize,
    pub rate: usize,
    pub delim: u8,
}

impl KeccakInternal {
    pub fn a_bytes(&self) -> [Word8; 200] {
        let mut arr: [Word8; 200] = [Word8::default(); 200];
        self.a
            .iter()
            .flat_map(|wrd64| wrd64.iter())
            .enumerate()
            .for_each(|(i, &wrd8)| arr[i] = wrd8);
        arr
    }
}

/// ## Usage Details:
///
/// IMPORTANT:
///     - Only input either 0 or 1 as inputs to `Word8` wires, or just use
///       the provided constructor in `Circuit`!
///     - Word8 is stored in the array as if it was little-endian: For example
///       lets say you input the number 0x4B (ends in: 0100 1011) into the Word8
///       placeholder then it would be stored as: [1,1,0,1,0,0,1,0]
///
/// ### Iterator and FromIterator
///
/// Iterator: You get the bits starting from the least significant bit, which
/// you can think of as the first wire on the left.
///
/// FromIterator: bits input into from right to left; least significant first to
/// most significant bit.
///
pub type Word8 = [WireId; 8];

pub fn setout(src: &[Word8], dst: &mut [Word8], len: usize) {
    dst[..len].copy_from_slice(&src[..len]);
}

/// NOTE: if you don't give enough bits the extra bits will be filled in with
/// WireId::default() which is the zero wire. On the other hand if you have too
/// many bits you will get a runtime panic! (this is so you realize you have
/// made a grave mistake). The only way to get the panic that you have
/// given too many bits is to collect a word8 directly.
///
pub fn to_word8(input: impl Iterator<Item = WireId>) -> Word8 {
    let mut arr: Word8 = Word8::default();
    (0..8).zip_longest(input).for_each(|x| match x {
        Both(i, num) => arr[i] = num,
        Left(_) => panic!("to_word8: Word8 cannot be constructed from less than 8 WireId"),
        Right(_) => panic!("to_word8: Word8 cannot be constructed from more than 8 WireId"),
    });
    arr
}

/// ## Usage Details:
///
/// IMPORTANT:
///     - Only input either 0 or 1 as inputs to `Word64` wires, or just use
///       the provided constructor in `Circuit`!
///     - Word64 is stored as if it was little-endian: For example lets say you
///     input the first byte as 0x4F (0100 1111) and second byte as 0x4B
///     (01001011) into the Word64 placeholder then it would be stored as:
///     [1,1,1,1,0,0,1,0, 1,1,0,1,0,0,1,0, (48 zeros)]
///
/// `Word64` is really just a placeholder for a u64. It does not store a u64
/// number, but can be assigned a u64 number before evaluation. Still this only
/// associates the bits of a u64 number with the `WireId`s in the `Word64`.
///
/// ### Iterator and FromIterator
///
/// Iterator: You get the bytes starting from the least significant byte, which
/// you can think of as the first Word8 on the left.
///
/// FromIterator: bytes input into from right to left; least significant first to
/// most significant byte.
///
pub type Word64 = [Word8; 8];

/// Rotates a Word64's bits by moving bit a position `i` into position `i+by`
/// modulo the lane size. The least significant bit is where i = 0 and the most
/// significant bit is where i = 63.
///
/// Example of left rotation: (u64 truncated to u8)
///     start:   0100 0101
///     becomes: 1000 1010
///
/// NOTE: in memory this would look like [1,0,1,0,0,0,1,0] => [0,1,0,1,0,0,0,1]
///
pub fn rotate_word64_left(input: Word64, by: usize) -> Word64 {
    let mut wrd64 = Word64::default();
    input
        .iter()
        .flat_map(|x| x.iter())
        .cycle()
        .skip(64 - (by % 64))
        .take(64)
        .zip(iproduct!(0..8, 0..8))
        .for_each(|(&wire_id, (i, j))| wrd64[i][j] = wire_id);
    wrd64
}

/// Rotates a Word64's bits by moving bit a position `i` into position `i-by`
/// modulo the lane size.
///
/// Example of left rotation: (u64 truncated to u8)
///     start:   0100 0101
///     becomes: 1010 0010
///
/// NOTE: in memory this would look like [1,0,1,0,0,0,1,0] => [0,1,0,0,0,1,0,1]
///
pub fn rotate_word64_right(input: Word64, by: usize) -> Word64 {
    let mut wrd64 = Word64::default();
    input
        .iter()
        .flat_map(|x| x.iter())
        .cycle()
        .skip(by % 64)
        .take(64)
        .zip(iproduct!(0..8, 0..8))
        .for_each(|(&wire_id, (i, j))| wrd64[i][j] = wire_id);
    wrd64
}

/// NOTE: if you don't give enough bits the extra bits will be filled in with
/// WireId::default() which is the zero wire. On the other hand if you have too
/// many bits you will get a runtime panic! (this is so you realize you have
/// made a grave mistake). The only way to get the panic that you have
/// given too many bits is to collect a word64 directly.
///
pub fn to_word64(input: impl Iterator<Item = WireId>) -> Word64 {
    let mut arr: Word64 = Word64::default();
    input
        .chunks(8)
        .into_iter()
        .map(|chunk| to_word8(chunk))
        .zip_longest(0..8)
        .for_each(|x| match x {
            Both(num, i) => arr[i] = num,
            Right(i) => panic!("to_word64: Word64 cannot be constructed from less than 64 WireId"),
            Left(_) => panic!("to_word64: Word64 cannot be constructed from more than 64 WireId"),
        });
    arr
}

// pub enum Binary {
//     One,
//     Zero,
// }

// impl Binary {
//     pub fn from_u64(num: u64) -> [Binary; 64] {
//         let mut n = num;
//         let mut output = [Zero; 64];
//         (0..64).for_each(|i| {
//             if n % 2 != 0 {
//                 output[i] = One;
//             }
//             n = n >> 1;
//         });
//         output
//     }
// }

pub const RC: [u64; 24] = [
    1u64,
    0x8082u64,
    0x800000000000808au64,
    0x8000000080008000u64,
    0x808bu64,
    0x80000001u64,
    0x8000000080008081u64,
    0x8000000000008009u64,
    0x8au64,
    0x88u64,
    0x80008009u64,
    0x8000000au64,
    0x8000808bu64,
    0x800000000000008bu64,
    0x8000000000008089u64,
    0x8000000000008003u64,
    0x8000000000008002u64,
    0x8000000000000080u64,
    0x800au64,
    0x800000008000000au64,
    0x8000000080008081u64,
    0x8000000000008080u64,
    0x80000001u64,
    0x8000000080008008u64,
];

pub const RHO: [usize; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];

pub const PI: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

/// This is copied from the std library because it was marked as a nightly only
/// feature, not because it is unstable, but because they were not sure if it
/// should be added and named where it was.
pub fn to_ne_u8(num: u64) -> [u8; std::mem::size_of::<u64>()] {
    unsafe { std::mem::transmute(num) }
}
pub fn from_ne_u64(bytes: [u8; std::mem::size_of::<u64>()]) -> u64 {
    unsafe { std::mem::transmute(bytes) }
}
// pub fn to_le_u8(num: u64) -> [u8; std::mem::size_of::<u64>()] {
//     to_ne_u8(num.to_le())
// }
// pub fn from_le_u64(bytes: [u8; std::mem::size_of::<u64>()]) -> u64 {
//     from_ne_u64(bytes).to_le()
// }

// pub fn flatten_word64(input: Word64) -> [WireId; 64] {
//     let mut wires_u64: [WireId; 64] = [WireId::default(); 64];
//     input
//         .iter()
//         .enumerate()
//         .for_each(|(i, &wrd8): (usize, &Word8)| {
//             wrd8.iter()
//                 .enumerate()
//                 .for_each(|(j, &wire_id)| wires_u64[(i * 8) + j] = wire_id)
//         });
//     wires_u64
// }

#[cfg(test)]
mod tests {
    use super::*;

    extern crate quickcheck;
    use self::quickcheck::quickcheck;

    quickcheck! {
        fn rotate_inverse_prop(rotate_by: usize) -> bool {
            let word64 = to_word64((0..64).map(WireId));
            word64 == rotate_word64_right(rotate_word64_left(word64, rotate_by), rotate_by)
        }
        fn rotate_mod(by: usize) -> bool {
            let word64 = to_word64((0..64).map(WireId));
            rotate_word64_left(word64, by + 64) == rotate_word64_left(word64, by)
                &&
            rotate_word64_right(word64, by + 64) == rotate_word64_right(word64, by)
        }
    }

    #[test]
    fn rotate_single_test() {
        let a_wrd64: Word64 = to_word64((0..64).map(WireId));
        let b_wrd64: Word64 = to_word64((63..64).chain(0..63).map(WireId));
        assert_eq!(b_wrd64, rotate_word64_left(a_wrd64, 1));
    }

    // #[test]
    // fn to_from_little_endian() {
    //     assert_eq!(
    //         to_le_u8(0b1000_0000_0000_0000),
    //         [0, 0b1000_0000, 0, 0, 0, 0, 0, 0]
    //     );
    //     assert_eq!(
    //         from_le_u64([0, 0b1000_0000, 0, 0, 0, 0, 0, 0]),
    //         0b1000_0000_0000_0000
    //     );
    // }

    // /// NOTE: The KeccakMatrix tests actually tests all `FromIterator` in `Word64`
    // /// because of the recurse definitions of `FromIterator`
    // #[test]
    // fn keccakmatrix_iterators() {
    //     let state: KeccakMatrix = (0..1600).map(WireId).collect();
    //     assert_eq!(state[0][0][0][0], WireId(0));
    //     assert_eq!(state[0][0][1][0], WireId(8));
    //     assert_eq!(state[2][3][5][0], WireId(872));
    //     assert_eq!(state[4][4][6][0], WireId(1584));
    //     assert_eq!(state[4][4][7][7], WireId(1599));

    //     let mut iter = state.into_iter();

    //     assert_eq!(
    //         iter.next().unwrap()[0],
    //         (0..64).map(|x| WireId(x)).collect()
    //     );
    // }

    // #[test]
    // #[should_panic(
    //     expected = "FromIterator: KeccakMatrix cannot be constructed from more than 5 KeccakRow"
    // )]
    // fn keccakmatrix_overflow_fromiterator() {
    //     let _state: KeccakMatrix = (0..1920).map(|x| WireId(x)).collect();
    // }

    // #[test]
    // #[should_panic(
    //     expected = "FromIterator: KeccakMatrix cannot be constructed from less than 5 KeccakRow"
    // )]
    // fn keccakmatrix_underflow_fromiterator() {
    //     let _state: KeccakMatrix = (0..1280).map(|x| WireId(x)).collect();
    // }

    // #[test]
    // #[should_panic(
    //     expected = "FromIterator: KeccakRow cannot be constructed from more than 5 Word64"
    // )]
    // fn keccakrow_overflow_fromiterator() {
    //     let _state: KeccakRow = (0..384).map(|x| WireId(x)).collect();
    // }

    // #[test]
    // #[should_panic(
    //     expected = "FromIterator: KeccakRow cannot be constructed from less than 5 Word64"
    // )]
    // fn keccakrow_underflow_fromiterator() {
    //     let _state: KeccakRow = (0..256).map(|x| WireId(x)).collect();
    // }

    // #[test]
    // #[should_panic(expected = "FromIterator: Word64 cannot be constructed from more than 8 Word8")]
    // fn word64_overflow_fromiterator() {
    //     let _state: Word64 = (0..70).map(|x| WireId(x)).collect();
    // }
}
