use super::*;
use std::iter::FromIterator;
use std::ops::Deref;
extern crate itertools;

/// The idea here is to have each `WireId` have an input of either 0 or 1 and
/// then group those wires together to form a single `u64` number where `wire1`
/// corresponds to the first bit (also called the right most bit) of the `u64`,
/// then `wire2` the second bit and so on.  
///
/// TODO turn Word64 into a struct and implement iterator for it, there is no
/// point in ever having access to the entire array.
#[derive(Clone, Copy)]
pub struct Word64([WireId; 64]);

/// Rotates a Word64's bits by moving bit a position `i` into position `i+n`
/// module the lane size.
///
pub fn u64_rot_left(input: Word64, by: usize) -> Word64 {
    unimplemented!();
    // input.into_iter().cycle().skip(by).take(64).collect()
    //
    // let tail = input[63];
    // let mut word64: Word64 = Word64::default();
    // (1..63).rev().for_each(|x| word64[x] = input[x - 1]);
    // word64[0] = tail;
    // word64
}

impl Default for Word64 {
    fn default() -> Word64 {
        Word64([WireId::default(); 64])
    }
}

impl Deref for Word64 {
    type Target = [WireId; 64];

    fn deref(&self) -> &[WireId; 64] {
        &self.0
    }
}

impl FromIterator<WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        let mut arr: [WireId; 64] = [WireId::default(); 64];
        (0..64)
            .zip(iter.into_iter())
            .for_each(|(i, num)| arr[i] = num);
        Word64(arr)
    }
}

/// It is a 5 by 5 matrix used as the internal state of Keccak hash
/// function or other matrix need by the Keccak hash function.
#[derive(Default)]
pub struct KeccakMatrix<T>([[T; 5]; 5]);

impl<T> Deref for KeccakMatrix<T> {
    type Target = [[T; 5]; 5];

    fn deref(&self) -> &[[T; 5]; 5] {
        &self.0
    }
}

/// Fills the matrix from left to right then top to bottom.
///
/// (0..24)
///
/// 0  | 1  | 2  | 3  | 4
/// 5  | 6  | 7  | 8  | 9
/// 10 | 11 | 12 | 13 | 14
/// 15 | 16 | 17 | 18 | 19
/// 20 | 21 | 22 | 23 | 24
impl<T> FromIterator<T> for KeccakMatrix<T>
where
    T: Default + Copy,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut arr = [[T::default(); 5]; 5];
        iproduct!(0..5, 0..5)
            .zip(iter.into_iter())
            .for_each(|((x, y), num)| arr[x][y] = num);
        KeccakMatrix(arr)
    }
}

const rotation_offset: KeccakMatrix<u64> = KeccakMatrix([
    [0, 36, 3, 18, 41],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]);

const round_constants: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];
