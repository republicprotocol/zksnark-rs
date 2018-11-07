use super::*;
use std::iter::FromIterator;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;

extern crate itertools;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;

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
pub fn left_rotate(input: Word64, by: usize) -> Word64 {
    input.iter().cycle().skip(by).collect()
}

/// Rotates a Word64's bits by moving bit a position `i` into position `i-n`
/// module the lane size.
///
pub fn right_rotate(input: Word64, by: usize) -> Word64 {
    input.iter().cycle().skip(64 - by).collect()
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

impl DerefMut for Word64 {
    fn deref_mut(&mut self) -> &mut [WireId; 64] {
        &mut self.0
    }
}

impl FromIterator<WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        let mut arr: Word64;
        unsafe {
            arr = mem::uninitialized();
            (0..64).zip_longest(iter.into_iter()).for_each(|x| match x {
                Both(i, num) => arr[i] = num,
                Left(_) => {
                    panic!("FromIterator: Word64 cannot be constructed from more than 64 WireId")
                }
                Right(_) => {
                    panic!("FromIterator: Word64 cannot be constructed from less than 64 WireId")
                }
            });
        }
        arr
    }
}

impl<'a> FromIterator<&'a WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = &'a WireId>>(iter: I) -> Self {
        iter.into_iter().collect()
    }
}

/// It is a 5 by 5 matrix used as the internal state of Keccak hash
/// function or other matrix need by the Keccak hash function.
#[derive(Clone, Copy)]
pub struct KeccakRow([Word64; 5]);

impl Default for KeccakRow {
    fn default() -> KeccakRow {
        KeccakRow([Word64::default(); 5])
    }
}

impl Deref for KeccakRow {
    type Target = [Word64; 5];

    fn deref(&self) -> &[Word64; 5] {
        &self.0
    }
}

impl DerefMut for KeccakRow {
    fn deref_mut(&mut self) -> &mut [Word64; 5] {
        &mut self.0
    }
}

impl FromIterator<WireId> for KeccakRow {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        iter.into_iter()
            .chunks(64)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Word64>())
            .collect()
    }
}

impl FromIterator<Word64> for KeccakRow {
    fn from_iter<I: IntoIterator<Item = Word64>>(iter: I) -> Self {
        let mut arr: KeccakRow;
        unsafe {
            arr = mem::uninitialized();
            (0..5).zip_longest(iter.into_iter()).for_each(|x| match x {
                Both(i, num) => arr[i] = num,
                Left(_) => {
                    panic!("FromIterator: KeccakRow cannot be constructed from more than 5 Word64")
                }
                Right(_) => {
                    panic!("FromIterator: KeccakRow cannot be constructed from less than 5 Word64")
                }
            });
        }
        arr
    }
}

/// It is a 5 by 5 matrix used as the internal state of Keccak hash
/// function or other matrix need by the Keccak hash function.
#[derive(Clone, Copy)]
pub struct KeccakMatrix([KeccakRow; 5]);

impl Default for KeccakMatrix {
    fn default() -> KeccakMatrix {
        KeccakMatrix([KeccakRow::default(); 5])
    }
}

impl Deref for KeccakMatrix {
    type Target = [KeccakRow; 5];

    fn deref(&self) -> &[KeccakRow; 5] {
        &self.0
    }
}

impl DerefMut for KeccakMatrix {
    fn deref_mut(&mut self) -> &mut [KeccakRow; 5] {
        &mut self.0
    }
}

impl FromIterator<WireId> for KeccakMatrix {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        iter.into_iter()
            .chunks(64)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Word64>())
            .collect()
    }
}

impl FromIterator<Word64> for KeccakMatrix {
    fn from_iter<I: IntoIterator<Item = Word64>>(iter: I) -> Self {
        iter.into_iter()
            .chunks(5)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<KeccakRow>())
            .collect()
    }
}

impl FromIterator<KeccakRow> for KeccakMatrix {
    fn from_iter<I: IntoIterator<Item = KeccakRow>>(iter: I) -> Self {
        let mut arr: KeccakMatrix;
        unsafe {
            arr = mem::uninitialized();
            (0..5).zip_longest(iter.into_iter()).for_each(|x| match x {
                Both(i, num) => arr[i] = num,
                Left(_) => panic!(
                    "FromIterator: KeccakMatrix cannot be constructed from more than 5 KeccakRow"
                ),
                Right(_) => panic!(
                    "FromIterator: KeccakMatrix cannot be constructed from less than 5 KeccakRow"
                ),
            });
        }
        arr
    }
}

// const rotation_offset: KeccakMatrix<u64> = KeccakMatrix([
//     [0, 36, 3, 18, 41],
//     [1, 44, 10, 45, 2],
//     [62, 6, 43, 15, 61],
//     [28, 55, 25, 21, 56],
//     [27, 20, 39, 8, 14],
// ]);

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
