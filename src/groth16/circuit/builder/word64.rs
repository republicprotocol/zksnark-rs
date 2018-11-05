use super::*;
use std::iter::FromIterator;
extern crate itertools;

/// (For Internal Use Only!) u64 equivalent for circuits.
///
/// ## Usage Details:
///
/// ### Iterator and FromIterator
///
/// Iterator: You actually get the bits starting from the least significant bit
/// (the right most) which I call wire1. This is so you can conceptualize the
/// array as literally being the bits of a u64 number read just like you expect
/// expect (Otherwise it would be backwards).
///
/// FromIterator: bits input into the underlying array from right to left, so
/// the first bit from the iterator goes to wire1 which is the least significant
/// bit. This is also to make creating `Word64` easer to conceptualize since you
/// are creating the number the same way you would do it on paper.
///
/// NOTE: if you don't give enough bits (input `WireId`s) then the rest will be
/// filled with `zero_wire`s and if you give too many bits they will be ignored.
///
/// ## Internal Details:
///
/// The idea here is to have each `WireId` have an input of either 0 or 1 and
/// then group those wires together to form a single `u64` number where wire1
/// corresponds to the first bit (also called the right most bit) of the `u64`,
/// then wire2 the second bit and so on. Its important to note that a `Word64`
/// is actually just a placeholder for a `u64` that still needs to be set to a
/// value in order to be a `u64` equivalent. In practice `Word64` is like a
/// type for a circuit's input except that by itself `Word64` does not guarantee
/// that its input wires only take 0 or 1.
///
/// For the sake of clarity, lets assign a value to the `Word64` while
/// explaining how to interpret the wire values:
///
/// Example: 0000 0000 0000 0000  is 0xD1 (209) so the value of the left most
///          0000 0000 0000 0000        WireId would be 0 and the right most
///          0000 0000 0000 0000        value would be 1 or if you are
///          0000 0000 0000 0000        consuming the Word64 as an iterator
///          0000 0000 1101 0001        then the first value you would get
///                                     is 1.
#[derive(Clone, Copy)]
pub struct Word64 {
    inner: [WireId; 64],
    cursor: usize,
}

impl Default for Word64 {
    fn default() -> Word64 {
        Word64 {
            inner: [WireId::default(); 64],
            cursor: 63,
        }
    }
}

impl Iterator for Word64 {
    type Item = WireId;

    fn next(&mut self) -> Option<WireId> {
        if self.cursor < 64 {
            let current = self.cursor;
            self.cursor -= 1;
            Some(self.inner[current])
        } else {
            None
        }
    }
}

impl FromIterator<WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        let mut arr: [WireId; 64] = [WireId::default(); 64];
        (0..64)
            .rev()
            .zip(iter.into_iter())
            .for_each(|(i, num)| arr[i] = num);
        Word64 {
            inner: arr,
            cursor: 63,
        }
    }
}

/// (For Internal Use Only!) In Keccak all internal arrays are of length 5
/// including the row size of the internal Keccack matrix, so this type
/// serves both purposes.
///
/// `Iterator` works as expected, starting with the first element in the row
///
/// `FromIterator` fills from left to right, like you would expect. The two
/// exceptions are the defaults used when there is not enough input and when
/// there is too much input. In the first case I fill in any empty cells with
/// `Word64` made up of `unity_wire`s. In the second case I ignore any extra
/// input.
#[derive(Clone, Copy)]
pub struct KeccakRow {
    inner: [Word64; 5],
    cursor: usize,
}

impl Default for KeccakRow {
    fn default() -> KeccakRow {
        KeccakRow {
            inner: [Word64::default(); 5],
            cursor: 0,
        }
    }
}

impl Iterator for KeccakRow {
    type Item = Word64;

    fn next(&mut self) -> Option<Word64> {
        if self.cursor < 5 {
            self.cursor += 1;
            Some(self.inner[self.cursor - 1])
        } else {
            None
        }
    }
}

impl FromIterator<Word64> for KeccakRow {
    fn from_iter<I: IntoIterator<Item = Word64>>(iter: I) -> Self {
        let mut arr = [Word64::default(); 5];
        (0..5)
            .zip(iter.into_iter())
            .for_each(|(i, num)| arr[i] = num);
        KeccakRow {
            inner: arr,
            cursor: 0,
        }
    }
}

/// (For Internal Use Only!) It is a 5 by 5 matrix used as the internal state of
/// Keccak hash function and other constants.
///
/// `Iterator` works as expected, starting with the first row in the matrix.
/// (Yes it gives you back an iterator for each row)
///
/// `FromIterator` fills from top to bottom, like you would expect. The two
/// exceptions are the defaults used when there is not enough input and when
/// there is too much input. In the first case I fill in any empty rows with
/// `KeccakRow`s made up of `Word64`s of `unity_wire`s. In the second case I
/// ignore any extra input.
#[derive(Clone, Copy)]
pub struct KeccakMatrix {
    inner: [KeccakRow; 5],
    cursor: usize,
}

impl Default for KeccakMatrix {
    fn default() -> KeccakMatrix {
        KeccakMatrix {
            inner: [KeccakRow::default(); 5],
            cursor: 0,
        }
    }
}

impl Iterator for KeccakMatrix {
    type Item = KeccakRow;

    fn next(&mut self) -> Option<KeccakRow> {
        if self.cursor < 5 {
            self.cursor += 1;
            Some(self.inner[self.cursor - 1])
        } else {
            None
        }
    }
}

impl FromIterator<KeccakRow> for KeccakMatrix {
    fn from_iter<I: IntoIterator<Item = KeccakRow>>(iter: I) -> Self {
        let mut arr = [KeccakRow::default(); 5];
        (0..5)
            .zip(iter.into_iter())
            .for_each(|(i, num)| arr[i] = num);
        KeccakMatrix {
            inner: arr,
            cursor: 0,
        }
    }
}

/// Rotates a Word64's bits by moving bit a position `i` into position `i+by`
/// module the lane size.
///
/// TODO: Write tests!
///
pub fn left_rotate(input: Word64, by: usize) -> Word64 {
    input.cycle().skip(by).take(64).collect()
}

/// Rotates a Word64's bits by moving bit a position `i` into position `i-by`
/// module the lane size.
///
/// TODO: Write tests!
/// FIXME: Either one of the rotates is wrong
///
pub fn right_rotate(input: Word64, by: usize) -> Word64 {
    input.cycle().skip(by).take(64).collect()
}

// const rotation_offset: KeccakMatrix = [
//     [0, 36, 3, 18, 41].into_iter(),
//     [1, 44, 10, 45, 2].into_iter(),
//     [62, 6, 43, 15, 61].into_iter(),
//     [28, 55, 25, 21, 56].into_iter(),
//     [27, 20, 39, 8, 14].into_iter(),
// ]
//     .into_iter()
//     .collect();

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
