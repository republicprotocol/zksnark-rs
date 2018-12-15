use super::*;

extern crate itertools;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use std::iter::FromIterator;
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};

pub trait BinaryInput {}

pub trait CanConvert<T> {}

#[derive(Clone, Copy, Debug)]
pub enum Binary {
    Zero,
    One,
}

pub struct ValidateOrder {
    pub is_x_within_range: WireId,
    pub is_y_greater_than_c: WireId,
    pub hash_x_y: [Word8; 32],
}

pub struct ValidateBalance {
    pub x_hash: [Word8; 32],
    pub y_hash: [Word8; 32],
    pub z_hash: [Word8; 32],
    pub is_z_eq_x_min_y: WireId,
}

// TODO: Write a binary version of WireId

/// ## Usage Details:
///
/// IMPORTANT:
///     - Only input either 0 or 1 as inputs to `Word8` wires, or just use
///       the provided constructor in `Circuit`!
///     - Word8 is stored in the array as if it was little-endian: For example
///       lets say you input the number 0x4B (ends in: 0100 1011) into the Word8
///       placeholder then it would be stored as: [1,1,0,1,0,0,1,0]
///
#[derive(Clone, Copy, Debug)]
pub struct Word8([WireId; 8]);

impl Word8 {
    pub fn iter(&self) -> Iter<WireId> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a Word8 {
    type Item = &'a WireId;
    type IntoIter = Iter<'a, WireId>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<WireId> for Word8 {
    fn from_iter<T: IntoIterator<Item = WireId>>(iter: T) -> Self {
        let mut arr: Word8 = Word8::default();
        (0..8).zip_longest(iter).for_each(|x| match x {
            Both(i, num) => arr[i] = num,
            Left(_) => panic!("to_word8: Word8 cannot be constructed from less than 8 WireId"),
            Right(_) => panic!("to_word8: Word8 cannot be constructed from more than 8 WireId"),
        });
        arr
    }
}

impl<'a> FromIterator<&'a WireId> for Word8 {
    fn from_iter<T: IntoIterator<Item = &'a WireId>>(iter: T) -> Self {
        let mut arr: Word8 = Word8::default();
        (0..8).zip_longest(iter).for_each(|x| match x {
            Both(i, &num) => arr[i] = num,
            Left(_) => panic!("to_word8: Word8 cannot be constructed from less than 8 WireId"),
            Right(_) => panic!("to_word8: Word8 cannot be constructed from more than 8 WireId"),
        });
        arr
    }
}

impl<'a> BinaryInput for &'a Word8 {}
impl<'a> CanConvert<u8> for &'a Word8 {}

impl PartialEq for Word8 {
    fn eq(&self, other: &Word8) -> bool {
        self.0 == other.0
    }
}
impl Eq for Word8 {}

impl Default for Word8 {
    fn default() -> Word8 {
        Word8([WireId::default(); 8])
    }
}

impl Index<usize> for Word8 {
    type Output = WireId;

    fn index<'a>(&'a self, index: usize) -> &'a WireId {
        &self.0[index]
    }
}

impl IndexMut<usize> for Word8 {
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut WireId {
        &mut self.0[index]
    }
}

/// This is a convenience function to create a `Word8` from exactly 8
/// WireId any more or less will cause a panic
pub fn to_word8(input: impl Iterator<Item = WireId>) -> Word8 {
    let mut arr: Word8 = Word8::default();
    (0..8).zip_longest(input).for_each(|x| match x {
        Both(i, num) => arr[i] = num,
        Left(_) => panic!("to_word8: Word8 cannot be constructed from less than 8 WireId"),
        Right(_) => panic!("to_word8: Word8 cannot be constructed from more than 8 WireId"),
    });
    arr
}

// TODO: when you get the time refactor this to work just on
// references. The reason you don't now is the way this function
// interacts with to_word8 and the way you are using to_word8
pub fn flatten_word8<'a>(input: impl IntoIterator<Item = &'a Word8>) -> Vec<WireId> {
    input.into_iter().flat_map(|x| x.iter()).cloned().collect()
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
#[derive(Clone, Copy, Debug)]
pub struct Word64([Word8; 8]);

impl Word64 {
    pub fn iter(&self) -> Iter<Word8> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<Word8> {
        self.0.iter_mut()
    }
}

pub struct Word64Iter<'a> {
    count: usize,
    wrd64: &'a Word64,
}

impl<'a> Iterator for Word64Iter<'a> {
    type Item = &'a WireId;

    fn next(&mut self) -> Option<&'a WireId> {
        if self.count < 64 {
            let x = Some(&self.wrd64[self.count / 8][self.count % 8]);
            self.count += 1;
            x
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a Word64 {
    type Item = &'a WireId;
    type IntoIter = Word64Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Word64Iter {
            count: 0,
            wrd64: &self,
        }
    }
}

impl<'a> BinaryInput for &'a Word64 {}
impl<'a> CanConvert<u64> for &'a Word64 {}

impl PartialEq for Word64 {
    fn eq(&self, other: &Word64) -> bool {
        self.0 == other.0
    }
}
impl Eq for Word64 {}

impl Default for Word64 {
    fn default() -> Word64 {
        Word64([Word8::default(); 8])
    }
}

impl Index<usize> for Word64 {
    type Output = Word8;

    fn index<'a>(&'a self, index: usize) -> &'a Word8 {
        &self.0[index]
    }
}

impl IndexMut<usize> for Word64 {
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut Word8 {
        &mut self.0[index]
    }
}

impl<T> BinaryInput for [T] where T: BinaryInput {}
impl<T> BinaryInput for Vec<T> where T: BinaryInput {}
impl<'a, T> BinaryInput for Iter<'a, T> where T: BinaryInput {}

/// Rotates a Word64's bits by moving bit at position `i` into position `i+by`
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

/// This is a convenience function to create a `Word64` from exactly
/// 64 WireId any more or less will cause a panic
pub fn to_word64(input: impl Iterator<Item = WireId>) -> Word64 {
    let mut arr: Word64 = Word64::default();
    input
        .chunks(8)
        .into_iter()
        .map(|chunk| to_word8(chunk))
        .zip_longest(0..8)
        .for_each(|x| match x {
            Both(num, i) => arr[i] = num,
            Right(_) => panic!("to_word64: Word64 cannot be constructed from less than 64 WireId"),
            Left(_) => panic!("to_word64: Word64 cannot be constructed from more than 64 WireId"),
        });
    arr
}

pub fn flatten_word64<'a>(input: impl Iterator<Item = &'a Word64>) -> Vec<WireId> {
    input
        .flat_map(|x| x.iter().flat_map(|i| i.iter()))
        .cloned()
        .collect()
}

pub const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
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

#[cfg(test)]
mod tests {
    use super::*;

    extern crate quickcheck;
    use self::quickcheck::quickcheck;

    #[test]
    fn word8_iterator() {
        let wrd8: Word8 = ((0..8).map(WireId)).collect();
        let wrd8_2 = wrd8.into_iter().collect();
        assert_eq!(wrd8, wrd8_2);
    }

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
        fn flatten_to_word64_prop(rand: Vec<usize>) -> bool {
            let wrd64: Vec<WireId> = rand.into_iter().chain(0..64).take(64).map(|x| WireId(x)).collect();
            let copy = wrd64.clone();

            copy == flatten_word64([to_word64(wrd64.into_iter())].iter())
        }
    }

    #[test]
    fn rotate_single_test() {
        let a_wrd64: Word64 = to_word64((0..64).map(WireId));
        let b_wrd64: Word64 = to_word64((63..64).chain(0..63).map(WireId));
        assert_eq!(b_wrd64, rotate_word64_left(a_wrd64, 1));
    }
}
