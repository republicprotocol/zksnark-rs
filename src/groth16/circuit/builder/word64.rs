use super::*;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};

extern crate itertools;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;

/// ## Usage Details:
///
/// IMPORTANT:
///     - Only input either 0 or 1 as inputs to `Word8` wires, or just use
///       the provided constructor in `Circuit`!
///     - Word8 is stored little-endian: For example lets say you input the
///       number 0x4B (ends in: 0100 1011) into the Word8 placeholder then it
///       would be stored as: [1,1,0,1,0,0,1,0]
///
/// ### Iterator and FromIterator
///
/// Iterator: You get the bits starting from the least significant bit, which
/// you can think of as the first wire on the left.
///
/// FromIterator: bits input into from right to left; least significant first to
/// most significant bit.
///
/// NOTE: if you don't give enough bits the extra bits will be filled in with
/// WireId::default() which is the zero wire. On the other hand if you have too
/// many bits you will get a runtime panic! (this is so you realize you have
/// made a grave mistake). The only way to get the panic that you have
/// given too many bits is to collect a word8 directly.
///
#[derive(Clone, Copy)]
pub struct Word8([WireId; 8]);

impl PartialEq for Word8 {
    fn eq(&self, other: &Word8) -> bool {
        self.0[..] == other.0[..]
    }
}

impl Eq for Word8 {}

impl fmt::Debug for Word8 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_list().entries(self.iter()).finish()
    }
}

impl Default for Word8 {
    fn default() -> Word8 {
        Word8([WireId::default(); 8])
    }
}

impl Index<usize> for Word8 {
    type Output = WireId;

    fn index(&self, i: usize) -> &WireId {
        self.0.index(i)
    }
}

impl IndexMut<usize> for Word8 {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut WireId {
        self.0.index_mut(i)
    }
}

impl Word8 {
    pub fn iter(&self) -> Iter<WireId> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<WireId> {
        self.0.iter_mut()
    }
    pub fn rotate_left_mut(&mut self, mid: usize) {
        self.0.rotate_left(mid % 8)
    }
    pub fn rotate_right_mut(&mut self, mid: usize) {
        self.0.rotate_right(mid % 8)
    }
    /// Rotates a Word8's bits by moving bit a position `i` into position `i+by`
    /// modulo the lane size. The least significant bit is where i = 0 and the most
    /// significant bit is where i = 7.
    ///
    /// Example of left rotation:
    ///     start:   0100 0101
    ///     becomes: 1000 1010
    ///
    pub fn rotate_left(&self, by: usize) -> Word8 {
        self.iter().cycle().skip(by % 8).take(8).collect()
    }

    /// Rotates a Word8's bits by moving bit a position `i` into position `i-by`
    /// modulo the lane size.
    ///
    /// Example of left rotation:
    ///     start:   0100 0101
    ///     becomes: 1010 0010
    ///
    pub fn rotate_right(&self, by: usize) -> Word8 {
        self.iter().cycle().skip(8 - (by % 8)).take(8).collect()
    }
}

impl<'a> IntoIterator for &'a Word8 {
    type Item = &'a WireId;
    type IntoIter = Iter<'a, WireId>;

    fn into_iter(self) -> Iter<'a, WireId> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut Word8 {
    type Item = &'a mut WireId;
    type IntoIter = IterMut<'a, WireId>;

    fn into_iter(self) -> IterMut<'a, WireId> {
        self.0.iter_mut()
    }
}

impl FromIterator<WireId> for Word8 {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        let mut arr: Word8 = Word8::default();
        (0..8).zip_longest(iter.into_iter()).for_each(|x| match x {
            Both(i, num) => arr[i] = num,
            Left(i) => arr[i] = WireId::default(),
            Right(_) => panic!("FromIterator: Word8 cannot be constructed from more than 8 WireId"),
        });
        arr
    }
}

impl<'a> FromIterator<&'a WireId> for Word8 {
    fn from_iter<I: IntoIterator<Item = &'a WireId>>(iter: I) -> Self {
        let mut arr: Word8 = Word8::default();
        (0..8).zip_longest(iter.into_iter()).for_each(|x| match x {
            Both(i, &num) => arr[i] = num,
            Left(i) => arr[i] = WireId::default(),
            Right(_) => panic!("FromIterator: Word8 cannot be constructed from more than 8 WireId"),
        });
        arr
    }
}

/// ## Usage Details:
///
/// IMPORTANT:
///     - Only input either 0 or 1 as inputs to `Word64` wires, or just use
///       the provided constructor in `Circuit`!
///     - Word64 is stored little-endian: For example lets say you input the
///       first byte as 0x4F (0100 1111) and second byte as 0x4B (01001011) into
///       the Word64 placeholder then it would be stored as:
///       [1,1,1,1,0,0,1,0, 1,1,0,1,0,0,1,0, (48 zeros)]
///
/// `Word64` is really just a placeholder for a u64. It does not store a u64
/// number, but can be assigned a u64 number before evaluation. Still this only
/// associates the bits of a u64 number with the `WireId`s in the `Word64`.
///
///
/// ### Iterator and FromIterator
///
/// Iterator: You get the bytes starting from the least significant byte, which
/// you can think of as the first Word8 on the left.
///
/// FromIterator: bytes input into from right to left; least significant first to
/// most significant byte.
///
/// NOTE: if you don't give enough bits the extra bits will be filled in with
/// WireId::default() which is the zero wire. On the other hand if you have too
/// many bits you will get a runtime panic! (this is so you realize you have
/// made a grave mistake). The only way to get the panic that you have
/// given too many bits is to collect a word64 directly.
///
#[derive(Clone, Copy)]
pub struct Word64([Word8; 8]);

impl PartialEq for Word64 {
    fn eq(&self, other: &Word64) -> bool {
        self.0[..] == other.0[..]
    }
}

impl Eq for Word64 {}

impl fmt::Debug for Word64 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_list().entries(self.iter()).finish()
    }
}

impl Default for Word64 {
    fn default() -> Word64 {
        Word64([Word8::default(); 8])
    }
}

impl Index<usize> for Word64 {
    type Output = Word8;

    fn index(&self, i: usize) -> &Word8 {
        self.0.index(i)
    }
}

impl IndexMut<usize> for Word64 {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut Word8 {
        self.0.index_mut(i)
    }
}

impl Word64 {
    pub fn iter(&self) -> Iter<Word8> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<Word8> {
        self.0.iter_mut()
    }
    /// Rotates a Word64's bits by moving bit a position `i` into position `i+by`
    /// modulo the lane size. The least significant bit is where i = 0 and the most
    /// significant bit is where i = 63.
    ///
    /// Example of left rotation: (u64 truncated to u8)
    ///     start:   0100 0101
    ///     becomes: 1000 1010
    ///
    pub fn rotate_left(&self, by: usize) -> Word64 {
        self.iter()
            .flat_map(|x| x.iter())
            .cycle()
            .skip(by % 64)
            .take(64)
            .collect()
    }

    /// Rotates a Word64's bits by moving bit a position `i` into position `i-by`
    /// modulo the lane size.
    ///
    /// Example of left rotation: (u64 truncated to u8)
    ///     start:   0100 0101
    ///     becomes: 1010 0010
    ///
    pub fn rotate_right(&self, by: usize) -> Word64 {
        self.iter()
            .flat_map(|x| x.iter())
            .cycle()
            .skip(64 - (by % 64))
            .take(64)
            .collect()
    }
}

impl<'a> IntoIterator for &'a Word64 {
    type Item = &'a Word8;
    type IntoIter = Iter<'a, Word8>;

    fn into_iter(self) -> Iter<'a, Word8> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut Word64 {
    type Item = &'a mut Word8;
    type IntoIter = IterMut<'a, Word8>;

    fn into_iter(self) -> IterMut<'a, Word8> {
        self.0.iter_mut()
    }
}

impl FromIterator<WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        iter.into_iter()
            .chunks(8)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Word8>())
            .collect()
    }
}

impl<'a> FromIterator<&'a WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = &'a WireId>>(iter: I) -> Self {
        iter.into_iter()
            .chunks(8)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Word8>())
            .collect()
    }
}

impl FromIterator<Word8> for Word64 {
    fn from_iter<I: IntoIterator<Item = Word8>>(iter: I) -> Self {
        let mut arr: Word64 = Word64::default();
        (0..8).zip_longest(iter.into_iter()).for_each(|x| match x {
            Both(i, num) => arr[i] = num,
            Left(i) => arr[i] = Word8::default(),
            Right(_) => panic!("FromIterator: Word64 cannot be constructed from more than 8 Word8"),
        });
        arr
    }
}

impl<'a> FromIterator<&'a Word8> for Word64 {
    fn from_iter<I: IntoIterator<Item = &'a Word8>>(iter: I) -> Self {
        let mut arr: Word64 = Word64::default();
        (0..8).zip_longest(iter.into_iter()).for_each(|x| match x {
            Both(i, &num) => arr[i] = num,
            Left(i) => arr[i] = Word8::default(),
            Right(_) => panic!("FromIterator: Word64 cannot be constructed from more than 8 Word8"),
        });
        arr
    }
}

/// (For Internal Use Only!) In Keccak all internal arrays are of length 5
/// including the row size of the internal Keccack matrix, so this type
/// serves both purposes.
///
/// `Iterator` works as expected, starting with the first element in the row
///
/// `FromIterator` fills from left to right, like you would expect. It panics if
/// you either give it too many `Word64` or too few; must be equal to 5.
///
#[derive(Clone, Copy)]
pub struct KeccakRow([Word64; 5]);

impl PartialEq for KeccakRow {
    fn eq(&self, other: &KeccakRow) -> bool {
        self.iter()
            .zip(other.iter())
            .fold(true, |acc, (l, r)| acc && l == r)
    }
}

impl Eq for KeccakRow {}

impl fmt::Debug for KeccakRow {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_list().entries(self.iter()).finish()
    }
}

impl Default for KeccakRow {
    fn default() -> KeccakRow {
        KeccakRow([Word64::default(); 5])
    }
}

impl Index<isize> for KeccakRow {
    type Output = Word64;

    fn index(&self, i: isize) -> &Word64 {
        let mut i = i % 5;
        if i < 0 {
            i = i + 5;
        }
        self.0.index(i as usize)
    }
}

impl IndexMut<isize> for KeccakRow {
    fn index_mut<'a>(&'a mut self, i: isize) -> &'a mut Word64 {
        let mut i = i % 5;
        if i < 0 {
            i = i + 5;
        }
        self.0.index_mut(i as usize)
    }
}

impl KeccakRow {
    pub fn iter(&self) -> Iter<Word64> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<Word64> {
        self.0.iter_mut()
    }
}

impl<'a> IntoIterator for &'a KeccakRow {
    type Item = &'a Word64;
    type IntoIter = Iter<'a, Word64>;

    fn into_iter(self) -> Iter<'a, Word64> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut KeccakRow {
    type Item = &'a mut Word64;
    type IntoIter = IterMut<'a, Word64>;

    fn into_iter(self) -> IterMut<'a, Word64> {
        self.0.iter_mut()
    }
}

impl FromIterator<WireId> for KeccakRow {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        iter.into_iter()
            .chunks(8)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Word8>())
            .collect()
    }
}

impl FromIterator<Word8> for KeccakRow {
    fn from_iter<I: IntoIterator<Item = Word8>>(iter: I) -> Self {
        iter.into_iter()
            .chunks(8)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Word64>())
            .collect()
    }
}

impl FromIterator<Word64> for KeccakRow {
    fn from_iter<I: IntoIterator<Item = Word64>>(iter: I) -> Self {
        let mut arr: KeccakRow = KeccakRow::default();
        (0..5).zip_longest(iter.into_iter()).for_each(|x| match x {
            Both(i, num) => arr[i] = num,
            Left(_) => {
                panic!("FromIterator: KeccakRow cannot be constructed from less than 5 Word64")
            }
            Right(_) => {
                panic!("FromIterator: KeccakRow cannot be constructed from more than 5 Word64")
            }
        });
        arr
    }
}

/// (For Internal Use Only!) It is a 5 by 5 matrix used as the internal state of
/// Keccak hash function and other constants.
///
/// `Iterator` works as expected, starting with the first row in the matrix.
/// (Yes it gives you back a KeccakRow that you can call `.iter()` on.)
///
/// `FromIterator` fills from top to bottom, like you would expect. It panics if
/// you either give it too many `KeccakRow` or too few; must be equal to 5.
///
#[derive(Clone, Copy)]
pub struct KeccakMatrix([KeccakRow; 5]);

impl PartialEq for KeccakMatrix {
    fn eq(&self, other: &KeccakMatrix) -> bool {
        self.iter()
            .zip(other.iter())
            .fold(true, |acc, (l, r)| acc && l == r)
    }
}

impl Eq for KeccakMatrix {}

impl fmt::Debug for KeccakMatrix {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_list().entries(self.iter()).finish()
    }
}

impl Default for KeccakMatrix {
    fn default() -> KeccakMatrix {
        KeccakMatrix([KeccakRow::default(); 5])
    }
}

impl Index<isize> for KeccakMatrix {
    type Output = KeccakRow;

    fn index(&self, i: isize) -> &KeccakRow {
        let mut i = i % 5;
        if i < 0 {
            i = i + 5;
        }
        self.0.index(i as usize)
    }
}

impl IndexMut<isize> for KeccakMatrix {
    fn index_mut<'a>(&'a mut self, i: isize) -> &'a mut KeccakRow {
        let mut i = i % 5;
        if i < 0 {
            i = i + 5;
        }
        self.0.index_mut(i as usize)
    }
}

impl KeccakMatrix {
    pub fn iter(&self) -> Iter<KeccakRow> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<KeccakRow> {
        self.0.iter_mut()
    }
}

impl<'a> IntoIterator for &'a KeccakMatrix {
    type Item = &'a KeccakRow;
    type IntoIter = Iter<'a, KeccakRow>;

    fn into_iter(self) -> Iter<'a, KeccakRow> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut KeccakMatrix {
    type Item = &'a mut KeccakRow;
    type IntoIter = IterMut<'a, KeccakRow>;

    fn into_iter(self) -> IterMut<'a, KeccakRow> {
        self.0.iter_mut()
    }
}

impl FromIterator<WireId> for KeccakMatrix {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        iter.into_iter()
            .chunks(8)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Word8>())
            .collect()
    }
}

impl FromIterator<Word8> for KeccakMatrix {
    fn from_iter<I: IntoIterator<Item = Word8>>(iter: I) -> Self {
        iter.into_iter()
            .chunks(8)
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
        let mut arr: KeccakMatrix = KeccakMatrix::default();
        (0..5).zip_longest(iter.into_iter()).for_each(|x| match x {
            Both(i, num) => arr[i] = num,
            Left(_) => panic!(
                "FromIterator: KeccakMatrix cannot be constructed from less than 5 KeccakRow"
            ),
            Right(_) => panic!(
                "FromIterator: KeccakMatrix cannot be constructed from more than 5 KeccakRow"
            ),
        });
        arr
    }
}

pub const ROUND_CONSTANTS: [u64; 24] = [
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

/// This is copied from the std library because it was marked as a nightly only
/// feature, not because it is unstable, but because they were not sure if it
/// should be added and named where it was.
fn to_ne_u8(num: u64) -> [u8; std::mem::size_of::<u64>()] {
    unsafe { std::mem::transmute(num) }
}
fn from_ne_u64(bytes: [u8; std::mem::size_of::<u64>()]) -> u64 {
    unsafe { std::mem::transmute(bytes) }
}
pub fn to_le_u8(num: u64) -> [u8; std::mem::size_of::<u64>()] {
    to_ne_u8(num.to_le())
}
pub fn from_le_u64(bytes: [u8; std::mem::size_of::<u64>()]) -> u64 {
    from_ne_u64(bytes).to_le()
}

pub fn flatten_word64(input: Word64) -> [WireId; 64] {
    let mut wires_u64: [WireId; 64] = [WireId::default(); 64];
    input
        .iter()
        .enumerate()
        .for_each(|(i, &wrd8): (usize, &Word8)| {
            wrd8.iter()
                .enumerate()
                .for_each(|(j, &wire_id)| wires_u64[(i * 8) + j] = wire_id)
        });
    wires_u64
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate quickcheck;
    use self::quickcheck::quickcheck;

    quickcheck! {
        fn rotate_inverse_prop(rotate_by: usize) -> bool {
            let word64: Word64 = (0..64).map(WireId).collect();
            word64 == word64.rotate_left(rotate_by).rotate_right(rotate_by)
        }
        fn rotate_mod(by: usize) -> bool {
            let word64: Word64 = (0..64).map(WireId).collect();
            word64.rotate_left(by + 64) == word64.rotate_left(by)
                &&
            word64.rotate_right(by + 64) == word64.rotate_right(by)
        }
        fn endian_prop(num: u64) -> bool {
            from_le_u64(to_le_u8(num)) == num
        }
    }

    fn rotate_single_test() {
        let a_wrd64: Word64 = (0..64).map(WireId).collect();
        let b_wrd64: Word64 = (64..65).chain(1..63).map(WireId).collect();
        assert_eq!(b_wrd64, a_wrd64.rotate_right(1));
    }

    #[test]
    fn to_from_little_endian() {
        assert_eq!(
            to_le_u8(0b1000_0000_0000_0000),
            [0, 0b1000_0000, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            from_le_u64([0, 0b1000_0000, 0, 0, 0, 0, 0, 0]),
            0b1000_0000_0000_0000
        );
    }

    /// NOTE: The KeccakMatrix tests actually tests all `FromIterator` in `Word64`
    /// because of the recurse definitions of `FromIterator`
    #[test]
    fn keccakmatrix_iterators() {
        let state: KeccakMatrix = (0..1600).map(WireId).collect();
        assert_eq!(state[0][0][0][0], WireId(0));
        assert_eq!(state[0][0][1][0], WireId(8));
        assert_eq!(state[2][3][5][0], WireId(872));
        assert_eq!(state[4][4][6][0], WireId(1584));
        assert_eq!(state[4][4][7][7], WireId(1599));

        let mut iter = state.into_iter();

        assert_eq!(
            iter.next().unwrap()[0],
            (0..64).map(|x| WireId(x)).collect()
        );
    }

    #[test]
    #[should_panic(
        expected = "FromIterator: KeccakMatrix cannot be constructed from more than 5 KeccakRow"
    )]
    fn keccakmatrix_overflow_fromiterator() {
        let _state: KeccakMatrix = (0..1920).map(|x| WireId(x)).collect();
    }

    #[test]
    #[should_panic(
        expected = "FromIterator: KeccakMatrix cannot be constructed from less than 5 KeccakRow"
    )]
    fn keccakmatrix_underflow_fromiterator() {
        let _state: KeccakMatrix = (0..1280).map(|x| WireId(x)).collect();
    }

    #[test]
    #[should_panic(
        expected = "FromIterator: KeccakRow cannot be constructed from more than 5 Word64"
    )]
    fn keccakrow_overflow_fromiterator() {
        let _state: KeccakRow = (0..384).map(|x| WireId(x)).collect();
    }

    #[test]
    #[should_panic(
        expected = "FromIterator: KeccakRow cannot be constructed from less than 5 Word64"
    )]
    fn keccakrow_underflow_fromiterator() {
        let _state: KeccakRow = (0..256).map(|x| WireId(x)).collect();
    }

    #[test]
    #[should_panic(expected = "FromIterator: Word64 cannot be constructed from more than 8 Word8")]
    fn word64_overflow_fromiterator() {
        let _state: Word64 = (0..70).map(|x| WireId(x)).collect();
    }
}
