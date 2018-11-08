use super::*;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};

extern crate itertools;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;

/// (For Internal Use Only!) u64 equivalent for circuits.
///
/// ## Usage Details:
///
/// IMPORTANT: Only input either 0 or 1 as inputs to `Word64` wires, or just use
/// the provided constructor in `Circuit`!
///
/// `Word64` is really just a placeholder for a u64. It does not store a u64
/// number, but can be assigned a u64 number before evaluation. Still this only
/// associates the bits of a u64 number with the `WireId`s in the `Word64`.
///
/// ### Iterator and FromIterator
///
/// Iterator: You get the bits starting from the least significant bit (the
/// right most) which you can think of as the first wire. This is so you can
/// conceptualize the array as literally being the bits of a u64 number read
/// just like you would expect. When working with `Word64` when you access the
/// values, `{let word64: Word64; word64[0] == word64.iter().next().unwrap()}`,
/// will give you the least significant bit first.
///
/// NOTE: The bit are actually inserted into the array backwards, but you don't
/// need to think about that. The debug reverses the array's order before
/// printing as well.
///
/// FromIterator: bits input into from right to left; least significant first to
/// most significant bit. Effectively creating a `Word64` the same way you would
/// write it on paper.
///
/// NOTE: if you don't give enough bits or too many bits you will get a runtime
/// panic! (this is so you realize you have made a grave mistake).
///
/// ## Example
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
pub struct Word64([WireId; 64]);

impl PartialEq for Word64 {
    fn eq(&self, other: &Word64) -> bool {
        self.0[..] == other.0[..]
    }
}

impl Eq for Word64 {}

impl fmt::Debug for Word64 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_list().entries(self.iter().rev()).finish()
    }
}

impl Default for Word64 {
    fn default() -> Word64 {
        Word64([WireId::default(); 64])
    }
}

impl Index<usize> for Word64 {
    type Output = WireId;

    fn index(&self, i: usize) -> &WireId {
        self.0.index(i)
    }
}

impl IndexMut<usize> for Word64 {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut WireId {
        self.0.index_mut(i)
    }
}

impl Word64 {
    pub fn iter(&self) -> Iter<WireId> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<WireId> {
        self.0.iter_mut()
    }
    pub fn rotate_left_mut(&mut self, mid: usize) {
        self.0.rotate_left(mid % 64)
    }
    pub fn rotate_right_mut(&mut self, mid: usize) {
        self.0.rotate_right(mid % 64)
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
        self.iter().cycle().skip(by % 64).take(64).collect()
    }

    /// Rotates a Word64's bits by moving bit a position `i` into position `i-by`
    /// modulo the lane size.
    ///
    /// Example of left rotation: (u64 truncated to u8)
    ///     start:   0100 0101
    ///     becomes: 1010 0010
    ///
    pub fn rotate_right(&self, by: usize) -> Word64 {
        self.iter().cycle().skip(64 - (by % 64)).take(64).collect()
    }
}

impl<'a> IntoIterator for &'a Word64 {
    type Item = &'a WireId;
    type IntoIter = Iter<'a, WireId>;

    fn into_iter(self) -> Iter<'a, WireId> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut Word64 {
    type Item = &'a mut WireId;
    type IntoIter = IterMut<'a, WireId>;

    fn into_iter(self) -> IterMut<'a, WireId> {
        self.0.iter_mut()
    }
}

impl FromIterator<WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        let mut arr: Word64 = Word64::default();
        (0..64).zip_longest(iter.into_iter()).for_each(|x| match x {
            Both(i, num) => arr[i] = num,
            Left(_) => {
                panic!("FromIterator: Word64 cannot be constructed from less than 64 WireId")
            }
            Right(_) => {
                panic!("FromIterator: Word64 cannot be constructed from more than 64 WireId")
            }
        });
        arr
    }
}

impl<'a> FromIterator<&'a WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = &'a WireId>>(iter: I) -> Self {
        let mut arr: Word64 = Word64::default();
        (0..64).zip_longest(iter.into_iter()).for_each(|x| match x {
            Both(i, &num) => arr[i] = num,
            Left(_) => {
                panic!("FromIterator: Word64 cannot be constructed from less than 64 WireId")
            }
            Right(_) => {
                panic!("FromIterator: Word64 cannot be constructed from more than 64 WireId")
            }
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
            .chunks(64)
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

#[cfg(test)]
mod tests {
    use super::*;

    extern crate quickcheck;
    use self::quickcheck::quickcheck;

    quickcheck! {
        fn rotate_mut_inverse_prop(rotate_by: usize) -> bool {
            let mut word64: Word64 = (0..64).map(WireId).collect();
            word64.rotate_right_mut(rotate_by);
            word64.rotate_left_mut(rotate_by);
            word64 == (0..64).map(WireId).collect()
        }
        fn rotate_inverse_prop(rotate_by: usize) -> bool {
            let word64: Word64 = (0..64).map(WireId).collect();
            word64 == word64.rotate_left(rotate_by).rotate_right(rotate_by)
        }
        fn rotate_left_prop(rotate_by: usize) -> bool {
            let w: Word64 = (0..64).map(WireId).collect();
            let mut mut_w = w.clone();
            mut_w.rotate_left_mut(rotate_by);
            mut_w == w.rotate_left(rotate_by)
        }
        fn rotate_right_prop(rotate_by: usize) -> bool {
            let w: Word64 = (0..64).map(WireId).collect();
            let mut mut_w = w.clone();
            mut_w.rotate_right_mut(rotate_by);
            mut_w == w.rotate_right(rotate_by)
        }
        fn rotate_mod(by: usize) -> bool {
            let word64: Word64 = (0..64).map(WireId).collect();
            word64.rotate_left(by + 64) == word64.rotate_left(by)
                &&
            word64.rotate_right(by + 64) == word64.rotate_right(by)
        }
    }

    /// NOTE: The KeccakMatrix tests actually tests all `FromIterator` in `Word64`
    /// because of the recurse definitions of `FromIterator`
    #[test]
    fn keccakmatrix_iterators() {
        let state: KeccakMatrix = (0..1600).map(WireId).collect();
        assert_eq!(state[0][0][0], WireId(0));
        assert_eq!(state[0][0][1], WireId(1));
        assert_eq!(state[2][3][23], WireId(855));
        assert_eq!(state[4][4][62], WireId(1598));
        assert_eq!(state[4][4][63], WireId(1599));

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
    #[should_panic(
        expected = "FromIterator: Word64 cannot be constructed from more than 64 WireId"
    )]
    fn word64_overflow_fromiterator() {
        let _state: Word64 = (0..70).map(|x| WireId(x)).collect();
    }

    #[test]
    #[should_panic(
        expected = "FromIterator: Word64 cannot be constructed from less than 64 WireId"
    )]
    fn word64_underflow_fromiterator() {
        let _state: Word64 = (0..5).map(|x| WireId(x)).collect();
    }
}
