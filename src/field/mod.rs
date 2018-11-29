//! Defines the `Field` trait along with other utility functions for working
//! with fields.
//!
//! Usually you won't need to dive into this module unless you want to define
//! a new type of with either `Field` or `Polynomial` traits.
//!
//! The `z251` module is an implementation of a `Field`, useful for testing and
//! getting to understand how many of the functions work. As such most of the
//! examples use `z251` in them.
//!
//! Also, the examples in this module all use a Vec<Z251> since there is an
//! implementation of `Polynomial` for Vec, but in the code that uses this
//! module uses `CoefficentPoly`.
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! use zksnark::field::z251::Z251;
//! use zksnark::field::*;
//!
//! // `evaluate` a `Polynomial`
//! //
//! // [1, 1, 1] would be f(x) = 1 + x + x^2 thus f(2) = 1 + 2 + 2^2
//! // Thus the evaluation would be 7
//! let poly_eval = vec![1, 1, 1]
//!     .into_iter()
//!     .map(Z251::from)
//!     .collect::<Vec<_>>();
//!
//! assert_eq!(poly_eval.evaluate(Z251::from(2)), Z251::from(7));
//!
//! // `polynomial_division`
//! //
//! let poly: Vec<Z251> = vec![1, 0, 3, 1].into_iter().map(Z251::from).collect();
//! let polyDividend: Vec<Z251> = vec![0, 0, 9, 1].into_iter().map(Z251::from).collect();
//!
//! let num: Vec<Z251> = vec![1].into_iter().map(Z251::from).collect();
//! let den: Vec<Z251> = vec![1, 0, 245].into_iter().map(Z251::from).collect();
//!
//! assert_eq!(polynomial_division(poly, polyDividend), (num, den));
//!```
extern crate itertools;

use self::itertools::unfold;
use itertools::Itertools;
use std::ops::*;
use std::str::FromStr;

#[doc(hidden)]
pub mod z251;

/// `FieldIdentity` only makes sense when defined with a Field. The reason
/// this trait is not a part of [`Field`] is to provide a "zero" element and a
/// "one" element to types that cannot define a multiplicative inverse to be a
/// `Field`. Currently this includes: `isize` and is used in `z251`.
///
/// As such `zero()` is the value that equals an element added to its additive
/// inverse and the `one()` is the value that equals an element multiplied by
/// its multiplicative inverse.
pub trait FieldIdentity {
    fn zero() -> Self;
    fn one() -> Self;
}

impl FieldIdentity for isize {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

/// A `Field` here has the same classical mathematical definition of a field.
pub trait Field:
    Sized
    + Add<Output = Self>
    + Neg<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + FieldIdentity
    + Copy
    + PartialEq
    + Eq
{
    fn mul_inv(self) -> Self;
    fn add_inv(self) -> Self {
        -self
    }
}

/// The core reason we need a function like this is to let us assign
/// `WireId`s as the bits from a stream of `u8`.
///
/// Each `T` in the returned vector is either `zero()` or `one()` of
/// the field and represents a single bit from a number stream. For
/// Example, there would be 8 `T` for each `u8` in the stream. Where
/// `vec[0] == from.first().[0]` (first value in the returned vector
/// is the least significant bit (first bit) of the first number in
/// `from`.
///
/// Note: I use the type of the input numbers to determine their size.
/// Rust will happily default to a type that is Not what you intended
/// and will result in undesirable behaviour. Tell Rust your number
/// types.
///
/// ```
/// use zksnark::field::z251::Z251;
/// use zksnark::field::*;
///
/// let vec: Vec<u8> = vec![0b0000_0101];
/// let tmp: Vec<Z251> = to_field_bits(&vec);
///
/// assert_eq!(tmp[0], Z251::one());
/// assert_eq!(tmp[1], Z251::zero());
/// assert_eq!(tmp[2], Z251::one());
/// assert_eq!(tmp[3], Z251::zero());
///
/// assert_eq!(tmp[4], Z251::zero());
/// assert_eq!(tmp[5], Z251::zero());
/// assert_eq!(tmp[6], Z251::zero());
/// assert_eq!(tmp[7], Z251::zero());
///
/// assert_eq!(tmp.len(), 8);
///
/// let vec: Vec<u64> = vec![32769];
/// let tmp: Vec<Z251> = to_field_bits(&vec);
///
/// assert_eq!(tmp[0], Z251::one());
/// assert_eq!(tmp[15], Z251::one());
///
/// assert_eq!(tmp.len(), 64);
/// ```
pub fn to_field_bits<'a, T, N: 'a>(from: impl IntoIterator<Item = &'a N>) -> Vec<T>
where
    T: Field,
    N: Sized + Rem<Output = N> + Shr<Output = N> + Eq + From<u8> + Copy,
{
    from.into_iter()
        .flat_map(|num| {
            (0..(std::mem::size_of::<N>() * 8)).map(move |x| {
                if num.shr(N::from(x as u8)) % N::from(2) == N::from(0) {
                    T::zero()
                } else {
                    T::one()
                }
            })
        }).collect()
}

/// The core reason we need a function like this is to let us cast
/// some bits into `u8` or `u64` where the bits are Field elements
/// (`zero()` or `one()`) from evaluating `WireId`s
///
/// Again, each `T` in the returned vector is either `zero()` or
/// `one()` and represents a single bit from one of the `u8` or `u64`
/// your are creating. For Example, there would be 8 `T` for each `u8`
/// in the stream. Where `num_at_bit[0] == from.first()` (first value
/// in the returned vector is the least significant bit (first bit) of
/// the first number.
///
/// Note: I use the type of the input numbers to determine their size.
/// Rust will happily default to a type that is Not what you intended
/// and will result in undesirable behaviour. Tell Rust your number
/// types.
///
/// ```
/// use zksnark::field::z251::Z251;
/// use zksnark::field::*;
///
/// let itr: Vec<Z251> =
///     [1,0,1,0,0,0,0,0].into_iter()
///                      .map(|x: &usize| Z251::from(*x))
///                      .collect();
///                                
/// let tmp: Vec<u8> = from_field_bits(&itr);
///
/// assert_eq!(tmp[0], 5);
///
/// assert_eq!(tmp.len(), 1);
/// ```
pub fn from_field_bits<'a, T: 'a, N>(from: impl IntoIterator<Item = &'a T>) -> Vec<N>
where
    T: Field,
    N: Sized + BitXor<Output = N> + Shl<Output = N> + Eq + From<u8>,
{
    from.into_iter().chunks(std::mem::size_of::<N>() * 8)
        .into_iter()
        .map(|chunk| {
            chunk.enumerate().fold(N::from(0), |acc, (i, &t)| {
                if t == T::one() {
                    acc ^ (N::from(1) << N::from(i as u8))
                } else if t == T::zero() {
                    acc
                }   else {
                    panic!("from_field_bits: was given a field element that was neither zero() or one()");
                }
            })
        }).collect()
}

/// A line, `Polynomial`, represented as a vector of `Field` elements where the
/// position in the vector determines the power of the exponent.
///
/// # Remarks
///
/// If you want examples of how to implement a `Polynomial` go to the `Z251`
/// module.
///
/// The polynomial is represented as a list of coefficients where the powers of
/// "x" are implicit.
///
/// For Example: [1, 3, 0, 5] is f(x) = 1 + 3x + 5x^3
///
/// # Note
///
/// ```
/// use zksnark::field::z251::Z251;
/// use zksnark::field::*;
///
/// // The (*) is overloaded to give you back an array
/// let tmp = vec![1,2,0,4].into_iter()
///                        .map(Z251::from)
///                        .collect::<Vec<_>>();
///
/// assert_eq!(*tmp, [1.into(),2.into(),0.into(),4.into()]);
/// ```
pub trait Polynomial<T>: From<Vec<T>> + Deref<Target = [T]>
where
    T: Field,
{
    /// This defines how to turn a `Polynomial` into a vector of `Field`. In
    /// other words, it gives you back the coefficients of the `Polynomial`.
    ///
    /// However, since Deref is required for `Polynomial` you may prefer to get
    /// the coefficients through an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use zksnark::field::z251::Z251;
    /// use zksnark::field::*;
    ///
    /// // Get coefficients through an iterator
    /// let poly = vec![1,2,0,4].into_iter()
    ///                         .map(Z251::from)
    ///                         .collect::<Vec<_>>();
    ///
    /// let mut iter = poly.iter();
    ///
    /// assert_eq!(iter.next(), Some(&Z251::from(1)));
    /// ```
    fn coefficients(&self) -> Vec<T> {
        self.iter().map(|&x| x).collect()
    }

    /// Returns the highest exponent of the polynomial.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use zksnark::field::z251::Z251;
    /// use zksnark::field::*;
    ///
    /// // [1, 2, 0, 4] would be f(x) = 1 + 2x + 0x^2 + 4x^3
    /// // Thus the degree is 3
    /// assert_eq!(
    ///     vec![1, 2, 0, 4]
    ///         .into_iter()
    ///         .map(Z251::from)
    ///         .collect::<Vec<_>>()
    ///         .degree(),
    ///     3
    /// );
    /// // [1, 1, 1, 1, 9] would be f(x) = 1 + x + x^2 + x^3 + 9x^4
    /// // Thus the degree is 4
    /// assert_eq!(
    ///     vec![1, 1, 1, 1, 9]
    ///         .into_iter()
    ///         .map(Z251::from)
    ///         .collect::<Vec<_>>()
    ///         .degree(),
    ///     4
    /// );
    /// ```
    fn degree(&self) -> usize {
        let tmp = self.iter().rev().skip_while(|&&x| x == T::zero()).count();
        match tmp {
            0 => 0,
            x => x - 1,
        }
    }

    /// Takes the polynomial and evaluates it at the specified value.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use zksnark::field::z251::Z251;
    /// use zksnark::field::*;
    ///
    /// // [1, 1, 1] would be f(x) = 1 + x + x^2 thus f(2) = 1 + 2 + 2^2
    /// assert_eq!(
    ///    vec![1, 1, 1]
    ///        .into_iter()
    ///        .map(Z251::from)
    ///        .collect::<Vec<_>>()
    ///        .evaluate(Z251::from(2)),
    ///    Z251::from(7)
    /// );
    /// // [1, 1, 4] would be f(x) = 1 + x + 4x^2 thus f(2) = 1 + 2 + 4*2^2
    /// assert_eq!(
    ///     vec![1, 1, 4]
    ///         .into_iter()
    ///         .map(Z251::from)
    ///         .collect::<Vec<_>>()
    ///         .evaluate(Z251::from(2)),
    ///     Z251::from(19)
    /// );
    ///
    /// // (1, 2, 3, 4) would be f(x) = 1 + 2x + 3x^2 + 4x^3
    /// // thus f(3) = 1 + 2 * 3 + 3 * 3^2 + 4 * 3^3
    /// assert_eq!(
    ///     (1..5)
    ///         .map(Z251::from)
    ///         .collect::<Vec<_>>()
    ///         .evaluate(Z251::from(3)),
    ///     Z251::from(142)
    /// );
    /// ```
    fn evaluate(&self, x: T) -> T {
        self.coefficients()
            .iter()
            .rev()
            .fold(T::zero(), |acc, y| (acc * x) + *y)
    }
    fn remove_leading_zeros(&mut self) {
        *self = self
            .coefficients()
            .into_iter()
            .rev()
            .skip_while(|&c| c == T::zero())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .into();
    }
}

impl<T> Polynomial<T> for Vec<T> where T: Field {}

fn ext_euc_alg<T>(a: T, b: T) -> (T, T, T)
where
    T: Div<Output = T> + Mul<Output = T> + Sub<Output = T> + Eq + FieldIdentity + Copy,
{
    let (ref mut r0, ref mut r1) = (a, b);
    let (ref mut s0, ref mut s1) = (T::one(), T::zero());
    let (ref mut t0, ref mut t1) = (T::zero(), T::one());

    let (mut r, mut s, mut t, mut q): (T, T, T, T);

    while *r1 != T::zero() {
        q = *r0 / *r1;
        r = *r0 - q * (*r1);
        s = *s0 - q * (*s1);
        t = *t0 - q * (*t1);

        *r0 = *r1;
        *r1 = r;
        *s0 = *s1;
        *s1 = s;
        *t0 = *t1;
        *t1 = t;
    }

    (*r0, *s0, *t0)
}

fn chinese_remainder<T>(rems: &[T], moduli: &[T]) -> T
where
    T: Div<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Add<Output = T>
        + Eq
        + FieldIdentity
        + Copy,
{
    let prod = moduli.iter().fold(T::one(), |acc, x| acc * *x);

    moduli
        .iter()
        .map(|x| prod / *x)
        .zip(moduli)
        .map(|(x, a)| {
            let (_, m, _) = ext_euc_alg(x, *a);
            m * x
        }).zip(rems)
        .map(|(a, b)| a * *b)
        .fold(T::zero(), |acc, x| acc + x)
}

/// The devision of two `Polynomial`
///
/// # Examples
///
/// ```
/// use zksnark::field::z251::Z251;
/// use zksnark::field::*;
///
/// let poly: Vec<Z251> = vec![1, 0, 3, 1].into_iter().map(Z251::from).collect();
/// let polyDividend: Vec<Z251> = vec![0, 0, 9, 1].into_iter().map(Z251::from).collect();
///
/// let num: Vec<Z251> = vec![1].into_iter().map(Z251::from).collect();
/// let den: Vec<Z251> = vec![1, 0, 245].into_iter().map(Z251::from).collect();
///
/// assert_eq!(polynomial_division(poly, polyDividend), (num, den));
/// ```
///
pub fn polynomial_division<P, T>(mut poly: P, mut dividend: P) -> (P, P)
where
    P: Polynomial<T>,
    T: Field,
{
    if dividend
        .coefficients()
        .into_iter()
        .skip_while(|&c| c == T::zero())
        .count()
        == 0
    {
        panic!("Dividend must be non-zero");
    }

    if dividend.degree() > poly.degree() {
        return (P::from(vec![T::zero()]), P::from(vec![T::zero()]));
    }

    poly.remove_leading_zeros();
    dividend.remove_leading_zeros();

    let mut q = vec![T::zero(); poly.degree() + 1 - dividend.degree()];
    let mut r = poly.coefficients();
    let d = dividend.degree();
    let c = dividend.coefficients()[d];

    while r.degree() >= d && r.len() != 0 {
        let s = r[r.degree()] / c;
        q[r.degree() - d] = s;
        r.as_mut_slice()
            .iter_mut()
            .rev()
            .skip_while(|&&mut c| c == T::zero())
            .zip(dividend.coefficients().into_iter().map(|a| a * s).rev())
            .for_each(|(r, b)| *r = *r - b);

        r.remove_leading_zeros();
    }

    (q.into(), r.into())
}

/// Yields an infinite list of powers of x starting from x^0.
///
/// ```rust
/// use zksnark::field::z251::Z251;
/// use zksnark::field::*;
///
/// assert_eq!(
///     powers(Z251::from(5)).take(3).collect::<Vec<_>>(),
///     vec![1, 5, 25]
///         .into_iter()
///         .map(Z251::from)
///         .collect::<Vec<_>>()
/// );
///
/// assert_eq!(
///     powers(Z251::from(2)).take(5).collect::<Vec<_>>(),
///     [1, 2, 4, 8, 16]
///         .iter_mut()
///         .map(|x| Z251::from(*x))
///         .collect::<Vec<_>>()
/// );
/// ```
pub fn powers<T>(x: T) -> impl Iterator<Item = T>
where
    T: Field,
{
    use std::iter::once;
    let identity = T::one();

    once(identity).chain(unfold(identity, move |state| {
        *state = *state * x;
        Some(*state)
    }))
}

/// Discrete Fourier Transformation
///
pub fn dft<T>(seq: &[T], root: T) -> Vec<T>
where
    T: Field,
{
    powers(root)
        .take(seq.len())
        .map(|ri| {
            seq.iter()
                .zip(powers(ri))
                .map(|(&a, r)| a * r)
                .fold(T::zero(), |acc, x| acc + x)
        }).collect::<Vec<_>>()
}

/// Inverse Discrete Fourier Transformation
///
pub fn idft<T>(seq: &[T], root: T) -> Vec<T>
where
    T: Field + From<usize>,
{
    powers(root.mul_inv())
        .take(seq.len())
        .map(|ri| {
            seq.iter()
                .zip(powers(ri))
                .map(|(&a, r)| a * r)
                .fold(T::zero(), |acc, x| acc + x)
                * T::from(seq.len()).mul_inv()
        }).collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::z251::*;
    use super::*;

    extern crate quickcheck;
    use self::quickcheck::quickcheck;

    quickcheck! {
        fn field_bits_u8_prop(vec: Vec<u8>) -> bool {
            let field_bits: Vec<Z251> = to_field_bits(&vec);
            vec == from_field_bits(&field_bits)
        }
        fn field_bits_u64_prop(vec: Vec<u64>) -> bool {
            let field_bits: Vec<Z251> = to_field_bits(&vec);
            vec == from_field_bits(&field_bits)
        }
        fn field_bits_i64_prop(vec: Vec<i64>) -> bool {
            let field_bits: Vec<Z251> = to_field_bits(&vec);
            vec == from_field_bits(&field_bits)
        }

        fn polynomial_evaluate_prop(vec: Vec<usize>, eval_at: usize) -> bool {
            let poly: Vec<Z251> = vec.into_iter().map(|x| Z251::from(x % 251)).collect();
            let x: Z251 = Z251::from(eval_at);
            poly.evaluate(x) == poly
                .coefficients()
                .as_slice()
                .iter()
                .zip(powers(x))
                .fold(Z251::zero(), |acc, (&c, x)| acc + c * x)
        }
        fn degree_prop(vec: Vec<usize>) -> bool {
            let poly: Vec<Z251> = vec.into_iter().map(|x| Z251::from(x % 251)).collect();
            let coeffs = poly.coefficients();
            let mut degree = match coeffs.len() {
                0 => 0,
                d => d - 1,
            };

            for c in coeffs.iter().rev() {
                if *c == Z251::zero() && degree != 0 {
                    degree -= 1;
                } else {
                    return degree == poly.degree();
                }
            }

            degree == poly.degree()
        }
    }

    #[test]
    fn powers_test() {
        let root = Z251 { inner: 9 };
        assert_eq!(
            powers(root).take(5).collect::<Vec<_>>(),
            vec![
                Z251 { inner: 1 },
                Z251 { inner: 9 },
                Z251 { inner: 81 },
                Z251 { inner: 227 },
                Z251 { inner: 35 },
            ]
        );
    }

    #[test]
    fn dft_test() {
        // 25 divies 251 - 1 and 5 has order 25 in Z251
        let mut seq = [Z251::zero(); 25];
        seq[0] = 1.into();
        seq[1] = 2.into();
        seq[2] = 3.into();
        let root = 5.into();

        let result = vec![
            6, 86, 169, 189, 203, 131, 237, 118, 115, 91, 248, 177, 8, 48, 34, 136, 177, 203, 125,
            57, 237, 81, 9, 30, 122,
        ].into_iter()
        .map(Z251::from)
        .collect::<Vec<_>>();

        assert_eq!(dft(&seq[..], root), result);
    }

    #[test]
    fn idft_test() {
        // 25 divies 251 - 1 and 5 has order 25 in Z251
        let mut seq = [Z251::zero(); 25];
        seq[0] = 1.into();
        seq[1] = 2.into();
        seq[2] = 3.into();
        let root = 5.into();

        assert_eq!(idft(&dft(&seq[..], root)[..], root), seq.to_vec());
    }

    #[test]
    fn degree_test() {
        let a = [3, 0, 0, 0, 179, 0, 0, 6]
            .iter()
            .map(|&c| Z251::from(c))
            .collect::<Vec<_>>();
        let b = [29, 112, 68]
            .iter()
            .map(|&c| Z251::from(c))
            .collect::<Vec<_>>();
        let c = [3, 0, 0, 0, 179, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0]
            .iter()
            .map(|&c| Z251::from(c))
            .collect::<Vec<_>>();

        assert_eq!(a.degree(), 7);
        assert_eq!(b.degree(), 2);
        assert_eq!(c.degree(), 7);
    }

    #[test]
    fn polynomial_division_test() {
        let a = [3, 0, 0, 0, 179, 0, 0, 6]
            .iter()
            .map(|&c| Z251::from(c))
            .collect::<Vec<_>>();
        let b = [29, 112, 68]
            .iter()
            .map(|&c| Z251::from(c))
            .collect::<Vec<_>>();
        let q = [209, 207, 78, 1, 131, 37]
            .iter()
            .map(|&c| Z251::from(c))
            .collect::<Vec<_>>();
        let r = [217, 207]
            .iter()
            .map(|&c| Z251::from(c))
            .collect::<Vec<_>>();

        assert_eq!((q, r), polynomial_division(a, b));
    }

    #[test]
    #[should_panic]
    fn polynomial_divisionby0_test() {
        let a = [3, 0, 0, 0, 179, 0, 0, 6]
            .iter()
            .map(|&c| Z251::from(c))
            .collect::<Vec<_>>();
        let b = [0, 0, 0, 0, 0, 0, 0, 0]
            .iter()
            .map(|&c| Z251::from(c))
            .collect::<Vec<_>>();

        polynomial_division(a, b);
    }
}
