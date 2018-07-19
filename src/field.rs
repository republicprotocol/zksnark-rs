extern crate itertools;

use self::itertools::unfold;
use std::ops::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Z251 {
    pub inner: u8,
}

pub trait Field:
    Sized
    + Add<Output = Self>
    + Neg<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn mul_inv(self) -> Self;

    fn add_identity() -> Self;
    fn mul_identity() -> Self;
}

impl Add for Z251 {
    type Output = Z251;

    fn add(self, rhs: Z251) -> Self::Output {
        let sum: u16 = self.inner as u16 + rhs.inner as u16;

        Z251 {
            inner: (sum % 251) as u8,
        }
    }
}

impl Neg for Z251 {
    type Output = Z251;

    fn neg(self) -> Self::Output {
        Z251 {
            inner: 251 - self.inner,
        }
    }
}

impl Sub for Z251 {
    type Output = Z251;

    fn sub(self, rhs: Z251) -> Self::Output {
        self + -rhs
    }
}

impl Mul for Z251 {
    type Output = Z251;

    fn mul(self, rhs: Z251) -> Self::Output {
        let product = (self.inner as u16) * (rhs.inner as u16);

        Z251 {
            inner: (product % 251) as u8,
        }
    }
}

impl Div for Z251 {
    type Output = Z251;

    fn div(self, rhs: Z251) -> Self::Output {
        let (_, mut inv, _) = ext_euc_alg(rhs.inner as isize, 251);
        while inv < 0 {
            inv += 251
        }

        self * Z251 { inner: inv as u8 }
    }
}

impl Field for Z251 {
    fn mul_inv(self) -> Self {
        Z251::mul_identity().div(self)
    }

    fn add_identity() -> Self {
        Z251 { inner: 0 }
    }
    fn mul_identity() -> Self {
        Z251 { inner: 1 }
    }
}

impl From<usize> for Z251 {
    fn from(n: usize) -> Self {
        assert!(n < 251);
        Z251 { inner: n as u8 }
    }
}

impl ZeroElement for isize {
    fn zero() -> Self {
        0
    }
}

impl OneElement for isize {
    fn one() -> Self {
        1
    }
}

pub trait ZeroElement {
    fn zero() -> Self;
}

pub trait OneElement {
    fn one() -> Self;
}

pub trait Polynomial<T>: From<Vec<T>>
where
    T: Field + PartialEq + Copy,
{
    fn coefficients(&self) -> Vec<T>;
    fn degree(&self) -> usize {
        degree(&self.coefficients())
    }
    fn evaluate(&self, x: T) -> T {
        self.coefficients()
            .as_slice()
            .iter()
            .zip(powers(x))
            .fold(T::add_identity(), |acc, (&c, x)| acc + c * x)
    }
    fn remove_leading_zeros(&mut self) {
        *self = self.coefficients()
            .into_iter()
            .rev()
            .skip_while(|&c| c == T::add_identity())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .into();
    }
}

impl<T> Polynomial<T> for Vec<T>
where
    T: Field + PartialEq + Copy,
{
    fn coefficients(&self) -> Vec<T> {
        self.clone()
    }
}

fn ext_euc_alg<T>(a: T, b: T) -> (T, T, T)
where
    T: Div<Output = T> + Mul<Output = T> + Sub<Output = T> + Eq + ZeroElement + OneElement + Copy,
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

pub fn chinese_remainder<T>(rems: &[T], moduli: &[T]) -> T
where
    T: Div<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Add<Output = T>
        + Eq
        + ZeroElement
        + OneElement
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
        })
        .zip(rems)
        .map(|(a, b)| a * *b)
        .fold(T::zero(), |acc, x| acc + x)
}

fn degree<T>(coeffs: &[T]) -> usize
where
    T: Field + PartialEq,
{
    let mut degree = match coeffs.len() {
        0 => 0,
        d => d - 1,
    };

    for c in coeffs.iter().rev() {
        if *c == T::add_identity() && degree != 0 {
            degree -= 1;
        } else {
            return degree;
        }
    }

    degree
}

pub fn polynomial_division<P, T>(mut poly: P, mut dividend: P) -> (P, P)
where
    P: Polynomial<T>,
    T: Field + PartialEq + Copy,
{
    if dividend
        .coefficients()
        .into_iter()
        .skip_while(|&c| c == T::add_identity())
        .count() == 0
    {
        panic!("Dividend must be non-zero");
    }

    if dividend.degree() > poly.degree() {
        return (
            P::from(vec![T::add_identity()]),
            P::from(vec![T::add_identity()]),
        );
    }

    poly.remove_leading_zeros();
    dividend.remove_leading_zeros();

    let mut q = vec![T::add_identity(); poly.degree() + 1 - dividend.degree()];
    let mut r = poly.coefficients();
    let d = dividend.degree();
    let c = dividend.coefficients()[d];

    while degree(&r) >= d && r.len() != 0 {
        let s = r[degree(&r)] / c;
        q[degree(&r) - d] = s;
        r.as_mut_slice()
            .iter_mut()
            .rev()
            .skip_while(|&&mut c| c == T::add_identity())
            .zip(dividend.coefficients().into_iter().map(|a| a * s).rev())
            .for_each(|(r, b)| *r = *r - b);

        r.remove_leading_zeros();
    }

    (q.into(), r.into())
}

pub fn powers<T>(x: T) -> impl Iterator<Item = T>
where
    T: Field + Copy,
{
    use std::iter::once;
    let identity = T::mul_identity();

    once(identity).chain(unfold(identity, move |state| {
        *state = *state * x;
        Some(*state)
    }))
}

pub fn dft<T>(seq: &[T], root: T) -> Vec<T>
where
    T: Field + Copy,
{
    powers(root)
        .take(seq.len())
        .map(|ri| {
            seq.iter()
                .zip(powers(ri))
                .map(|(&a, r)| a * r)
                .fold(T::add_identity(), |acc, x| acc + x)
        })
        .collect::<Vec<_>>()
}

pub fn idft<T>(seq: &[T], root: T) -> Vec<T>
where
    T: Field + Copy + From<usize>,
{
    powers(root.mul_inv())
        .take(seq.len())
        .map(|ri| {
            seq.iter()
                .zip(powers(ri))
                .map(|(&a, r)| a * r)
                .fold(T::add_identity(), |acc, x| acc + x)
                * T::from(seq.len()).mul_inv()
        })
        .collect::<Vec<_>>()
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
    let mut seq = [Z251::add_identity(); 25];
    seq[0] = 1.into();
    seq[1] = 2.into();
    seq[2] = 3.into();
    let root = 5.into();

    let result = vec![
        6, 86, 169, 189, 203, 131, 237, 118, 115, 91, 248, 177, 8, 48, 34, 136, 177, 203, 125, 57,
        237, 81, 9, 30, 122,
    ].into_iter()
        .map(Z251::from)
        .collect::<Vec<_>>();

    assert_eq!(dft(&seq[..], root), result);
}

#[test]
fn idft_test() {
    // 25 divies 251 - 1 and 5 has order 25 in Z251
    let mut seq = [Z251::add_identity(); 25];
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
