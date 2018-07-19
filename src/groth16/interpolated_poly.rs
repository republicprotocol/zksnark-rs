use super::super::field::*;
use super::dummy_poly::DummyPoly;
use std::iter::{repeat, Sum};
use std::ops::*;

#[derive(Clone, Debug, PartialEq)]
pub struct InterpolatedPoly {
    pub nonzero_points: Vec<(Z251, Z251)>,
}

impl InterpolatedPoly {
    #[cfg(test)]
    fn is_zero(&self) -> bool {
        self.nonzero_points.len() == 0
    }
}

impl From<Vec<Z251>> for InterpolatedPoly {
    fn from(cs: Vec<Z251>) -> Self {
        if cs.len() > 25 {
            panic!("Degree of polynomial must be less than 25");
        }

        let coeffs = cs.into_iter()
            .chain(repeat(0.into()))
            .take(25)
            .collect::<Vec<_>>();

        let points = dft(&coeffs, 5.into());

        let nonzero_points = powers(Z251::from(5))
            .zip(points.into_iter())
            .filter(|&(_, c)| c != 0.into())
            .collect::<Vec<_>>();

        InterpolatedPoly { nonzero_points }
    }
}

impl Polynomial<Z251> for InterpolatedPoly {
    fn coefficients(&self) -> Vec<Z251> {
        let mut nz_points = self.nonzero_points.clone();
        let points = powers(Z251::from(5))
            .take(25)
            .map(|p| match nz_points.first() {
                Some(&(x, _)) => {
                    if x == p {
                        let (_, y) = nz_points.remove(0);
                        y
                    } else {
                        0.into()
                    }
                }
                _ => 0.into(),
            })
            .collect::<Vec<_>>();

        idft(&points, 5.into())
    }
}

impl Add for InterpolatedPoly {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (DummyPoly::from(self.coefficients()) + DummyPoly::from(rhs.coefficients()))
            .coefficients()
            .into()
    }
}

impl Neg for InterpolatedPoly {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.nonzero_points.as_mut_slice().iter_mut().for_each(|(_, y)| *y = -*y);
        self
    }
}

impl Sub for InterpolatedPoly {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (DummyPoly::from(self.coefficients()) - DummyPoly::from(rhs.coefficients()))
            .coefficients()
            .into()
    }
}

impl Mul<Z251> for InterpolatedPoly {
    type Output = Self;

    fn mul(self, rhs: Z251) -> Self::Output {
        (DummyPoly::from(self.coefficients()) * rhs)
            .coefficients()
            .into()
    }
}

impl Mul for InterpolatedPoly {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // let mut self_points = self.nonzero_points.clone();
        // let mut rhs_points = rhs.nonzero_points.clone();
        // let nonzero_points = powers(Z251::from(5))
        //     .take(25)
        //     .map(|p| match (self_points.first(), rhs_points.first()) {
        //         (Some(&(x, _)), Some(&(y, _))) => {
        //             if x == p && y == p {
        //                 let ((_, x), (_, y)) = (self_points.remove(0), rhs_points.remove(0));
        //                 (p, x * y)
        //             } else {
        //                 (p, 0.into())
        //             }
        //         }
        //         _ => (p, 0.into()),
        //     })
        //     .filter(|&(_, y)| y != 0.into())
        //     .collect::<Vec<_>>();

        // InterpolatedPoly { nonzero_points }

        (DummyPoly::from(self.coefficients()) * DummyPoly::from(rhs.coefficients()))
            .coefficients()
            .into()
    }
}

impl Div for InterpolatedPoly {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        (DummyPoly::from(self.coefficients()) / DummyPoly::from(rhs.coefficients()))
            .coefficients()
            .into()
    }
}

impl Sum for InterpolatedPoly {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(
            InterpolatedPoly {
                nonzero_points: vec![],
            },
            |acc, x| acc + x,
        )
    }
}

#[cfg(test)]
use super::super::encryption::*;
use super::*;

#[test]
fn interpolated_from_vec() {
    let mut coeffs: Vec<Z251> = Vec::new();
    let mut inter_poly: InterpolatedPoly;

    for _ in 0..100 {
        for _ in 0..25 {
            coeffs.push(Z251::random());
        }
        inter_poly = coeffs.clone().into();

        for i in 0..25 {
            assert_eq!(inter_poly.evaluate(i.into()), coeffs.evaluate(i.into()));
        }

        coeffs.clear();
    }
}

#[test]
fn interpolated_add() {
    // Trivial addition
    let a = InterpolatedPoly::from(vec![]);
    let b = InterpolatedPoly::from(vec![]);
    assert!((a + b).is_zero());

    // Addition with one trivial term
    let a = InterpolatedPoly::from(vec![]);
    let b = vec![Z251::from(1), Z251::from(2), Z251::from(3)].into();
    assert_eq!(
        a + b,
        vec![Z251::from(1), Z251::from(2), Z251::from(3)].into()
    );

    // Addition with one zero term
    let a = InterpolatedPoly::from(vec![Z251::from(0)]);
    let b = vec![Z251::from(1), Z251::from(2), Z251::from(3)].into();
    assert_eq!(
        a + b,
        vec![Z251::from(1), Z251::from(2), Z251::from(3)].into()
    );

    // Addition with leading zeros
    let a = InterpolatedPoly::from(vec![Z251::from(4), Z251::from(5), Z251::from(6)]);
    let b = vec![Z251::from(1), Z251::from(2), Z251::from(3), Z251::from(0)].into();
    assert_eq!(
        a + b,
        vec![Z251::from(5), Z251::from(7), Z251::from(9), Z251::from(0)].into()
    );

    // Addition with overflow
    let a = InterpolatedPoly::from(vec![Z251::from(234), Z251::from(100), Z251::from(6)]);
    let b = vec![Z251::from(123), Z251::from(234), Z251::from(3)].into();
    assert_eq!(
        a + b,
        vec![Z251::from(106), Z251::from(83), Z251::from(9)].into()
    );
}

#[test]
fn interpolated_neg() {
    // Check to see if the negative is the additive inverse
    // Generate random quadratic polynomials

    for _ in 0..1000 {
        let a = InterpolatedPoly::from(vec![
            Z251::random_elem(),
            Z251::random_elem(),
            Z251::random_elem(),
        ]);
        let b = -a.clone();
        assert!((a + b).is_zero());
    }
}

#[test]
fn interpolated_sub() {
    // Check that if c = a - b then a = b + c
    for _ in 0..1000 {
        let a = InterpolatedPoly::from(vec![
            Z251::random_elem(),
            Z251::random_elem(),
            Z251::random_elem(),
        ]);
        let b = InterpolatedPoly::from(vec![
            Z251::random_elem(),
            Z251::random_elem(),
            Z251::random_elem(),
        ]);
        let c = a.clone() - b.clone();
        assert_eq!(a, b + c);
    }
}

#[test]
fn interpolated_sum() {
    let mut polys = Vec::with_capacity(20);
    let mut sum: InterpolatedPoly;

    for _ in 0..100 {
        polys.clear();
        sum = vec![Z251::add_identity(); 3].into();

        for _ in 0..20 {
            let a = InterpolatedPoly::from(vec![
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
            ]);
            polys.push(a.clone());
            sum = sum + a;
        }

        assert_eq!(sum, polys.clone().into_iter().sum());
    }
}

#[test]
fn interpolated_mul() {
    // Trivial multiplication
    let a = InterpolatedPoly::from(vec![]);
    let b = InterpolatedPoly::from(vec![]);
    assert!((a * b).is_zero());

    // Multiplication with one trivial term
    let a = InterpolatedPoly::from(vec![]);
    let b = InterpolatedPoly::from(vec![Z251::from(1), Z251::from(2), Z251::from(3)]);
    assert!((a * b).is_zero());

    // Multiplication with one zero term
    let a = InterpolatedPoly::from(vec![Z251::from(0)]);
    let b = InterpolatedPoly::from(vec![Z251::from(1), Z251::from(2), Z251::from(3)]);
    assert!((a * b).is_zero());

    // Multiplication with leading zeros
    let a = InterpolatedPoly::from(vec![Z251::from(4), Z251::from(5), Z251::from(6)]);
    let b = InterpolatedPoly::from(vec![
        Z251::from(1),
        Z251::from(2),
        Z251::from(3),
        Z251::from(0),
    ]);
    assert_eq!(
        a * b,
        vec![
            Z251::from(4),
            Z251::from(13),
            Z251::from(28),
            Z251::from(27),
            Z251::from(18),
        ].into()
    );

    // Multiplication with overflow
    let a = InterpolatedPoly::from(vec![Z251::from(234), Z251::from(100), Z251::from(6)]);
    let b = InterpolatedPoly::from(vec![Z251::from(123), Z251::from(234), Z251::from(3)]);
    assert_eq!(
        a * b,
        vec![
            Z251::from(168),
            Z251::from(39),
            Z251::from(242),
            Z251::from(198),
            Z251::from(18),
        ].into()
    );
}

#[test]
fn interpolated_scalar_mul() {
    // Scalar multiplication with trivial polynomial
    let a = InterpolatedPoly::from(vec![]);
    let s = Z251::from(69);
    assert!((a * s).is_zero());

    // Scalar multiplication with zero polynomial
    let a = InterpolatedPoly::from(vec![0.into()]);
    let s = Z251::from(69);
    assert!((a * s).is_zero());

    // Scalar multiplication with non-zero polynomial
    let a = InterpolatedPoly::from(vec![Z251::from(1), Z251::from(2), Z251::from(3)]);
    let s = Z251::from(69);
    assert_eq!(
        a * s,
        InterpolatedPoly::from(vec![Z251::from(69), Z251::from(138), Z251::from(207)])
    );

    // Scalar multiplication with overflow
    let a = InterpolatedPoly::from(vec![Z251::from(20), Z251::from(2), Z251::from(3)]);
    let s = Z251::from(69);
    assert_eq!(
        a * s,
        InterpolatedPoly::from(vec![Z251::from(125), Z251::from(138), Z251::from(207)])
    );

    // Scalar multiplication zero scalar
    let a = InterpolatedPoly::from(vec![Z251::from(20), Z251::from(2), Z251::from(3)]);
    let s = Z251::from(0);
    assert!((a * s).is_zero());
}

#[test]
fn interpolated_div() {
    // Check that if a * b = c then a = c / b

    for _ in 0..1000 {
        let mut a = InterpolatedPoly::from(vec![
            Z251::random_elem(),
            Z251::random_elem(),
            Z251::random_elem(),
        ]);
        let b = InterpolatedPoly::from(vec![
            Z251::random_elem(),
            Z251::random_elem(),
            Z251::random_elem(),
        ]);
        if b.is_zero() {
            continue;
        }
        a.remove_leading_zeros();
        let c = a.clone() * b.clone();

        assert_eq!(a, c / b);
    }
}