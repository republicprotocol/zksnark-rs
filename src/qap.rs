extern crate itertools;

use self::itertools::unfold;
use super::encryption::*;
use super::field::*;
use std::collections::HashMap;
use std::ops::*;

#[derive(Clone, PartialEq, Eq)]
struct InterpolatedPoly {
    nonzero_points: Vec<(Z251, Z251)>,
}

impl InterpolatedPoly {
    fn evaluate(&self, p: Z251, roots: &[Z251]) -> Z251 {
        self.nonzero_points
            .clone()
            .into_iter()
            .fold(Z251::add_identity(), |acc, (x, y)| {
                acc + roots
                    .iter()
                    .filter(|&&z| z != x)
                    .fold(y, |acc, &r| acc * (p - r) / (x - r))
            })
    }

    fn set_value(&mut self, p: (Z251, Z251)) {
        self.nonzero_points.push(p);
    }

    fn coefficients(&self) -> Vec<Z251> {
        let mut points = HashMap::new();

        let roots = powers(Z251::from(5)).take(25);

        for root in roots {
            points.insert(root, Z251::from(0));
        }
        for (x, y) in self.nonzero_points.clone() {
            if let Some(value) = points.get_mut(&x) {
                *value = *value + y;
            }
        }

        idft(
            powers(Z251::from(5))
                .take(25)
                .map(|x| *points.get(&x).unwrap())
                .collect::<Vec<_>>()
                .as_slice(),
            5.into(),
        )
    }
}

impl Add for InterpolatedPoly {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut points: HashMap<_, Z251> = HashMap::new();
        for (x, y) in self.nonzero_points
            .into_iter()
            .chain(rhs.nonzero_points.into_iter())
        {
            let value = points.entry(x).or_insert(0.into());
            *value = *value + y;
        }

        let nonzero_points = points.drain().collect::<Vec<_>>();

        InterpolatedPoly { nonzero_points }
    }
}

impl Mul<Z251> for InterpolatedPoly {
    type Output = Self;

    fn mul(self, rhs: Z251) -> Self::Output {
        let nonzero_points = self.nonzero_points
            .into_iter()
            .map(|(x, y)| (x, rhs * y))
            .collect::<Vec<_>>();

        InterpolatedPoly { nonzero_points }
    }
}

struct QAP {
    u: Vec<InterpolatedPoly>,
    v: Vec<InterpolatedPoly>,
    w: Vec<InterpolatedPoly>,
    t: InterpolatedPoly,
    input: usize,
    degree: usize,
    size: usize,
}

struct SigmaG1 {
    alpha: Z251,
    beta: Z251,
    delta: Z251,
    xi: Vec<Z251>,
    sum_gamma: Vec<Z251>,
    sum_delta: Vec<Z251>,
    xi_t: Vec<Z251>,
}

struct SigmaG2 {
    beta: Z251,
    gamma: Z251,
    delta: Z251,
    xi: Vec<Z251>,
}

struct Proof {
    a: Z251,
    b: Z251,
    c: Z251,
}

fn setup(qap: QAP) -> (SigmaG1, SigmaG2) {
    let (alpha, beta, gamma, delta, x) = (
        Z251::random(),
        Z251::random(),
        Z251::random(),
        Z251::random(),
        Z251::random(),
    );
    let mut xi = vec![Z251::mul_identity()];
    let mut current = Z251::mul_identity();
    for _ in 0..qap.degree {
        current = current * x;
        xi.append(&mut vec![current]);
    }

    let roots = powers(5.into()).take(25).collect::<Vec<Z251>>();

    let mut sum_gamma = Vec::new();
    for i in 0..qap.input {
        sum_gamma.append(&mut vec![
            (beta * qap.u[i].evaluate(x, &roots)
                + alpha * qap.v[i].evaluate(x, &roots)
                + qap.w[i].evaluate(x, &roots)) / gamma,
        ]);
    }

    let mut sum_delta = Vec::new();
    for i in qap.input..qap.size {
        sum_delta.append(&mut vec![
            (beta * qap.u[i].evaluate(x, &roots)
                + alpha * qap.v[i].evaluate(x, &roots)
                + qap.w[i].evaluate(x, &roots)) / delta,
        ]);
    }

    let mut xi_t = Vec::new();
    for _ in 0..qap.degree - 1 {
        xi_t.append(&mut vec![xi[0] * qap.t.evaluate(x, &roots) / delta]);
    }

    let alpha = alpha.encrypt();
    let beta = beta.encrypt();
    let gamma = gamma.encrypt();
    let delta = delta.encrypt();
    xi = xi.clone().into_iter().map(|x| x.encrypt()).collect();
    sum_delta = sum_delta.clone().into_iter().map(|x| x.encrypt()).collect();
    sum_gamma = sum_gamma.clone().into_iter().map(|x| x.encrypt()).collect();
    xi_t = xi_t.clone().into_iter().map(|x| x.encrypt()).collect();

    let sigmag1 = SigmaG1 {
        alpha,
        beta,
        delta,
        xi: xi.clone(),
        sum_delta,
        sum_gamma,
        xi_t,
    };
    let sigmag2 = SigmaG2 {
        beta,
        gamma,
        delta,
        xi,
    };

    (sigmag1, sigmag2)
}

fn prove(sigmag1: SigmaG1, sigmag2: SigmaG2, qap: QAP, weights: Vec<Z251>) -> Proof {
    assert_eq!(qap.u.len(), weights.len());
    assert_eq!(qap.v.len(), weights.len());
    assert_eq!(qap.w.len(), weights.len());

    let roots = powers(5.into()).take(25).collect::<Vec<Z251>>();
    let (r, s) = (Z251::random(), Z251::random());

    let u_combin = qap.u
        .into_iter()
        .zip(weights.clone().into_iter())
        .fold(
            InterpolatedPoly {
                nonzero_points: Vec::new(),
            },
            |acc, (x, a)| acc + x * a,
        )
        .coefficients();
    let v_combin = qap.v
        .into_iter()
        .zip(weights.clone().into_iter())
        .fold(
            InterpolatedPoly {
                nonzero_points: Vec::new(),
            },
            |acc, (x, a)| acc + x * a,
        )
        .coefficients();

    let a = sigmag1.alpha
        + u_combin
            .into_iter()
            .zip(sigmag1.xi.into_iter())
            .fold(0.into(), |acc, (x, a)| acc + x * a) + r * sigmag1.delta;
    let b = sigmag2.beta
        + v_combin
            .into_iter()
            .zip(sigmag2.xi.into_iter())
            .fold(0.into(), |acc, (x, a)| acc + x * a) + s * sigmag2.delta;

    unimplemented!()
}

#[test]
fn evaluate() {
    // 6 x^7 + 179 x^4 + 3
    let points = (0..9)
        .zip([3, 188, 121, 14, 57, 65, 238, 142].into_iter().map(|&x| x))
        .collect::<Vec<(usize, usize)>>();
    let poly = InterpolatedPoly {
        nonzero_points: points
            .iter()
            .map(|&(x, y)| (x.into(), y.into()))
            .collect::<Vec<_>>(),
    };
    let roots = (0..9).map(|r| r.into()).collect::<Vec<_>>();

    assert_eq!(poly.evaluate(Z251 { inner: 0 }, &roots), Z251 { inner: 3 });
    assert_eq!(
        poly.evaluate(Z251 { inner: 1 }, &roots),
        Z251 { inner: 188 }
    );
    assert_eq!(
        poly.evaluate(Z251 { inner: 2 }, &roots),
        Z251 { inner: 121 }
    );
    assert_eq!(poly.evaluate(Z251 { inner: 3 }, &roots), Z251 { inner: 14 });
    assert_eq!(poly.evaluate(Z251 { inner: 4 }, &roots), Z251 { inner: 57 });
    assert_eq!(poly.evaluate(Z251 { inner: 5 }, &roots), Z251 { inner: 65 });
    assert_eq!(
        poly.evaluate(Z251 { inner: 6 }, &roots),
        Z251 { inner: 238 }
    );
    assert_eq!(
        poly.evaluate(Z251 { inner: 7 }, &roots),
        Z251 { inner: 142 }
    );
}

#[test]
fn poly_add_test() {
    let points1 = (0..9)
        .zip([1, 100, 0, 14, 0, 66, 108, 0].into_iter().map(|&x| x))
        .collect::<Vec<(usize, usize)>>();
    let poly1 = InterpolatedPoly {
        nonzero_points: points1
            .iter()
            .map(|&(x, y)| (x.into(), y.into()))
            .collect::<Vec<_>>(),
    };
    let points2 = (0..9)
        .zip([2, 88, 121, 0, 57, 250, 130, 142].into_iter().map(|&x| x))
        .collect::<Vec<(usize, usize)>>();
    let poly2 = InterpolatedPoly {
        nonzero_points: points2
            .iter()
            .map(|&(x, y)| (x.into(), y.into()))
            .collect::<Vec<_>>(),
    };
    let points = (0..9)
        .zip([3, 188, 121, 14, 57, 65, 238, 142].into_iter().map(|&x| x))
        .collect::<Vec<(usize, usize)>>();
    let poly = InterpolatedPoly {
        nonzero_points: points
            .iter()
            .map(|&(x, y)| (x.into(), y.into()))
            .collect::<Vec<_>>(),
    };

    let sum = poly1 + poly2;

    for i in 0..9 {
        assert_eq!(
            sum.evaluate(i.into(), &(0..9).map(|x| x.into()).collect::<Vec<_>>()[..]),
            poly.evaluate(i.into(), &(0..9).map(|x| x.into()).collect::<Vec<_>>()[..])
        );
    }
}

#[test]
fn coefficients_test() {
    // 6 x^7 + 179 x^4 + 3
    let nonzero_points = powers(5.into())
        .take(25)
        .zip(
            [
                188, 65, 33, 222, 74, 128, 100, 241, 40, 73, 188, 116, 32, 163, 176, 224, 236, 120,
                177, 57, 40, 0, 91, 166, 137,
            ].into_iter()
                .map(|&x| x.into()),
        )
        .collect::<Vec<(Z251, Z251)>>();
    let poly = InterpolatedPoly { nonzero_points };

    assert_eq!(
        poly.coefficients(),
        vec![
            3, 0, 0, 0, 179, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ].into_iter()
            .map(|x| x.into())
            .collect::<Vec<Z251>>()
    );
}
