use self::circuit::*;
use self::coefficient_poly::{root_poly, CoefficientPoly};
use super::field::z251::Z251;
use super::field::*;
use std::iter::{once, repeat, Sum};
use std::ops::{Add, Div, Mul, Neg, Sub};

mod circuit;
mod coefficient_poly;

pub trait Random {
    fn random_elem() -> Self;
}

pub trait EllipticEncryptable {
    type G1;
    type G2;
    type GT;

    fn encrypt_g1(self) -> Self::G1;
    fn encrypt_g2(self) -> Self::G2;
    fn exp_encrypted_g1(self, Self::G1) -> Self::G1;
    fn exp_encrypted_g2(self, Self::G2) -> Self::G2;
    fn pairing(Self::G1, Self::G2) -> Self::GT;
}

pub trait Identity {
    fn is_identity(&self) -> bool;
}

pub struct QAP<P> {
    u: Vec<P>,
    v: Vec<P>,
    w: Vec<P>,
    t: P,
    input: usize,
    degree: usize,
}

impl<T> From<T> for QAP<CoefficientPoly<Z251>>
where
    T: RootRepresentation<Z251>,
{
    fn from(root_rep: T) -> Self {
        let (mut u, mut v, mut w) = (Vec::new(), Vec::new(), Vec::new());

        for points in root_rep.u() {
            u.push(CoefficientPoly::from((root_rep.roots(), points)));
        }
        for points in root_rep.v() {
            v.push(CoefficientPoly::from((root_rep.roots(), points)));
        }
        for points in root_rep.w() {
            w.push(CoefficientPoly::from((root_rep.roots(), points)));
        }

        assert_eq!(u.len(), v.len());
        assert_eq!(u.len(), w.len());

        let t = root_poly(root_rep.roots());
        let input = root_rep.input();
        let degree = t.degree();

        QAP {
            u,
            v,
            w,
            t,
            input,
            degree,
        }
    }
}

pub struct SigmaG1<T> {
    alpha: T,
    beta: T,
    delta: T,
    xi: Vec<T>,
    sum_gamma: Vec<T>,
    sum_delta: Vec<T>,
    xi_t: Vec<T>,
}

pub struct SigmaG2<T> {
    beta: T,
    gamma: T,
    delta: T,
    xi: Vec<T>,
}

pub struct Proof<U, V> {
    a: U,
    b: V,
    c: U,
}

pub fn setup<P, T, U, V>(qap: &QAP<P>) -> (SigmaG1<U>, SigmaG2<V>)
where
    P: Add + Polynomial<T>,
    T: EllipticEncryptable<G1 = U, G2 = V> + Random + Field + Copy + PartialEq,
{
    let (alpha, beta, gamma, delta, x) = (
        T::random_elem(),
        T::random_elem(),
        T::random_elem(),
        T::random_elem(),
        T::random_elem(),
    );
    let xi = powers(x).take(qap.degree).collect::<Vec<_>>();
    let sum_gamma = qap.u
        .as_slice()
        .iter()
        .zip(qap.v.as_slice().iter().zip(qap.w.as_slice().iter()))
        .map(|(ui, (vi, wi))| {
            ((beta * ui.evaluate(x) + alpha * vi.evaluate(x) + wi.evaluate(x)) / gamma).encrypt_g1()
        })
        .take(qap.input + 1)
        .collect::<Vec<_>>();
    let sum_delta = qap.u
        .as_slice()
        .iter()
        .zip(qap.v.as_slice().iter().zip(qap.w.as_slice().iter()))
        .map(|(ui, (vi, wi))| {
            ((beta * ui.evaluate(x) + alpha * vi.evaluate(x) + wi.evaluate(x)) / delta).encrypt_g1()
        })
        .skip(qap.input + 1)
        .collect::<Vec<_>>();
    let xi_t = xi.as_slice()
        .iter()
        .take(xi.len() - 1)
        .map(|&i| ((i * qap.t.evaluate(x)) / delta).encrypt_g1())
        .collect::<Vec<_>>();

    let sigmag1 = SigmaG1 {
        alpha: alpha.encrypt_g1(),
        beta: beta.encrypt_g1(),
        delta: delta.encrypt_g1(),
        xi: xi.as_slice()
            .iter()
            .map(|&i| i.encrypt_g1())
            .collect::<Vec<_>>(),
        sum_delta,
        sum_gamma,
        xi_t,
    };
    let sigmag2 = SigmaG2 {
        beta: beta.encrypt_g2(),
        gamma: gamma.encrypt_g2(),
        delta: delta.encrypt_g2(),
        xi: xi.as_slice()
            .iter()
            .map(|&i| i.encrypt_g2())
            .collect::<Vec<_>>(),
    };

    (sigmag1, sigmag2)
}

pub fn prove<P, T, U, V>(
    qap: &QAP<P>,
    (sigmag1, sigmag2): (&SigmaG1<U>, &SigmaG2<V>),
    weights: &[T],
) -> Proof<U, V>
where
    P: Add
        + Sub<Output = P>
        + Mul<T, Output = P>
        + Mul<Output = P>
        + Div<Output = P>
        + Polynomial<T>
        + Sum
        + Clone,
    T: EllipticEncryptable<G1 = U, G2 = V> + Random + Field + Copy + PartialEq,
    U: Add<Output = U> + Sub<Output = U> + Sum + Copy,
    V: Add<Output = V> + Sum + Copy,
{
    let (r, s) = (T::random_elem(), T::random_elem());

    let u_sum = qap.u
        .clone()
        .into_iter()
        .zip(weights.iter())
        .map(|(p, &a)| p * a)
        .sum::<P>();
    let v_sum = qap.v
        .clone()
        .into_iter()
        .zip(weights.iter())
        .map(|(p, &a)| p * a)
        .sum::<P>();
    let w_sum = qap.w
        .clone()
        .into_iter()
        .zip(weights.iter())
        .map(|(p, &a)| p * a)
        .sum::<P>();

    let a_g1 = u_sum
        .coefficients()
        .into_iter()
        .zip(sigmag1.xi.as_slice().iter())
        .map(|(a, &x)| a.exp_encrypted_g1(x))
        .sum::<U>();
    let b_g1 = v_sum
        .coefficients()
        .into_iter()
        .zip(sigmag1.xi.as_slice().iter())
        .map(|(a, &x)| a.exp_encrypted_g1(x))
        .sum::<U>();
    let b_g2 = v_sum
        .coefficients()
        .into_iter()
        .zip(sigmag2.xi.as_slice().iter())
        .map(|(a, &x)| a.exp_encrypted_g2(x))
        .sum::<V>();

    let a = a_g1 + sigmag1.alpha + r.exp_encrypted_g1(sigmag1.delta);
    let b = b_g2 + sigmag2.beta + s.exp_encrypted_g2(sigmag2.delta);

    let h = (u_sum * v_sum - w_sum) / qap.t.clone();

    let c = h.coefficients()
        .into_iter()
        .zip(sigmag1.xi_t.clone().into_iter())
        .map(|(c, x)| c.exp_encrypted_g1(x))
        .sum::<U>()
        + weights
            .iter()
            .skip(qap.input + 1)
            .zip(sigmag1.sum_delta.clone().into_iter())
            .map(|(c, x)| c.exp_encrypted_g1(x))
            .sum::<U>() + s.exp_encrypted_g1(a)
        + r.exp_encrypted_g1(sigmag1.beta + b_g1 + s.exp_encrypted_g1(sigmag1.delta))
        - (r * s).exp_encrypted_g1(sigmag1.delta);

    Proof { a, b, c }
}

pub fn verify<P, T, U, V, W>(
    _qap: &QAP<P>,
    (sigmag1, sigmag2): (SigmaG1<U>, SigmaG2<V>),
    inputs: &[T],
    proof: Proof<U, V>,
) -> bool
where
    T: Field + Copy + EllipticEncryptable<G1 = U, G2 = V, GT = W>,
    U: Sum,
    W: Add<Output = W> + PartialEq,
{
    let sum_term = sigmag1
        .sum_gamma
        .into_iter()
        .zip(once(T::mul_identity()).chain(inputs.iter().map(|&x| x)))
        .map(|(x, a)| a.exp_encrypted_g1(x))
        .sum::<U>();

    T::pairing(sigmag1.alpha, sigmag2.beta)
        + T::pairing(sum_term, sigmag2.gamma)
        + T::pairing(proof.c, sigmag2.delta) == T::pairing(proof.a, proof.b)
}

#[cfg(test)]
mod tests {
    extern crate bn;
    extern crate rand;

    use self::bn::*;
    use self::circuit::dummy_rep::DummyRep;
    use super::super::encryption::Encryptable;
    use super::*;
    use std::str::FromStr;
    use std::time::{Duration, Instant};

    impl Random for Z251 {
        fn random_elem() -> Self {
            let mut r = Z251::random();
            while r == Z251::add_identity() {
                r = Z251::random();
            }
            r
        }
    }

    impl EllipticEncryptable for Z251 {
        type G1 = Self;
        type G2 = Self;
        type GT = Self;

        fn encrypt_g1(self) -> Self::G1 {
            self * 69.into()
        }
        fn encrypt_g2(self) -> Self::G2 {
            self * 69.into()
        }
        fn exp_encrypted_g1(self, g1: Self::G1) -> Self::G1 {
            self * g1
        }
        fn exp_encrypted_g2(self, g2: Self::G2) -> Self::G2 {
            self * g2
        }
        fn pairing(g1: Self::G1, g2: Self::G2) -> Self::GT {
            g1 * g2
        }
    }

    impl Identity for Z251 {
        fn is_identity(&self) -> bool {
            *self == Self::add_identity()
        }
    }

    impl Sum for Z251 {
        fn sum<I>(iter: I) -> Self
        where
            I: Iterator<Item = Self>,
        {
            iter.fold(Z251::from(0), |acc, x| acc + x)
        }
    }

    pub fn constant<T>(c: usize) -> CoefficientPoly<T>
    where
        T: From<usize>,
    {
        vec![c.into()].into()
    }

    #[test]
    fn single_mult_honest() {
        let qap: QAP<CoefficientPoly<Z251>> = QAP {
            u: vec![constant(0), constant(0), constant(1), constant(0)],
            v: vec![constant(0), constant(0), constant(0), constant(1)],
            w: vec![constant(0), constant(1), constant(0), constant(0)],
            t: vec![Z251::from(250), Z251::from(1)].into(),
            input: 2,
            degree: 1,
        };
        let weights: Vec<Z251> = vec![1.into(), 17.into(), 100.into(), 83.into()];

        for _ in 0..1000 {
            let (sigmag1, sigmag2) = setup(&qap);

            let alpha = sigmag1.alpha / 69.into();
            let beta = sigmag1.beta / 69.into();
            let gamma = sigmag2.gamma / 69.into();
            let delta = sigmag1.delta / 69.into();

            // sigmag1 tests
            assert_eq!(sigmag1.xi.len(), 1);
            assert_eq!(sigmag1.xi[0], Z251::from(1).encrypt_g1());
            assert_eq!(sigmag1.sum_gamma.len(), 3);
            assert_eq!(sigmag1.sum_gamma[0], Z251::from(0).encrypt_g1());
            assert_eq!(sigmag1.sum_gamma[1], (Z251::from(1) / gamma).encrypt_g1());
            assert_eq!(sigmag1.sum_gamma[2], (beta / gamma).encrypt_g1());
            assert_eq!(sigmag1.sum_delta.len(), 1);
            assert_eq!(sigmag1.sum_delta[0], (alpha / delta).encrypt_g1());
            assert_eq!(sigmag1.xi_t.len(), 0);

            // sigmag2 tests
            assert_eq!(sigmag2.xi.len(), 1);
            assert_eq!(sigmag2.xi[0], Z251::from(1).encrypt_g2());

            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);

            assert!(verify(
                &qap,
                (sigmag1, sigmag2),
                &vec![Z251::from(17), Z251::from(100)],
                proof
            ));
        }
    }

    #[test]
    fn single_mult_random_proof() {
        let mut count = 0;
        let total = 10000;

        let qap: QAP<CoefficientPoly<Z251>> = QAP {
            u: vec![constant(0), constant(0), constant(1), constant(0)],
            v: vec![constant(0), constant(0), constant(0), constant(1)],
            w: vec![constant(0), constant(1), constant(0), constant(0)],
            t: vec![Z251::from(250), Z251::from(1)].into(),
            input: 2,
            degree: 1,
        };

        for _ in 0..total {
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = Proof {
                a: Z251::random_elem(),
                b: Z251::random_elem(),
                c: Z251::random_elem(),
            };

            if verify(
                &qap,
                (sigmag1, sigmag2),
                &vec![Z251::from(17), Z251::from(100)],
                proof,
            ) {
                count += 1;
            }
        }

        // A proof has 3 elements, and given any two there always exists
        // exactly one choice for the final element such that the proof
        // will be verified. This means that a random proof should succeed
        // 1 out of every 250 times, or in other words 0.4% of the time in
        // the case of a field with 251 elements.
        //
        // This means that this test can possibly fail, but it is very unlikely.
        let ratio = (count as f64) / (total as f64);
        assert!(ratio > 0.002);
        assert!(ratio < 0.006);
    }

    #[test]
    fn quadratic_share_honest() {
        let qap: QAP<CoefficientPoly<Z251>> = QAP {
            u: [
                [1, 124, 126],
                [0, 127, 125],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ].iter()
                .map(|v| v.iter().map(|&c| c.into()).collect::<Vec<_>>().into())
                .collect::<Vec<_>>(),
            v: [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [3, 123, 126],
                [248, 4, 250],
                [1, 124, 126],
                [248, 4, 250],
                [1, 124, 126],
            ].iter()
                .map(|v| v.iter().map(|&c| c.into()).collect::<Vec<_>>().into())
                .collect::<Vec<_>>(),
            w: [
                [0, 0, 0],
                [0, 0, 0],
                [1, 124, 126],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [3, 123, 126],
                [248, 4, 250],
            ].iter()
                .map(|v| v.iter().map(|&c| c.into()).collect::<Vec<_>>().into())
                .collect::<Vec<_>>(),
            t: [245, 11, 245, 1]
                .iter()
                .map(|&c| c.into())
                .collect::<Vec<_>>()
                .into(),
            input: 2,
            degree: 3,
        };

        for _ in 0..1000 {
            let (x, a, b, c) = (
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
            );
            let share = a * x * x + b * x + c;
            let weights: Vec<Z251> = vec![1.into(), x, share, a, b, c, a * x, x * (a * x + b)];
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);

            assert!(verify(&qap, (sigmag1, sigmag2), &vec![x, share], proof));
        }
    }

    #[test]
    fn quadratic_share_random_proof() {
        let mut count = 0;
        let total = 10000;

        let qap: QAP<CoefficientPoly<Z251>> = QAP {
            u: [
                [1, 124, 126],
                [0, 127, 125],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ].iter()
                .map(|v| v.iter().map(|&c| c.into()).collect::<Vec<_>>().into())
                .collect::<Vec<_>>(),
            v: [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [3, 123, 126],
                [248, 4, 250],
                [1, 124, 126],
                [248, 4, 250],
                [1, 124, 126],
            ].iter()
                .map(|v| v.iter().map(|&c| c.into()).collect::<Vec<_>>().into())
                .collect::<Vec<_>>(),
            w: [
                [0, 0, 0],
                [0, 0, 0],
                [1, 124, 126],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [3, 123, 126],
                [248, 4, 250],
            ].iter()
                .map(|v| v.iter().map(|&c| c.into()).collect::<Vec<_>>().into())
                .collect::<Vec<_>>(),
            t: [245, 11, 245, 1]
                .iter()
                .map(|&c| c.into())
                .collect::<Vec<_>>()
                .into(),
            input: 2,
            degree: 3,
        };

        for _ in 0..total {
            let (x, a, b, c) = (
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
            );
            let share = a * x * x + b * x + c;
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = Proof {
                a: Z251::random_elem(),
                b: Z251::random_elem(),
                c: Z251::random_elem(),
            };

            if verify(&qap, (sigmag1, sigmag2), &vec![x, share], proof) {
                count += 1;
            }
        }

        // A proof has 3 elements, and given any two there always exists
        // exactly one choice for the final element such that the proof
        // will be verified. This means that a random proof should succeed
        // 1 out of every 250 times, or in other words 0.4% of the time in
        // the case of a field with 251 elements.
        //
        // This means that this test can possibly fail, but it is very unlikely.
        let ratio = (count as f64) / (total as f64);
        assert!(ratio > 0.002);
        assert!(ratio < 0.006);
    }

    #[test]
    fn qap_from_roots() {
        let root_rep = DummyRep::<Z251> {
            u: vec![
                vec![(3.into(), 1.into())],
                vec![(1.into(), 1.into()), (2.into(), 1.into())],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            v: vec![
                vec![],
                vec![],
                vec![],
                vec![(1.into(), 1.into())],
                vec![(2.into(), 1.into())],
                vec![(3.into(), 1.into())],
                vec![(2.into(), 1.into())],
                vec![(3.into(), 1.into())],
            ],
            w: vec![
                vec![],
                vec![],
                vec![(3.into(), 1.into())],
                vec![],
                vec![],
                vec![],
                vec![(1.into(), 1.into())],
                vec![(2.into(), 1.into())],
            ],
            roots: vec![1.into(), 2.into(), 3.into()],
            input: 2,
        };

        let qap = root_rep.into();

        for _ in 0..1000 {
            let (x, a, b, c) = (
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
            );
            let share = a * x * x + b * x + c;
            let weights: Vec<Z251> = vec![1.into(), x, share, a, b, c, a * x, x * (a * x + b)];
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);

            assert!(verify(&qap, (sigmag1, sigmag2), &vec![x, share], proof));
        }
    }

    #[test]
    fn qap_from_file() {
        // Quadratic polynomial share
        let code = &*::std::fs::read_to_string("test_programs/quad_share.zk").unwrap();
        let qap = DummyRep::from(code).into();

        for _ in 0..1000 {
            let (x, a, b, c) = (
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
            );
            let share = a * x * x + b * x + c;
            let weights: Vec<Z251> = vec![1.into(), x, share, a, b, c, a * x, x * (a * x + b)];
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);

            assert!(verify(&qap, (sigmag1, sigmag2), &vec![x, share], proof));
        }

        // Cubic polynomial share
        let code = &*::std::fs::read_to_string("test_programs/cubic_share.zk").unwrap();
        let qap = DummyRep::from(code).into();

        for _ in 0..1000 {
            let (x, a, b, c, d) = (
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
            );
            let share = ((a * x + b) * x + c) * x + d;
            let weights: Vec<Z251> = vec![
                1.into(),
                x,
                share,
                a,
                b,
                c,
                d,
                a * x,
                (a * x + b) * x,
                ((a * x + b) * x + c) * x,
            ];
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);

            assert!(verify(&qap, (sigmag1, sigmag2), &vec![x, share], proof));
        }
    }

    #[test]
    fn qap_from_ast() {
        // Quadratic polynomial share
        let root_rep: DummyRep<Z251> = ASTParser::try_parse(&*::std::fs::read_to_string(
            "test_programs/lispesque_quad.zk",
        ).unwrap())
            .unwrap();
        let qap = root_rep.into();

        for _ in 0..1000 {
            let (x, a, b, c) = (
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
            );
            let share = a * x * x + b * x + c;

            // The order of the weights is now determined by
            // the order that the variables appear in the file
            let weights: Vec<Z251> = vec![1.into(), x, share, a * x, a, x * (a * x + b), b, c];
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);

            assert!(verify(&qap, (sigmag1, sigmag2), &vec![x, share], proof));
        }

        // Cubic polynomial share
        let root_rep: DummyRep<Z251> = ASTParser::try_parse(&*::std::fs::read_to_string(
            "test_programs/lispesque_cubic.zk",
        ).unwrap())
            .unwrap();
        let qap = root_rep.into();

        for _ in 0..1000 {
            let (x, a, b, c, d) = (
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
                Z251::random_elem(),
            );
            let share = a * x * x * x + b * x * x + c * x + d;

            // The order of the weights is now determined by
            // the order that the variables appear in the file
            let weights: Vec<Z251> = vec![
                1.into(),
                x,
                share,
                a * x,
                a,
                x * (a * x + b),
                b,
                x * (x * (a * x + b) + c),
                c,
                d,
            ];
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);

            assert!(verify(&qap, (sigmag1, sigmag2), &vec![x, share], proof));
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq)]
    struct FrLocal(Fr);

    #[derive(Clone, Copy, PartialEq)]
    struct G1Local(G1);
    #[derive(Clone, Copy, PartialEq)]
    struct G2Local(G2);
    #[derive(PartialEq)]
    struct GtLocal(Gt);

    impl Add for FrLocal {
        type Output = FrLocal;

        fn add(self, rhs: FrLocal) -> Self::Output {
            FrLocal(self.0 + rhs.0)
        }
    }

    impl Neg for FrLocal {
        type Output = FrLocal;

        fn neg(self) -> Self::Output {
            FrLocal(-self.0)
        }
    }

    impl Sub for FrLocal {
        type Output = FrLocal;

        fn sub(self, rhs: FrLocal) -> Self::Output {
            FrLocal(self.0 - rhs.0)
        }
    }

    impl Mul for FrLocal {
        type Output = FrLocal;

        fn mul(self, rhs: FrLocal) -> Self::Output {
            FrLocal(self.0 * rhs.0)
        }
    }

    impl Div for FrLocal {
        type Output = FrLocal;

        fn div(self, rhs: FrLocal) -> Self::Output {
            FrLocal(self.0 * rhs.0.inverse().expect("Tried to divide by zero"))
        }
    }

    impl Field for FrLocal {
        fn mul_inv(self) -> Self {
            FrLocal(self.0.inverse().expect("Tried to get mul inv of zero"))
        }

        fn add_identity() -> Self {
            FrLocal(Fr::zero())
        }
        fn mul_identity() -> Self {
            FrLocal(Fr::one())
        }
    }

    impl From<usize> for FrLocal {
        fn from(n: usize) -> Self {
            FrLocal(Fr::from_str(n.to_string().as_str()).expect("Could not convert string to Fr"))
        }
    }

    impl FromStr for FrLocal {
        type Err = ();
        
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match Fr::from_str(s) {
                None => Err(()),
                Some(n) => Ok(FrLocal(n)),
            }
        }
    }

    impl Random for FrLocal {
        fn random_elem() -> Self {
            let rng = &mut rand::thread_rng();
            let mut r = Fr::random(rng);
            while r == Fr::zero() {
                r = Fr::random(rng);
            }
            FrLocal(r)
        }
    }

    impl EllipticEncryptable for FrLocal {
        type G1 = G1Local;
        type G2 = G2Local;
        type GT = GtLocal;

        fn encrypt_g1(self) -> Self::G1 {
            let g = G1::one() * Fr::from_str("69").unwrap();
            G1Local(g * self.0)
        }
        fn encrypt_g2(self) -> Self::G2 {
            let g = G2::one() * Fr::from_str("96").unwrap();
            G2Local(g * self.0)
        }
        fn exp_encrypted_g1(self, g1: Self::G1) -> Self::G1 {
            G1Local(g1.0 * self.0)
        }
        fn exp_encrypted_g2(self, g2: Self::G2) -> Self::G2 {
            G2Local(g2.0 * self.0)
        }
        fn pairing(g1: Self::G1, g2: Self::G2) -> Self::GT {
            GtLocal(pairing(g1.0, g2.0))
        }
    }

    #[test]
    fn exp_encrypted_test() {
        for _ in 0..1000 {
            let (a, b) = (FrLocal::random_elem(), FrLocal::random_elem());
            assert!(a.exp_encrypted_g1(b.encrypt_g1()) == (a * b).encrypt_g1());
        }
    }

    impl Identity for FrLocal {
        fn is_identity(&self) -> bool {
            *self == Self::add_identity()
        }
    }

    impl Sum for FrLocal {
        fn sum<I>(iter: I) -> Self
        where
            I: Iterator<Item = Self>,
        {
            iter.fold(FrLocal::add_identity(), |acc, x| acc + x)
        }
    }

    impl<T> From<T> for QAP<CoefficientPoly<FrLocal>>
    where
        T: RootRepresentation<FrLocal>,
    {
        fn from(root_rep: T) -> Self {
            let (mut u, mut v, mut w) = (Vec::new(), Vec::new(), Vec::new());

            for points in root_rep.u() {
                u.push(CoefficientPoly::from((root_rep.roots(), points)));
            }
            for points in root_rep.v() {
                v.push(CoefficientPoly::from((root_rep.roots(), points)));
            }
            for points in root_rep.w() {
                w.push(CoefficientPoly::from((root_rep.roots(), points)));
            }

            assert_eq!(u.len(), v.len());
            assert_eq!(u.len(), w.len());

            let t = root_poly(root_rep.roots());
            let input = root_rep.input();
            let degree = t.degree();

            QAP {
                u,
                v,
                w,
                t,
                input,
                degree,
            }
        }
    }

    impl Add for G1Local {
        type Output = G1Local;

        fn add(self, rhs: G1Local) -> Self::Output {
            G1Local(self.0 + rhs.0)
        }
    }

    impl Sub for G1Local {
        type Output = G1Local;

        fn sub(self, rhs: G1Local) -> Self::Output {
            G1Local(self.0 - rhs.0)
        }
    }

    impl Sum for G1Local {
        fn sum<I>(iter: I) -> Self
        where
            I: Iterator<Item = Self>,
        {
            G1Local(iter.fold(G1::zero(), |acc, x| acc + x.0))
        }
    }

    impl Add for G2Local {
        type Output = G2Local;

        fn add(self, rhs: G2Local) -> Self::Output {
            G2Local(self.0 + rhs.0)
        }
    }

    impl Sub for G2Local {
        type Output = G2Local;

        fn sub(self, rhs: G2Local) -> Self::Output {
            G2Local(self.0 - rhs.0)
        }
    }

    impl Sum for G2Local {
        fn sum<I>(iter: I) -> Self
        where
            I: Iterator<Item = Self>,
        {
            G2Local(iter.fold(G2::zero(), |acc, x| acc + x.0))
        }
    }

    impl Add for GtLocal {
        type Output = GtLocal;

        fn add(self, rhs: GtLocal) -> Self::Output {
            GtLocal(self.0 * rhs.0)
        }
    }

    #[test]
    fn single_mult_honest_bn() {
        let qap: QAP<CoefficientPoly<FrLocal>> = QAP {
            u: vec![constant(0), constant(0), constant(1), constant(0)],
            v: vec![constant(0), constant(0), constant(0), constant(1)],
            w: vec![constant(0), constant(1), constant(0), constant(0)],
            t: vec![FrLocal::from(250), FrLocal::from(1)].into(),
            input: 2,
            degree: 1,
        };
        let weights: Vec<FrLocal> = vec![1.into(), 51.into(), 3.into(), 17.into()];

        for _ in 0..10 {
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);

            assert!(verify(
                &qap,
                (sigmag1, sigmag2),
                &vec![FrLocal::from(51), FrLocal::from(3)],
                proof
            ));
        }
    }

    #[test]
    fn bn_encrypt_quad_test() {
        let root_rep = ASTParser::try_parse(&*::std::fs::read_to_string(
            "test_programs/lispesque_quad.zk",
        ).unwrap())
            .unwrap();
        let qap: QAP<CoefficientPoly<FrLocal>> = root_rep.into();

        for _ in 0..10 {
            let (x, a, b, c) = (
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
            );
            let share = a * x * x + b * x + c;

            // The order of the weights is now determined by
            // the order that the variables appear in the file
            let weights: Vec<FrLocal> = vec![1.into(), x, share, a * x, a, x * (a * x + b), b, c];
            let (sigmag1, sigmag2) = setup(&qap);

            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);

            assert!(verify(&qap, (sigmag1, sigmag2), &vec![x, share], proof));
        }
    }

    #[test]
    fn bn_encrypt_cubic_test() {
        let root_rep = ASTParser::try_parse(&*::std::fs::read_to_string(
            "test_programs/lispesque_cubic.zk",
        ).unwrap())
            .unwrap();
        let qap: QAP<CoefficientPoly<FrLocal>> = root_rep.into();

        let trials = 500;
        let (mut setup_time, mut proof_time, mut verify_time) = (0, 0, 0);

        for _ in 0..trials {
            let (x, a, b, c, d) = (
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
            );
            let share = a * x * x * x + b * x * x + c * x + d;

            // The order of the weights is now determined by
            // the order that the variables appear in the file
            let weights: Vec<FrLocal> = vec![
                1.into(),
                x,
                share,
                a * x,
                a,
                x * (a * x + b),
                b,
                x * (x * (a * x + b) + c),
                c,
                d,
            ];

            let now = Instant::now();
            let (sigmag1, sigmag2) = setup(&qap);
            setup_time += now.elapsed().subsec_millis();
            
            let now = Instant::now();
            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);
            proof_time += now.elapsed().subsec_millis();

            let now = Instant::now();
            assert!(verify(&qap, (sigmag1, sigmag2), &vec![x, share], proof));
            verify_time += now.elapsed().subsec_millis();
        }

        println!("Average setup time: {}", setup_time / trials);
        println!("Average proof time: {}", proof_time / trials);
        println!("Average verify time: {}", verify_time / trials);
    }
}
