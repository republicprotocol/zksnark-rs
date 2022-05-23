extern crate rabe_bn as bn;
extern crate rand;

use self::bn::{Fr, Group, Gt, G1, G2};
pub use super::*;
use std::str::FromStr;

extern crate borsh;
use self::borsh::{BorshSerialize, BorshDeserialize};


#[derive(Clone, Copy, Eq, PartialEq)]
#[derive(BorshDeserialize, BorshSerialize, Debug)]
pub struct FrLocal(Fr);

#[derive(Clone, Copy, PartialEq)]
#[derive(BorshDeserialize, BorshSerialize, Debug)]
pub struct G1Local(G1);
#[derive(Clone, Copy, PartialEq)]
#[derive(BorshDeserialize, BorshSerialize, Debug)]
pub struct G2Local(G2);
#[derive(PartialEq)]
pub struct GtLocal(Gt);

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

impl FieldIdentity for FrLocal {
    fn zero() -> Self {
        FrLocal(Fr::zero())
    }
    fn one() -> Self {
        FrLocal(Fr::one())
    }
}

impl Field for FrLocal {
    fn mul_inv(self) -> Self {
        FrLocal(self.0.inverse().expect("Tried to get mul inv of zero"))
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
        GtLocal(bn::pairing(g1.0, g2.0))
    }
}

impl Identity for FrLocal {
    fn is_identity(&self) -> bool {
        *self == Self::zero()
    }
}

impl Sum for FrLocal {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(FrLocal::zero(), |acc, x| acc + x)
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

#[cfg(test)]
mod tests {
    use super::super::circuit::{ASTParser, TryParse};
    use super::super::tests::constant;
    use super::*;
    use std::time::Instant;

    #[test]
    fn exp_encrypted_test() {
        for _ in 0..1000 {
            let (a, b) = (FrLocal::random_elem(), FrLocal::random_elem());
            assert!(a.exp_encrypted_g1(b.encrypt_g1()) == (a * b).encrypt_g1());
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

            assert!(verify::<CoefficientPoly<FrLocal>, _, _, _, _>(
                (sigmag1, sigmag2),
                &vec![FrLocal::from(51), FrLocal::from(3)],
                proof
            ));
        }
    }

    #[test]
    fn bn_encrypt_quad_test() {
        let root_rep = ASTParser::try_parse(
            &*::std::fs::read_to_string("test_programs/lispesque_quad.zk").unwrap(),
        ).unwrap();
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

            assert!(verify::<CoefficientPoly<FrLocal>, _, _, _, _>(
                (sigmag1, sigmag2),
                &vec![x, share],
                proof
            ));
        }
    }

    #[test]
    fn bn_encrypt_cubic_test() {
        let root_rep = ASTParser::try_parse(
            &*::std::fs::read_to_string("test_programs/lispesque_cubic.zk").unwrap(),
        ).unwrap();
        let qap: QAP<CoefficientPoly<FrLocal>> = root_rep.into();

        let trials = 10;
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
            assert!(verify::<CoefficientPoly<FrLocal>, _, _, _, _>(
                (sigmag1, sigmag2),
                &vec![x, share],
                proof
            ));
            verify_time += now.elapsed().subsec_millis();
        }

        println!("Average setup time: {}", setup_time / trials);
        println!("Average proof time: {}", proof_time / trials);
        println!("Average verify time: {}", verify_time / trials);
    }

    #[test]
    fn bn_encrypt_deg_15_test() {
        let code = &*::std::fs::read_to_string("test_programs/deg_15.zk").unwrap();
        let root_rep = ASTParser::try_parse(code).unwrap();
        let qap: QAP<CoefficientPoly<FrLocal>> = root_rep.into();

        let trials = 10;
        let (mut setup_time, mut proof_time, mut verify_time) = (0, 0, 0);

        for _ in 0..trials {
            let (x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) = (
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
                FrLocal::random_elem(),
            );

            // The order of the weights is now determined by
            // the order that the variables appear in the file
            let inputs = &[x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p];
            let weights = weights(code, inputs).unwrap();

            let now = Instant::now();
            let (sigmag1, sigmag2) = setup(&qap);
            setup_time += now.elapsed().subsec_millis();

            let now = Instant::now();
            let proof = prove(&qap, (&sigmag1, &sigmag2), &weights);
            proof_time += now.elapsed().subsec_millis();

            let now = Instant::now();
            assert!(verify::<CoefficientPoly<FrLocal>, _, _, _, _>(
                (sigmag1, sigmag2),
                &weights[1..3],
                proof
            ));
            verify_time += now.elapsed().subsec_millis();
        }

        println!("Average setup time: {}", setup_time / trials);
        println!("Average proof time: {}", proof_time / trials);
        println!("Average verify time: {}", verify_time / trials);
    }
}
