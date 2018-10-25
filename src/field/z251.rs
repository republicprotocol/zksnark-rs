use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Z251 {
    pub inner: u8,
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

impl FieldIdentity for Z251 {
    fn zero() -> Self {
        Z251 { inner: 0 }
    }
    fn one() -> Self {
        Z251 { inner: 1 }
    }
}

impl Field for Z251 {
    fn mul_inv(self) -> Self {
        Z251::one().div(self)
    }
}

impl From<usize> for Z251 {
    fn from(n: usize) -> Self {
        assert!(n < 251);
        Z251 { inner: (n) as u8 }
    }
}

impl Into<usize> for Z251 {
    fn into(self) -> usize {
        self.inner as usize
    }
}

impl FromStr for Z251 {
    type Err = ::std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Z251::from(usize::from_str(s)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z251_add() {
        for i in 0_u16..251_u16 {
            for j in 0_u16..251_u16 {
                let lhs = Z251 { inner: i as u8 };
                let rhs = Z251 { inner: j as u8 };

                assert_eq!((lhs + rhs).inner, ((i + j) % 251) as u8);
            }
        }
    }

    #[test]
    fn z251_neg() {
        for i in 1..251 {
            let lhs = Z251 { inner: i };
            let rhs = -Z251 { inner: i };
            assert_eq!(lhs + rhs, Z251::zero());
        }
    }

    #[test]
    fn z251_mul_inv() {
        for i in 1..251 {
            let lhs = Z251 { inner: i };
            let rhs = Z251 { inner: i }.mul_inv();
            assert_eq!(lhs * rhs, Z251::one());
        }
    }

    #[test]
    fn crt() {
        let rems = [0, 3, 4];
        let moduli = [3, 4, 5];
        let mut ret = chinese_remainder(&rems[..], &moduli[..]);
        while ret < 0 {
            ret += moduli.iter().product::<isize>();
        }
        assert_eq!(ret, 39);

        let rems = [1, 2, 3, 4];
        let moduli = [2, 3, 5, 7];
        let mut ret = chinese_remainder(&rems[..], &moduli[..]);
        while ret < 0 {
            ret += moduli.iter().product::<isize>();
        }
        assert_eq!(ret, 53);
    }
}
