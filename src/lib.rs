mod encryption;
pub mod field;
pub mod groth16;

#[cfg(test)]
mod tests {
    use super::field::*;

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
            assert_eq!(lhs + rhs, Z251::add_identity());
        }
    }

    #[test]
    fn z251_mul_inv() {
        for i in 1..251 {
            let lhs = Z251 { inner: i };
            let rhs = Z251 { inner: i }.mul_inv();
            assert_eq!(lhs * rhs, Z251::mul_identity());
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
