extern crate rand;

use super::field::z251::Z251;
use super::field::FieldIdentity;

pub trait Encryptable {
    type Output;

    fn encrypt(self) -> Self::Output;
    fn random() -> Self;
}

pub trait EncryptProperties {
    fn detect_root(&self) -> bool;
    fn valid(&self) -> bool;
}

impl Encryptable for Z251 {
    type Output = Z251;

    fn encrypt(self) -> Self::Output {
        let mut ret = Z251::one();
        for _ in 0..self.inner {
            ret = ret * Z251 { inner: 69 };
        }

        ret
    }
    fn random() -> Self {
        Z251 {
            inner: rand::random::<u8>() % 251,
        }
    }
}

impl EncryptProperties for Z251 {
    fn detect_root(&self) -> bool {
        *self == Self::zero()
    }
    fn valid(&self) -> bool {
        true
    }
}
