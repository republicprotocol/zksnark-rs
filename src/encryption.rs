extern crate rand;

use super::field::z251::Z251;
use super::field::Field;

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
        let mut ret = Z251::mul_identity();
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
        *self == Self::add_identity()
    }
    fn valid(&self) -> bool {
        true
    }
}
