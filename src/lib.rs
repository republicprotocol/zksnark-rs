//! # Zero Knowledge SNARKs
//!
//! This crate provides functionality for creating and using zero knowledge
//! proofs. The implementation is based on
//! [groth16](https://eprint.iacr.org/2016/260.pdf).
//!
//! # Usage
//!
//! The main functions of the alrotihm are the `setup`, `prove` and `verify`
//! functions in the `groth16` module. Intermediate representations can be
//! generated from .zk files, which are written in a DSL that represents an
//! arithmetic circuit.
//!
//! ## Language
//!
//! The language for representing arithmetic circuits is quite basic and is
//! written in a lisp-esque style that uses parenthesised prefix notation. The
//! following is an example program for a circuit that computes a quadratic
//! polynomial `y = ax^2 + bx + c`:
//! ```text
//! (in x a b c)
//! (out y)
//! (verify x y)
//!
//! (program
//!     (= t1
//!         (* x a))
//!     (= t2
//!         (* x (+ t1 b)))
//!     (= y
//!         (* 1 (+ t2 c))))
//! ```
//! The order must always follow `in`, `out`, `verify` and then `program`.
//! Currently parentheses are 'sticky' in that there must not be any whitespace
//! between them and their interior tokens. The keywords are as follows:
//! * `in` precedes the list of input wires to the circuit, excluding the
//!   constant unity wire.
//! * `out` precedes the list of output wires from the circuit.
//! * `verify` precedes the list of wires that the verifier will check by
//!   providing them as input in the verification process.
//! * `program` precedes the list of multiplication subcircuits that constitute
//!   the entire arithmetic circuit. The multiplication subcircuits model a
//!   single multiplication gate that has fan in two, where the two inputs can
//!   be a linear combination of any number of circuit inputs and previous
//!   internal wires. They use the following keywords.
//! * `=` is the assignment operator, which takes two arguments. The first is
//!   the variable that is being assigned to, and represents the output wire of
//!   the multiplication gate. The second is the expression being assigned, and
//!   represents the linear combination of input wires.
//! * `*` is the multiplication operator, which is used both for the
//!   multiplication gate and also to represent the constant scaling in the
//!   linear combination inputs to the multiplication gate. It takes only two
//!   arguments; when used for a multiplication gate the order does not matter,
//!   but for constant scaling the constant must be the first argument.
//! * `+` is the addition operator, and as stated before can have an arbitrary
//!   number of arguments. Each argument can either be a variable, or a scaled
//!   variable (i.e. it can either look like `x`, or, for example, like `(* 5
//!   x)`).
//!
//! # Examples
//!
//! As an example, consider the simple arithmetic expression `x = 4ab + c + 6`.
//! We want to verify the wires `x` and `b`. The program file can look like the
//! following:
//! ```text
//! (in a b c)
//! (out x)
//! (verify b x)
//!
//! (program
//!     (= temp
//!         (* a b))
//!     (= x
//!         (* 1 (+ (* 4 temp) c 6))))
//! ```
//! Suppose that the prove wants to prove that they know values `a` and `c` for
//! which the circuit is satisfied when the verifier inputs `b = 2` and `x =
//! 34`. The following code is an example of the setup, prove and verify
//! process.
//! ```
//! extern crate zksnark;
//!
//! use zksnark::groth16;
//! use zksnark::groth16::{Proof, SigmaG1, SigmaG2, QAP};
//! use zksnark::groth16::circuit::{ASTParser, TryParse};
//! use zksnark::groth16::fr::FrLocal;
//! use zksnark::groth16::coefficient_poly::CoefficientPoly;
//!
//! // x = 4ab + c + 6
//! let qap: QAP<CoefficientPoly<FrLocal>> =
//!     ASTParser::try_parse(&*::std::fs::read_to_string("test_programs/simple.zk").unwrap())
//!         .unwrap()
//!         .into();
//! let weights: Vec<FrLocal> = vec![
//!     1.into(),  // Unity input
//!     2.into(),  // b = 2
//!     34.into(), // x = 34
//!     6.into(),  // temp = ab = 6
//!     3.into(),  // a = 3
//!     4.into(),  // c = 4
//! ];
//!
//! let (sigmag1, sigmag2) = groth16::setup(&qap);
//!
//! let proof = groth16::prove(&qap, (&sigmag1, &sigmag2), &weights);
//!
//! assert!(groth16::verify(
//!     &qap,
//!     (sigmag1, sigmag2),
//!     &vec![FrLocal::from(2), FrLocal::from(34)],
//!     proof
//! ));
//! ```
#![doc(
    html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
    html_favicon_url = "https://www.rust-lang.org/favicon.ico",
    html_root_url = "https://docs.rs/rand/0.5.4"
)]

mod encryption;
pub mod field;
pub mod groth16;

#[doc(hidden)]
pub use groth16::circuit::{ASTParser, TryParse};
#[doc(hidden)]
pub use groth16::coefficient_poly::CoefficientPoly;
#[doc(hidden)]
pub use groth16::fr::FrLocal;
#[doc(hidden)]
pub use groth16::{Proof, SigmaG1, SigmaG2, QAP};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn simple_circuit_test() {
        // x = 4ab + c + 6
        let code = &*::std::fs::read_to_string("test_programs/simple.zk").unwrap();
        let qap: QAP<CoefficientPoly<FrLocal>> =
            ASTParser::try_parse(code)
                .unwrap()
                .into();
        
        let mut assignments = HashMap::new();
        assignments.insert("a".to_string(), 3.into());
        assignments.insert("b".to_string(), 2.into());
        assignments.insert("c".to_string(), 4.into());
        let weights = groth16::weights(code, assignments).unwrap();

        let (sigmag1, sigmag2) = groth16::setup(&qap);

        let proof = groth16::prove(&qap, (&sigmag1, &sigmag2), &weights);

        assert!(groth16::verify(
            &qap,
            (sigmag1, sigmag2),
            &vec![FrLocal::from(2), FrLocal::from(34)],
            proof
        ));

        let (sigmag1, sigmag2) = groth16::setup(&qap);

        let proof = groth16::prove(&qap, (&sigmag1, &sigmag2), &weights);
        
        assert!(!groth16::verify(
            &qap,
            (sigmag1, sigmag2),
            &vec![FrLocal::from(2), FrLocal::from(25)],
            proof
        ));
    }
}
