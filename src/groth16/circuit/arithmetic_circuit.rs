use std::ops::{Add, Mul};
use std::fmt::Debug;

trait CircuitGate<F> : Debug {
    fn evaluate(&self) -> Option<F>;
}

trait Zero<F> {
    fn zero() -> F;
}

#[derive(Debug, PartialEq)]
struct Wire<F> {
    identifier: String,
    value: Option<F>,
}

impl<F> CircuitGate<F> for Wire<F>
where
    F: Clone + Debug,
{
    fn evaluate(&self) -> Option<F> {
        self.value.as_ref().map(|v| v.clone())
    }
}

#[derive(Debug)]
struct AddGate<'a, F: 'a> {
    inputs: &'a [&'a CircuitGate<F>],
    output: &'a Wire<F>,
}

impl<'a, F> CircuitGate<F> for AddGate<'a, F>
where
    F: 'a + Zero<F> + Add<Output = F> + Debug,
{
    fn evaluate(&self) -> Option<F> {
        self.inputs
            .iter()
            .try_fold(F::zero(), |acc, x| x.evaluate().map(|v| acc + v))
    }
}

#[derive(Debug)]
struct MulGate<'a, F: 'a> {
    left: &'a CircuitGate<F>,
    right: &'a CircuitGate<F>,
    output: &'a Wire<F>,
}

impl<'a, F> CircuitGate<F> for MulGate<'a, F>
where
    F: 'a + Mul<Output = F> + Debug,
{
    fn evaluate(&self) -> Option<F> {
        self.left
            .evaluate()
            .and_then(|l| self.right.evaluate().map(|r| l * r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Zero<usize> for usize {
        fn zero() -> usize {
            0
        }
    }

    #[test]
    fn add_gate_test() {
        let a_none = &Wire::<usize> {
            identifier: "a".to_string(),
            value: None,
        };
        let a_some = &Wire::<usize> {
            identifier: "a".to_string(),
            value: Some(2),
        };
        let b = &Wire::<usize> {
            identifier: "b".to_string(),
            value: Some(3),
        };
        let c = &Wire::<usize> {
            identifier: "c".to_string(),
            value: None,
        };
        let gate = AddGate {
            inputs: &[a_none, b],
            output: c,
        };

        assert_eq!(gate.evaluate(), None);
        
        let gate = AddGate {
            inputs: &[a_some, b],
            output: c,
        };

        assert_eq!(gate.evaluate(), Some(5));
    }

    #[test]
    fn mul_gate_test() {
        let a_none = &Wire::<usize> {
            identifier: "a".to_string(),
            value: None,
        };
        let a_some = &Wire::<usize> {
            identifier: "a".to_string(),
            value: Some(2),
        };
        let b = &Wire::<usize> {
            identifier: "b".to_string(),
            value: Some(3),
        };
        let c = &Wire::<usize> {
            identifier: "c".to_string(),
            value: None,
        };
        let gate = MulGate {
            left: a_none,
            right: b,
            output: c,
        };

        assert_eq!(gate.evaluate(), None);
        
        let gate = MulGate {
            left: a_some,
            right: b,
            output: c,
        };

        assert_eq!(gate.evaluate(), Some(6));
    }
}
