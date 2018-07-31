use std::ops::{Add, Mul};

trait CircuitGate<F> {
    fn evaluate(&self) -> Option<F>;
    fn output(&mut self) -> &mut Wire<F>;
    fn propagate(&mut self) -> Result<(), ()> {
        self.output().value = self.evaluate();
        self.output().value.as_ref().map(|_| ()).ok_or(())
    }
}

trait Zero<F> {
    fn zero() -> F;
}

#[derive(Debug, PartialEq)]
struct Wire<F> {
    identifier: String,
    value: Option<F>,
}

struct AddGate<'a, F: 'a> {
    input: &'a mut [&'a mut Wire<F>],
    output: &'a mut Wire<F>,
}

impl<'a, F> CircuitGate<F> for AddGate<'a, F>
where
    F: 'a + Zero<F> + Add<Output = F> + Clone,
{
    fn evaluate(&self) -> Option<F> {
        self.input.iter().try_fold(F::zero(), |acc, x| {
            x.value.as_ref().map(|v| acc + v.clone())
        })
    }

    fn output(&mut self) -> &mut Wire<F> {
        self.output
    }
}

struct MulGate<'a, F: 'a> {
    input: (&'a mut Wire<F>, &'a mut Wire<F>),
    output: &'a mut Wire<F>,
}

impl<'a, F> CircuitGate<F> for MulGate<'a, F>
where
    F: 'a + Mul<Output = F> + Clone,
{
    fn evaluate(&self) -> Option<F> {
        self.input
            .0
            .value
            .as_ref()
            .and_then(|l| self.input.1.value.as_ref().map(|r| l.clone() * r.clone()))
    }

    fn output(&mut self) -> &mut Wire<F> {
        self.output
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
        let a = &mut Wire::<usize> {
            identifier: "a".to_string(),
            value: Some(2),
        };
        let b = &mut Wire::<usize> {
            identifier: "b".to_string(),
            value: None,
        };
        let mut c = Wire::<usize> {
            identifier: "c".to_string(),
            value: None,
        };

        let mut gate = AddGate::<usize> {
            input: &mut [a, b],
            output: &mut c
        };

        assert_eq!(gate.evaluate(), None);
        assert_eq!(gate.propagate(), Err(()));

        gate.input[1].value = Some(3);

        assert_eq!(gate.evaluate(), Some(5));

        gate.propagate().expect("All input wires are assigned");

        assert_eq!(*gate.output(), Wire::<usize> {
            identifier: "c".to_string(),
            value: Some(5),
        });
    }
    
    #[test]
    fn mul_gate_test() {
        let a = &mut Wire::<usize> {
            identifier: "a".to_string(),
            value: Some(2),
        };
        let b = &mut Wire::<usize> {
            identifier: "b".to_string(),
            value: None,
        };
        let mut c = Wire::<usize> {
            identifier: "c".to_string(),
            value: None,
        };

        let mut gate = MulGate::<usize> {
            input: (a, b),
            output: &mut c
        };

        assert_eq!(gate.evaluate(), None);
        assert_eq!(gate.propagate(), Err(()));

        gate.input.1.value = Some(3);

        assert_eq!(gate.evaluate(), Some(6));

        gate.propagate().expect("All input wires are assigned");

        assert_eq!(*gate.output(), Wire::<usize> {
            identifier: "c".to_string(),
            value: Some(6),
        });
    }
}
