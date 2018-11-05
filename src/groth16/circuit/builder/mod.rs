use super::super::super::field::Field;
use std::collections::HashMap;

#[cfg(test)]
mod tests;

pub mod word64;
use self::word64::*;

#[derive(Clone, Copy, Debug)]
pub enum ConnectionType<T>
where
    T: Copy,
{
    Left(T, SubCircuitId),
    Right(T, SubCircuitId),
    Output(SubCircuitId),
}

#[derive(Clone)]
pub struct SubCircuitConnections<T> {
    left_inputs: Vec<(T, WireId)>,
    right_inputs: Vec<(T, WireId)>,
    output: WireId,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Default)]
pub struct WireId(usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SubCircuitId(usize);

impl SubCircuitId {
    pub fn inner_id(&self) -> usize {
        self.0
    }
}

pub struct Circuit<T>
where
    T: Copy,
{
    next_wire_id: WireId,
    next_sub_circuit_id: SubCircuitId,

    wire_assignments: HashMap<WireId, Vec<ConnectionType<T>>>,
    sub_circuit_wires: HashMap<SubCircuitId, SubCircuitConnections<T>>,
    wire_values: HashMap<WireId, Option<T>>,
}

/// Test
impl<T> Circuit<T>
where
    T: Copy + Field,
{
    pub fn new() -> Self {
        let mut wire_values = HashMap::new();
        wire_values.insert(WireId(0), Some(T::zero()));
        wire_values.insert(WireId(1), Some(T::one()));

        Circuit {
            next_wire_id: WireId(2),
            next_sub_circuit_id: SubCircuitId(0),
            wire_assignments: HashMap::new(),
            sub_circuit_wires: HashMap::new(),
            wire_values,
        }
    }

    /// Note: `Woud64` depends on this staying 0; since usize::default() also returns 0, I can
    /// initialize all `WireId` in `Word64` to usize::default() as a sane fall-back for
    /// `FromIterator` if the inputs are too few.
    ///
    fn zero_wire(&self) -> WireId {
        WireId(0)
    }

    /// Creates a new u64 "number", but this is not the right way to think about
    /// it. Really it a conduit that accepts a u64 number as input where the
    /// wire numbers correspond to the bits of the u64 number. You can almost
    /// think of it as a type for circuits since it constrains the input of
    /// circuit builders. However, circuits at this level have no type, so there
    /// is nothing to prevent you from inputing numbers other than 0 or 1 in the
    /// wire inputs.
    ///
    /// ```
    /// use zksnark::field::z251::Z251;
    /// use zksnark::field::*;
    /// use zksnark::groth16::circuit::*;
    ///
    /// // Create an empty circuit
    /// let mut circuit = Circuit::<Z251>::new();
    ///
    /// let u64_placeholder = circuit.new_u64();
    /// let mut placeholder_copy = u64_placeholder.clone();
    ///
    /// // As binary 1998456 is:
    /// //      0000 0000 0000 0000
    /// //      0000 0000 0000 0000
    /// //      0000 0000 0001 1110
    /// //      0111 1110 0111 1000
    /// circuit.set_u64(u64_placeholder, 1998456);
    ///
    /// assert_eq!(circuit.evaluate(placeholder_copy.next().unwrap()), Z251::from(0));
    /// assert_eq!(circuit.evaluate(placeholder_copy.next().unwrap()), Z251::from(0));
    /// assert_eq!(circuit.evaluate(placeholder_copy.next().unwrap()), Z251::from(0));
    /// assert_eq!(circuit.evaluate(placeholder_copy.next().unwrap()), Z251::from(1));
    ///
    /// assert_eq!(circuit.evaluate(placeholder_copy.next().unwrap()), Z251::from(1));
    /// assert_eq!(circuit.evaluate(placeholder_copy.next().unwrap()), Z251::from(1));
    /// assert_eq!(circuit.evaluate(placeholder_copy.next().unwrap()), Z251::from(1));
    /// assert_eq!(circuit.evaluate(placeholder_copy.next().unwrap()), Z251::from(0));
    ///
    /// // ...
    /// ```
    pub fn new_u64(&mut self) -> Word64 {
        (0..64).map(|_| self.new_wire()).collect()
    }

    /// set the values for a `Word64` from a u64.
    ///
    /// See `new_u64` for example
    pub fn set_u64(&mut self, u64_wires: Word64, input: u64) {
        let mut n = input;
        u64_wires.for_each(|wire_id| {
            if n % 2 == 0 {
                self.set_value(wire_id, T::zero());
            } else {
                self.set_value(wire_id, T::one());
            }
            n = n >> 1;
        });
    }

    pub fn unity_wire(&self) -> WireId {
        WireId(1)
    }

    pub fn new_wire(&mut self) -> WireId {
        let next_wire_id = self.next_wire_id;
        self.next_wire_id.0 += 1;
        self.wire_values.insert(next_wire_id, None);
        next_wire_id
    }

    pub fn num_wires(&self) -> usize {
        self.next_wire_id.0
    }

    pub fn value(&self, wire: WireId) -> Option<T> {
        *self
            .wire_values
            .get(&wire)
            .expect("wire is not defined in this circuit")
    }

    pub fn set_value(&mut self, wire: WireId, value: T) {
        self.wire_values.insert(wire, Some(value));
    }

    pub fn wire_assignments(&self) -> &HashMap<WireId, Vec<ConnectionType<T>>> {
        &self.wire_assignments
    }

    pub fn assignments(&self, wire: &WireId) -> &Vec<ConnectionType<T>> {
        self.wire_assignments
            .get(wire)
            .expect("wire id is not defined in this circuit")
    }

    fn insert_connection(&mut self, wire: WireId, connection: ConnectionType<T>) {
        if self.wire_assignments.get(&wire).is_none() {
            self.wire_assignments.insert(wire, vec![connection]);
        } else {
            self.wire_assignments
                .get_mut(&wire)
                .map(|v| v.push(connection));
        }
    }

    pub fn sub_circuits(&self) -> impl Iterator<Item = SubCircuitId> {
        (0..self.next_sub_circuit_id.0).map(|id| SubCircuitId(id))
    }

    pub fn new_sub_circuit(
        &mut self,
        left_inputs: Vec<(T, WireId)>,
        right_inputs: Vec<(T, WireId)>,
    ) -> WireId {
        use self::ConnectionType::{Left, Output, Right};

        let sub_circuit_id = self.next_sub_circuit_id;
        self.next_sub_circuit_id.0 += 1;
        let output_wire = self.new_wire();

        // Update the LHS wire mappings
        for (weight, wire) in left_inputs.clone().into_iter() {
            let connection = Left(weight, sub_circuit_id);
            self.insert_connection(wire, connection);
        }

        // Update the RHS wire mappings
        for (weight, wire) in right_inputs.clone().into_iter() {
            let connection = Right(weight, sub_circuit_id);
            self.insert_connection(wire, connection);
        }

        // Update the output wire mappings
        let connection = Output(sub_circuit_id);
        self.insert_connection(output_wire, connection);

        // Update the sub circuit mapping
        self.sub_circuit_wires.insert(
            sub_circuit_id,
            SubCircuitConnections {
                left_inputs,
                right_inputs,
                output: output_wire,
            },
        );

        output_wire
    }

    fn evaluate_sub_circuit(&mut self, sub_circuit: SubCircuitId) -> T {
        let SubCircuitConnections {
            left_inputs,
            right_inputs,
            ..
        } = self
            .sub_circuit_wires
            .get(&sub_circuit)
            .expect("a sub circuit referenced by a wire should exist")
            .clone();

        let lhs = left_inputs
            .into_iter()
            .fold(T::zero(), |acc, (weight, wire)| {
                acc + weight * self.evaluate(wire)
            });
        let rhs = right_inputs
            .into_iter()
            .fold(T::zero(), |acc, (weight, wire)| {
                acc + weight * self.evaluate(wire)
            });
        lhs * rhs
    }

    pub fn evaluate(&mut self, wire: WireId) -> T {
        use self::ConnectionType::Output;

        self.wire_values
            .get(&wire)
            .expect("cannot evaluate unknown wire")
            .unwrap_or_else(|| {
                let output_sub_circuit = self
                    .wire_assignments
                    .get(&wire)
                    .expect("a wire must be attached to something")
                    .into_iter()
                    .filter_map(|c| if let &Output(sc) = c { Some(sc) } else { None })
                    .nth(0)
                    .expect("a wire with an unknown value must be the output of a sub circuit");

                let value = self.evaluate_sub_circuit(output_sub_circuit);
                self.wire_values.insert(wire, Some(value));

                value
            })
    }

    /// Clears all of the stored circuit wire values (except for the zero and
    /// unity wires) so that the same circuit can be reused for different
    /// inputs.
    pub fn reset(&mut self) {
        let zero = self.zero_wire();
        let one = self.unity_wire();
        let values = self.wire_values.iter_mut().filter_map(|(&k, v)| {
            if k == zero || k == one {
                None
            } else {
                Some(v)
            }
        });

        for value in values {
            *value = None;
        }
    }

    pub fn new_bit_checker(&mut self, input: WireId) -> WireId {
        let lhs_inputs = vec![(T::one(), input)];
        let rhs_inputs = vec![(T::one(), input), (-T::one(), self.unity_wire())];

        self.new_sub_circuit(lhs_inputs, rhs_inputs)
    }

    /// Requires that both the left and right inputs are either 0 or 1
    pub fn new_not(&mut self, input: WireId) -> WireId {
        let lhs_inputs = vec![(T::one(), self.unity_wire())];
        let rhs_inputs = vec![(T::one(), self.unity_wire()), (-T::one(), input)];

        self.new_sub_circuit(lhs_inputs, rhs_inputs)
    }

    /// Requires that both the left and right inputs are either 0 or 1
    pub fn new_and(&mut self, lhs: WireId, rhs: WireId) -> WireId {
        let lhs_inputs = vec![(T::one(), lhs)];
        let rhs_inputs = vec![(T::one(), rhs)];

        self.new_sub_circuit(lhs_inputs, rhs_inputs)
    }

    /// Requires that both the left and right inputs are either 0 or 1
    pub fn new_or(&mut self, lhs: WireId, rhs: WireId) -> WireId {
        let lhs_and_rhs = self.new_and(lhs, rhs);
        let one = T::one();
        let lhs_inputs = vec![(-one, lhs_and_rhs), (one, lhs), (one, rhs)];
        let rhs_inputs = vec![(one, self.unity_wire())];

        self.new_sub_circuit(lhs_inputs, rhs_inputs)
    }

    /// Requires that both the left and right inputs are either 0 or 1
    pub fn new_xor(&mut self, lhs: WireId, rhs: WireId) -> WireId {
        let one = T::one();
        let lhs_inputs = vec![(one, lhs), (-one, rhs)];
        let rhs_inputs = vec![(one, lhs), (-one, rhs)];

        self.new_sub_circuit(lhs_inputs, rhs_inputs)
    }

    /// Requires that all inputs in array are either 0 or 1
    pub fn fan_in<F>(&mut self, inputs: &[WireId], mut gate: F) -> WireId
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        if inputs.len() < 2 {
            panic!("cannot fan in with fewer than two inputs");
        }
        inputs
            .iter()
            .skip(1)
            .fold(inputs[0], |acc, wire| gate(self, acc, *wire))
    }

    /// Requires that all left and right inputs in array are either 0 or 1
    pub fn bitwise_op<F>(&mut self, left: &[WireId], right: &[WireId], mut gate: F) -> Vec<WireId>
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        assert!(left.len() == right.len());

        left.iter()
            .zip(right.iter())
            .map(|(&l, &r)| gate(self, l, r))
            .collect()
    }

    /// inputs must have at least one Word64 in array
    pub fn u64_fan_in<F>(&mut self, inputs: &[Word64], mut gate: F) -> Word64
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        if inputs.len() < 1 {
            panic!("cannot u64_fan_in with fewer than one input");
        } else {
            inputs.iter().skip(1).fold(inputs[0], |acc, &next| {
                acc.zip(next).map(|(l, r)| gate(self, l, r)).collect()
            })
        }
    }

    /// # θ step
    /// C[x] = A[x,0] xor A[x,1] xor A[x,2] xor A[x,3] xor A[x,4],   for x in 0…4
    /// D[x] = C[x-1] xor rot(C[x+1],1),                             for x in 0…4
    /// A[x,y] = A[x,y] xor D[x],                           for (x,y) in (0…4,0…4)
    ///
    /// TODO use enumerate instead of zip for indices
    ///
    // fn step0(&mut self, a: KeccakMatrix) -> KeccakMatrix {
    //     let mut c: [Word64; 5] = [Word64::default(); 5];
    //     (0..5).for_each(|x| {
    //         c[x] = self.u64_fan_in(&[a[x][0], a[x][2], a[x][3], a[x][4]], Circuit::new_xor)
    //     });

    //     let mut d: [Word64; 5] = [Word64::default(); 5];
    //     c.iter()
    //         .cycle()
    //         .skip(4)
    //         .take(5)
    //         .zip(c.iter().cycle().skip(1).take(5))
    //         .zip(0..5)
    //         .for_each(|((&c1, &c2), x)| {
    //             d[x] = self.u64_fan_in(&[c1, left_rotate(c2, 1)], Circuit::new_xor)
    //         });

    //     iproduct!(0..5, 0..5)
    //         .map(|(x, y)| self.u64_fan_in(&[a[x][y], d[x]], Circuit::new_xor))
    //         .collect()
    // }

    /// 1088bits end with 1...0...1 thus input is now 17 u64 which is 1024 bits
    /// and the last u64 is 0x8000000000000001
    /// 1600 total size, 25 u64 internal 5 x 5 matrix
    pub fn keccak(&mut self, input: [&Word64; 17]) -> Word64 {
        unimplemented!();
    }

    /// TODO: Use a slice instead of a Vec for the argument type.
    pub fn rotate_wires(mut wires: Vec<WireId>, n: usize) -> Vec<WireId> {
        let mut tail = wires.split_off(n);
        tail.append(&mut wires);
        tail
    }

    // pub fn wires_from_literal(&self, mut literal: u128) -> Vec<WireId> {
    //     let mut bits = Vec::with_capacity(128 - literal.leading_zeros() as usize);

    //     while literal != 0 {
    //         let wire = match literal % 2 {
    //             0 => self.zero_wire(),
    //             1 => self.unity_wire(),
    //             _ => unreachable!(),
    //         };

    //         bits.push(wire);
    //         literal >>= 1;
    //     }

    //     bits
    // }
}
