use super::super::super::field::Field;
use std::collections::HashMap;

pub enum WireType {
    Input,
    Output,
    Internal,
}

#[derive(Clone, Copy)]
pub enum ConnectionType<T>
where
    T: Copy,
{
    Left(T, SubCircuitId),
    Right(T, SubCircuitId),
    Output(SubCircuitId),
}

#[derive(Clone)]
pub struct SubCircuitInputs<T> {
    left_inputs: Vec<(T, WireId)>,
    right_inputs: Vec<(T, WireId)>,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct WireId(usize);

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct SubCircuitId(usize);

pub struct Circuit<T>
where
    T: Copy,
{
    next_wire_id: WireId,
    next_sub_circuit_id: SubCircuitId,

    wire_assignments: HashMap<WireId, Vec<ConnectionType<T>>>,
    sub_circuit_wires: HashMap<SubCircuitId, SubCircuitInputs<T>>,
    wire_values: HashMap<WireId, Option<T>>,
}

impl<T> Circuit<T>
where
    T: Copy + Field,
{
    pub fn unity_wire(&self) -> WireId {
        WireId(0)
    }

    pub fn new_wire(&mut self) -> WireId {
        let next_wire_id = self.next_wire_id;
        self.next_wire_id.0 += 1;
        next_wire_id
    }

    pub fn new_sub_circuit(
        &mut self,
        left_inputs: Vec<(T, WireId)>,
        right_inputs: Vec<(T, WireId)>,
    ) -> WireId {
        let sub_circuit_id = self.next_sub_circuit_id;
        self.next_sub_circuit_id.0 += 1;
        let output_wire = self.new_wire();

        // Update the LHS wire mappings
        for (weight, wire) in left_inputs.clone().into_iter() {
            let connection = ConnectionType::Left(weight, sub_circuit_id);
            if self.wire_assignments.get(&wire).is_none() {
                self.wire_assignments.insert(wire, vec![connection]);
            } else {
                self.wire_assignments
                    .get_mut(&wire)
                    .map(|v| v.push(connection));
            }
        }

        // Update the RHS wire mappings
        for (weight, wire) in right_inputs.clone().into_iter() {
            let connection = ConnectionType::Right(weight, sub_circuit_id);
            if self.wire_assignments.get(&wire).is_none() {
                self.wire_assignments.insert(wire, vec![connection]);
            } else {
                self.wire_assignments
                    .get_mut(&wire)
                    .map(|v| v.push(connection));
            }
        }

        // Update the output wire mappings
        let connection = ConnectionType::Output(sub_circuit_id);
        if self.wire_assignments.get(&output_wire).is_none() {
            self.wire_assignments.insert(output_wire, vec![connection]);
        } else {
            self.wire_assignments
                .get_mut(&output_wire)
                .map(|v| v.push(connection));
        }

        // Update the sub circuit mapping
        self.sub_circuit_wires.insert(
            sub_circuit_id,
            SubCircuitInputs {
                left_inputs,
                right_inputs,
            },
        );

        output_wire
    }

    pub fn evaluate(&mut self, wire: WireId) -> T {
        if wire == self.unity_wire() {
            return T::mul_identity();
        }

        self.wire_values
            .get(&wire)
            .expect("cannot evaluate unknown wire")
            .unwrap_or_else(|| {
                let output_sub_circuit = self
                    .wire_assignments
                    .get(&wire)
                    .expect("a wire must be attached to something")
                    .into_iter()
                    .filter_map(|c| {
                        if let &ConnectionType::Output(sc) = c {
                            Some(sc)
                        } else {
                            None
                        }
                    }).nth(0)
                    .expect("a wire with an unknown value must be the output of a sub circuit");

                let SubCircuitInputs {
                    left_inputs: lhs_wires,
                    right_inputs: rhs_wires,
                } = self
                    .sub_circuit_wires
                    .get(&output_sub_circuit)
                    .expect("a sub circuit referenced by a wire should exist")
                    .clone();

                let lhs = lhs_wires
                    .into_iter()
                    .fold(T::add_identity(), |acc, (weight, wire)| {
                        acc + weight * self.evaluate(wire)
                    });
                let rhs = rhs_wires
                    .into_iter()
                    .fold(T::add_identity(), |acc, (weight, wire)| {
                        acc + weight * self.evaluate(wire)
                    });
                let value = lhs * rhs;
                self.wire_values.insert(wire, Some(value));

                value
            })
    }

    pub fn new_bit_checker(&mut self, input: WireId) -> WireId {
        let lhs = vec![(T::mul_identity(), input)];
        let rhs = vec![
            (T::mul_identity(), input),
            (T::mul_identity().mul_inv(), self.unity_wire()),
        ];

        self.new_sub_circuit(lhs, rhs)
    }
}
