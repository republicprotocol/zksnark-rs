use super::super::super::field::Field;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::ops::Deref;
extern crate itertools;

#[cfg(test)]
mod tests;

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

/// The idea here is to have each `WireId` have an input of either 0 or 1 and
/// then group those wires together to form a single `u64` number where `wire1`
/// corresponds to the first bit (also called the right most bit) of the `u64`,
/// then `wire2` the second bit and so on.  
///
pub struct Word64([WireId; 64]);

impl Deref for Word64 {
    type Target = [WireId; 64];

    fn deref(&self) -> &[WireId; 64] {
        &self.0
    }
}

impl FromIterator<WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        let mut arr: [WireId; 64] = [WireId::default(); 64];
        (0..64)
            .zip(iter.into_iter())
            .for_each(|(i, num)| arr[i] = num);
        Word64(arr)
    }
}

/// It is a 5 by 5 matrix used as the internal state of Keccak hash
/// function or other matrix need by the Keccak hash function.
pub struct KeccakMatrix<T>([[T; 5]; 5]);

impl<T> Deref for KeccakMatrix<T> {
    type Target = [[T; 5]; 5];

    fn deref(&self) -> &[[T; 5]; 5] {
        &self.0
    }
}

/// Fills the matrix from left to right then top to bottom.
///
/// (0..24)
///
/// 0  | 1  | 2  | 3  | 4
/// 5  | 6  | 7  | 8  | 9
/// 10 | 11 | 12 | 13 | 14
/// 15 | 16 | 17 | 18 | 19
/// 20 | 21 | 22 | 23 | 24
impl<T> FromIterator<T> for KeccakMatrix<T>
where
    T: Default + Copy,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut arr = [[T::default(); 5]; 5];
        iproduct!(0..5, 0..5)
            .zip(iter.into_iter())
            .for_each(|((x, y), num)| arr[x][y] = num);
        KeccakMatrix(arr)
    }
}

const rotation_offset: KeccakMatrix<u64> = KeccakMatrix([
    [0, 36, 3, 18, 41],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]);

const round_constants: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

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

/// The purpose of this struct is to build circuits into a representation that
/// can be turned into a `QAP` (Quadratic Arithmetic Program) which is defined
/// in "groth16/mod.rs". I'll start this off by describing what are the building
/// blocks for a `Circuit` are made of:
///
/// # Sub Circuit
///
/// You can think of a circuit as a collection of nodes connected by wires
/// without any cycles (Directed Acyclic Graph). Nodes are made up of
/// "sub circuit" which in reality look like this:
///
/// ```
/// // `Left` input wires-> \|/ \|  <- `Right` input wires
/// //                       +   +  <- Plus operation
/// //                        \ /   <- (implicit connections, not wires)
/// //                         *    <- Multiplication operation
/// //                         |    <- `Output` wire
/// ```
///
/// Lets walk through the example "sub circuit" which is the atomic unit of a
/// `Circuit`. In this example the left plus operation has 3 input wires and the
/// right plus operation has 2 input wires. Every plus operation must have at
/// least one input wire, but may have any number of extra input wires. Each
/// "wire" also has an associated weight and takes some input, unless it is an
/// `Output` wire. The way the "sub circuit" is evaluated is by evaluating all
/// input wires, followed by the plus operation and multiplication operation.
///
/// ```
/// // For Example, given the above "sub circuit" we can evaluate it as follows:
/// //              We have these inputs and weights on the left wires:
/// //                 - [(1,1), (0,2), (2,1)] : [(weights, inputs)]
///
/// //              And here are the inputs and weights on the right wires:
/// //                 - [(0,3), (0,1)] : [(weights, inputs)]
///
/// //              To evaluate the wires we multiply their inputs by their weight:
/// //              and then the plus operation adds the results together:
/// //                 - Left plus operation equals => ((1 * 1) + (0 * 2) + (2 * 1))
/// //                 - Right plus operation equals => ((0 * 3) + (1 * 0))
///
/// //              Finally the multiplication operation multiplies the result
/// //              of the previous two:
/// //                 - (((1 * 1) + (0 * 2) + (2 * 1)) * ((0 * 3) + (1 * 0)))
/// //              
/// //              Thus the output wire would have the value: 0
/// ```
///
/// Note: A single wire may connect to any number of sub circuits including to
/// the same sub circuit multiple times on both its left and right inputs.
/// (Exception, wires must never form a loop anywhere in the `Circuit`)
///
/// # `Circuit`
///
/// A `Circuit` is made up of many connecting sub circuits. To evaluate a
/// `Circuit` means to determine the value of the `Circuit`'s output wires.
/// Which are in turn made up of the sub circuits output wires that do not
/// connect to another sub circuit's input wires. This also means a `Circuit`
/// has some number of input wires which come from the sub circuits with input
/// wires that do not connect to other sub circuit's output wires. `Circuit`s
/// are pure in the sense that the input uniquely determines the output of the
/// `Circuit`.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use zksnark::field::z251::Z251;
/// use zksnark::field::*;
/// use zksnark::groth16::circuit::*;
///
/// // Create an empty circuit
/// let mut circuit = Circuit::<Z251>::new();
///
/// // In order to give our sub circuits input we need input wires that the sub
/// // circuits will take as arguments.
/// let input_wire = circuit.new_wire();
///
/// // lets start with a bit checker that returns 0 if the input is 0 or 1 and
/// // some other number in other cases. This is useful after turning the
/// // `Circuit` into a `CircuitInstance` where these wires can be checked to
/// // ensure they evaluate to zero. However, in this module we can only
/// // evaluate the whole `Circuit`
/// let checker_wire = circuit.new_bit_checker(input_wire);
///
/// // First lets give it a zero to check if it does what we think.
/// circuit.set_value(input_wire, Z251::from(0));
/// assert_eq!(circuit.evaluate(checker_wire), Z251::from(0));
///
/// // Next we need to reset all wire values in order to re-evaluate the circuit
/// // and in order to set the inputs to new values.
/// circuit.reset();
///
/// // Same check with 1 as input
/// circuit.set_value(input_wire, Z251::from(1));
/// assert_eq!(circuit.evaluate(checker_wire), Z251::from(0));
///
/// // Just to demonstrate a negative test
/// circuit.reset();
/// circuit.set_value(input_wire, Z251::from(2));
/// assert_ne!(circuit.evaluate(checker_wire), Z251::from(0));
///
/// // This gives you a basic idea of how to use some components, but not what
/// // this actually looks like. Let me draw a picture of what the circuit looks
/// // like now:
///
/// //  (weight 1, input_wire) -> |
/// //                           / \
/// //                           |  \| <- (weight -1, unity_wire)
/// //                           +   +
/// //                            \ /
/// //                             *
/// //                             | <- checker_wire
///
/// // Note: the unity wire, is a wire that always outputs the value 1. You can
/// // think of it as an input wire that is permanently set to an input of one.
///
/// // Now we can add something onto that `checker` wire
/// let not_wire = circuit.new_not(checker_wire);
///
/// // Now it looks like this:
///
/// //    (weight 1, input_wire) -> |
/// //                             / \
/// //                             |  \| <- (weight -1, one wire)
/// //                             +   +
/// //                              \ /
/// //                               *
/// //                               | <- checker_wire
/// // ------------------------------|-----------------------------------------
/// //                               | <- (weight -1, checker_wire)
/// // (weight 1, unity wire) -> |   |/ <- (weight 1, unity_wire)
/// //                           +   +
/// //                            \ /
/// //                             *
/// //                             | <- not_wire
///
/// // Note: the line "----" is to show where the sub circuit starts and ends
/// // conceptually. The checker_wire, when inserted into the input of the
/// // new_not, is the same wire; it just gets assigned a weight.
///
/// // reset, because we have different inputs from before
/// circuit.reset();
/// circuit.set_value(input_wire, Z251::from(0));
/// assert_eq!(circuit.evaluate(not_wire), Z251::from(1));
///
/// // Check that the not's input was either 0 or 1
/// // Note, this call will not need to "evaluate" the checker_wire since it
/// // was already evaluated to get the value of not_wire
/// assert_eq!(circuit.evaluate(checker_wire), Z251::from(0));
/// ```
///
impl<T> Circuit<T>
where
    T: Copy + Field,
{
    pub fn new() -> Self {
        // TODO: Initialise the unity wire value to be one?
        Circuit {
            next_wire_id: WireId(1),
            next_sub_circuit_id: SubCircuitId(0),
            wire_assignments: HashMap::new(),
            sub_circuit_wires: HashMap::new(),
            wire_values: HashMap::new(),
        }
    }

    pub fn unity_wire(&self) -> WireId {
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
    /// let u64_input = circuit.new_u64();
    ///
    /// // First lets give it a zero to check if it does what we think.
    /// // As binary 1998456 is: 0001 1110 0111 1110 0111 1000
    /// circuit.set_u64(&u64_input, 1998456);
    ///
    /// assert_eq!(circuit.evaluate(u64_input[0]), Z251::from(0));
    /// assert_eq!(circuit.evaluate(u64_input[1]), Z251::from(0));
    /// assert_eq!(circuit.evaluate(u64_input[2]), Z251::from(0));
    /// assert_eq!(circuit.evaluate(u64_input[3]), Z251::from(1));
    ///
    /// assert_eq!(circuit.evaluate(u64_input[4]), Z251::from(1));
    /// assert_eq!(circuit.evaluate(u64_input[5]), Z251::from(1));
    /// assert_eq!(circuit.evaluate(u64_input[6]), Z251::from(1));
    /// assert_eq!(circuit.evaluate(u64_input[7]), Z251::from(0));
    ///
    /// // ...
    /// ```
    pub fn new_u64(&mut self) -> Word64 {
        (0..64).map(|_| self.new_wire()).collect()
    }

    /// set the values for a `Word64` from a u64.
    ///
    /// See `new_u64` for example
    pub fn set_u64(&mut self, u64_wires: &Word64, input: u64) {
        let mut n = input;
        u64_wires.iter().for_each(|&wire_id| {
            if n % 2 == 0 {
                self.set_value(wire_id, T::zero());
            } else {
                self.set_value(wire_id, T::one());
            }
            n = n >> 1;
        });
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

        if wire == self.unity_wire() {
            return T::one();
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
                    .filter_map(|c| if let &Output(sc) = c { Some(sc) } else { None })
                    .nth(0)
                    .expect("a wire with an unknown value must be the output of a sub circuit");

                let value = self.evaluate_sub_circuit(output_sub_circuit);
                self.wire_values.insert(wire, Some(value));

                value
            })
    }

    /// Clears all of the stored circuit wire values (including those manually
    /// set) so that the same circuit can be reused for different inputs.
    pub fn reset(&mut self) {
        for value in self.wire_values.values_mut() {
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

    // pub fn shift_left(&mut self, input &[WireId]) -> &[WireId] {

    // }

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
    /// 1088bits end with 1...0...1
    /// thus input is now 17 u64 which is 1024 bits and the last u64 is 0x8000000000000001
    /// 1600 total size, 25 u64 internal 5 x 5 matrix
    pub fn keccak512_72(&mut self, input: &[WireId; 72]) -> &[WireId; 64] {
        unimplemented!();
    }
}
