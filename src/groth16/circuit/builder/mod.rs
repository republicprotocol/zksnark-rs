use super::super::super::field::Field;
use std::collections::HashMap;
use std::fmt;

extern crate itertools;
use itertools::Itertools;

#[cfg(test)]
mod tests;

mod types;
pub use self::types::{Word64, Word8};

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

#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
pub struct WireId(usize);

impl fmt::Debug for WireId {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.0)
    }
}

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

/// This is used internally in circuit bulider.
struct KeccakInternal {
    a: [Word64; 25],
    offset: usize,
    rate: usize,
    delim: u8,
}

impl KeccakInternal {
    fn a_bytes(&self) -> [Word8; 200] {
        let mut arr: [Word8; 200] = [Word8::default(); 200];
        self.a
            .iter()
            .flat_map(|wrd64| wrd64.iter())
            .enumerate()
            .for_each(|(i, &wrd8)| arr[i] = wrd8);
        arr
    }
}

impl<T> Circuit<T>
where
    T: Field,
{
    /// TODO: something might be wrong with setting the zero wire. @Ross
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

    /// The `Default` instances for `WireId`, `Word8`, `Word64`, `KeccakMatrix`,
    /// `KeccakRow` all depend on this being 0. In other words the default is to
    /// create `zero_wire` to fill in any blanks by creating `WireId(0)`.
    fn zero_wire(&self) -> WireId {
        WireId(0)
    }

    pub fn unity_wire(&self) -> WireId {
        WireId(1)
    }

    ////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////// New Wire Functions /////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    pub fn new_wire(&mut self) -> WireId {
        let next_wire_id = self.next_wire_id;
        self.next_wire_id.0 += 1;
        self.wire_values.insert(next_wire_id, None);
        next_wire_id
    }

    /// Creates a new u8 "number", but this is not the right way to think about
    /// it. Really it is a conduit that accepts a u8 number as input where the
    /// wire numbers correspond to the bits of the u8 number. You can almost
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
    /// let u8_input = circuit.new_word8();
    ///
    /// // As binary 0x4F is: 0100 1111
    /// circuit.set_word8(&u8_input, 0b0000_0010);
    /// assert_eq!(circuit.evaluate_word8(&u8_input), 0b0000_0010);
    ///
    /// // ...
    /// ```
    pub fn new_word8(&mut self) -> Word8 {
        let mut wrd8: Word8 = Word8::default();
        (0..8).for_each(|x| wrd8[x] = self.new_wire());
        wrd8
    }

    pub fn new_word8_array<'a>(&mut self, output: &'a mut [Word8]) {
        (0..output.len()).for_each(|i| output[i] = self.new_word8());
    }

    pub fn new_word8_vec(&mut self, size: usize) -> Vec<Word8> {
        (0..size).map(|_| self.new_word8()).collect()
    }

    /// Creates a new u64 "number", but this is not the right way to think about
    /// it. Really it is a conduit that accepts a u64 number as input where the
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
    /// let u64_input = circuit.new_word64();
    ///
    /// circuit.set_word64(&u64_input, 1);
    ///
    /// assert_eq!(circuit.evaluate_word64(&u64_input), 1);
    /// ```
    pub fn new_word64(&mut self) -> Word64 {
        let mut wrd64: Word64 = Word64::default();
        (0..8).for_each(|x| wrd64[x] = self.new_word8());
        wrd64
    }

    /// NOTE: this is only used internally to implement Keccak
    fn initial_keccakmatrix(&mut self) -> [Word64; 25] {
        [Word64::default(); 25]
    }

    // NOTE: this is only used for testing internal components
    // It creates a new keccak matrix with placeholders instead of
    // a default
    fn new_keccakmatrix(&mut self) -> [Word64; 25] {
        let mut matrix: [Word64; 25] = [Word64::default(); 25];
        (0..25).for_each(|x| matrix[x] = self.new_word64());
        matrix
    }

    // NOTE: this is only used for testing internal components
    // It sets the wireId's in the keccak matrix to the provided
    // values
    fn set_keccakmatrix(&mut self, matrix_wires: &[Word64; 25], input: &[u64; 25]) {
        matrix_wires
            .iter()
            .zip(input.iter())
            .for_each(|(wrd64, num)| self.set_word64(&wrd64, *num));
    }

    ////////////////////////////////////////////////////////////////////////////////
    /////////////////////////// Const Word8 and Word64 /////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    pub fn const_word8(&mut self, mut input: u8) -> Word8 {
        let mut wrd8: Word8 = Word8::default();
        (0..8).for_each(|i| {
            if input % 2 == 0 {
                wrd8[i] = self.zero_wire();
            } else {
                wrd8[i] = self.unity_wire();
            }
            input = input >> 1;
        });
        wrd8
    }

    pub fn const_word64(&mut self, input: u64) -> Word64 {
        let mut wrd64: Word64 = Word64::default();
        types::to_ne_u8(input)
            .iter()
            .enumerate()
            .for_each(|(i, &num)| wrd64[i] = self.const_word8(num));
        wrd64
    }

    ////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////// Set Wire Functions ///////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    pub fn set_value(&mut self, wire: WireId, value: T) {
        self.wire_values.insert(wire, Some(value));
    }

    /// set the values for a `Word8` from a u8.
    ///
    /// See `new_u8` for example
    ///
    pub fn set_word8(&mut self, u8_wires: &Word8, mut input: u8) {
        u8_wires.iter().for_each(|&wire_id| {
            if input % 2 == 0 {
                self.set_value(wire_id, T::zero());
            } else {
                self.set_value(wire_id, T::one());
            }
            input = input >> 1;
        });
    }

    /// Set the values for a `Word64` from a u64.
    ///
    /// See `new_u8` for example
    ///
    pub fn set_word64(&mut self, u64_wires: &Word64, input: u64) {
        u64_wires
            .iter()
            .zip(types::to_ne_u8(input).iter())
            .for_each(|(word, &num)| self.set_word8(word, num));
    }

    ////////////////////////////////////////////////////////////////////////////////
    ///////////////////// Set and create new Wire Functions ////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    /// This is a convenience function to both create a new Word8
    /// placeholder and set it with an input.
    pub fn set_new_word8(&mut self, input: u8) -> Word8 {
        let x = self.new_word8();
        self.set_word8(&x, input);
        x
    }

    /// This is a convenience function to both create a new Word64
    /// placeholder and set it with an input.
    pub fn set_new_word64(&mut self, input: u64) -> Word64 {
        let x = self.new_word64();
        self.set_word64(&x, input);
        x
    }

    /// This creates new `Word8`s, sets them with the `input` and
    /// places them in the `output` array.
    ///
    /// `output[0]` is set from `input.first()` and so on.
    ///
    /// If you give an iterator that has more than the size of the
    /// `output` array it will be ignored.
    ///
    /// ```
    /// use zksnark::field::z251::Z251;
    /// use zksnark::field::*;
    /// use zksnark::groth16::circuit::*;
    ///
    /// // Create an empty circuit
    /// let mut circuit = Circuit::<Z251>::new();
    ///
    /// let external_input: [u8; 7] = [9, 24, 45, 250, 99, 0, 7];
    /// let new_set_circuit_input = &mut [Word8::default(); 7];
    /// circuit.set_new_word8_array(external_input.iter()
    ///                             , new_set_circuit_input);
    ///
    /// let evaluated_circuit = &mut [0; 7];
    /// circuit.evaluate_word8_to_array(new_set_circuit_input.iter()
    ///                              , evaluated_circuit);
    ///
    /// assert_eq!(*evaluated_circuit, external_input);
    /// ```
    ///
    pub fn set_new_word8_array<'a>(
        &mut self,
        input: impl Iterator<Item = &'a u8>,
        output: &'a mut [Word8],
    ) {
        input
            .zip(0..output.len())
            .for_each(|(num, i)| output[i] = self.set_new_word8(*num));
    }

    /// This creates new `Word8`s, sets them with the `input` and
    /// gives them back as a `Vec`
    ///
    /// `vec[0]` is set from `input.first()` and so on.
    ///
    /// ```
    /// use zksnark::field::z251::Z251;
    /// use zksnark::field::*;
    /// use zksnark::groth16::circuit::*;
    ///
    /// // Create an empty circuit
    /// let mut circuit = Circuit::<Z251>::new();
    ///
    /// let external_input = vec![9, 24, 45, 250, 99, 0, 7];
    /// let new_set_circuit_input =
    ///     circuit.set_new_word8_vec(external_input.iter());
    ///
    /// let evaluated_circuit =
    ///     circuit.evaluate_word8_to_vec(new_set_circuit_input.iter());
    ///
    /// assert_eq!(evaluated_circuit, external_input);
    /// ```
    ///
    pub fn set_new_word8_vec<'a>(&mut self, input: impl Iterator<Item = &'a u8>) -> Vec<Word8> {
        input.map(|num| self.set_new_word8(*num)).collect()
    }

    ////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// Flatten Wires ////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    pub fn bit_check(&mut self, input: impl IntoIterator<Item = WireId>) -> Vec<WireId> {
        input.into_iter().map(|x| self.new_bit_checker(x)).collect()
    }

    ////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////// Wire Functions ////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    pub fn num_wires(&self) -> usize {
        self.next_wire_id.0
    }

    pub fn value(&self, wire: WireId) -> Option<T> {
        *self
            .wire_values
            .get(&wire)
            .expect("wire is not defined in this circuit")
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

    ////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////// Evaluate Functions ///////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

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

    /// This is the same as `evaluate`, just lifted over a `Word8`.
    /// This function will panic if any of the `WireId`s evaluate to
    /// something other than `zero()` or `one()`. In other words, this
    /// will panic if you get a non-binary value in one of the
    /// `WireId`s.
    ///
    /// Note: I wrote this with a if statement (which is not as nice
    /// to read as a match statement). This is done because of a rustc
    /// bug (at time of writing) that causes a compiler panic if you
    /// try to match on a `const` function like `zero()`.
    pub fn evaluate_word8(&mut self, word: &Word8) -> u8 {
        let mut num: u8 = 0;
        word.iter().enumerate().for_each(|(i, &wire)| {
            let eval = self.evaluate(wire);
            if eval == T::zero() {
            } else if eval == T::one() {
                num ^= 0b0000_0001 << i;
            } else {
                panic!("evaluate_word8: the evaluation of a wireId is neither 0 or 1");
            }
        });
        num
    }

    pub fn evaluate_word64(&mut self, word: &Word64) -> u64 {
        let mut arr: [u8; 8] = [0; 8];
        word.iter()
            .enumerate()
            .for_each(|(i, word8)| arr[i] = self.evaluate_word8(word8));
        types::from_ne_u64(arr)
    }

    pub fn evaluate_word8_to_vec<'a>(
        &mut self,
        stream: impl Iterator<Item = &'a Word8>,
    ) -> Vec<u8> {
        stream.map(|wrd8| self.evaluate_word8(wrd8)).collect()
    }

    pub fn evaluate_word8_to_array<'a>(
        &mut self,
        stream: impl Iterator<Item = &'a Word8>,
        output: &'a mut [u8],
    ) {
        stream
            .zip(0..output.len())
            .for_each(|(wrd8, i)| output[i] = self.evaluate_word8(&wrd8));
    }

    pub fn evaluate_wire_ids_as_word8s(&mut self, stream: impl Iterator<Item = WireId>) -> Vec<u8> {
        stream
            .chunks(8)
            .into_iter()
            .map(|chunk| self.evaluate_word8(&types::to_word8(chunk)))
            .collect()
    }

    // NOTE: only used internally as a convenience for testing
    fn evaluate_keccakmatrix(&mut self, matrix: &[Word64; 25]) -> [u64; 25] {
        let mut arr: [u64; 25] = [0; 25];
        (0..25).for_each(|x| arr[x] = self.evaluate_word64(&matrix[x]));
        arr
    }

    ////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////// Reset  ///////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

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

    ////////////////////////////////////////////////////////////////////////////////
    /////////////////////// Simple Binary Wire Functions ///////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

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

    ////////////////////////////////////////////////////////////////////////////////
    //////////////////////// Word8/Word64 Binary Functions /////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    /// NOTE: inputs must have at least one Word64 in array
    ///
    pub fn u64_fan_in<'a, F>(
        &mut self,
        mut inputs: impl Iterator<Item = &'a Word64>,
        mut gate: F,
    ) -> Word64
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        let base_case: Word64 = *inputs
            .next()
            .expect("u64_fan_in: input iterator must have at least one element");

        let mut output = Word64::default();

        inputs.fold(base_case, |acc, next| {
            acc.iter()
                .zip(next.iter())
                .flat_map(|(l, r)| l.iter().zip(r.iter()))
                .zip(iproduct!(0..8, 0..8))
                .for_each(|((&l, &r), (i, j))| output[i][j] = gate(self, l, r));
            output
        });
        output
    }

    /// Requires that all left and right inputs in array are either 0 or 1
    pub fn u8_fan_in<'a, F>(
        &mut self,
        mut inputs: impl Iterator<Item = &'a Word8>,
        mut gate: F,
    ) -> Word8
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        let mut wrd8: Word8 = *inputs
            .next()
            .expect("u8_fan_in: input iterator must have at least one element");
        inputs.fold(wrd8, |acc, next| {
            acc.iter()
                .zip(next.iter())
                .enumerate()
                .for_each(|(i, (&l, &r))| wrd8[i] = gate(self, l, r));
            wrd8
        });
        wrd8
    }

    pub fn u64_bitwise_op<F>(&mut self, left: &Word64, right: &Word64, mut gate: F) -> Word64
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        let mut wrd64 = Word64::default();
        left.iter()
            .zip(right.iter())
            .flat_map(|(l, r)| l.iter().zip(r.iter()))
            .zip(iproduct!(0..8, 0..8))
            .for_each(|((&l, &r), (i, j))| wrd64[i][j] = gate(self, l, r));
        wrd64
    }

    pub fn u8_bitwise_op<F>(&mut self, left: &Word8, right: &Word8, mut gate: F) -> Word8
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        let mut wrd8 = Word8::default();
        left.iter()
            .zip(right.iter())
            .enumerate()
            .for_each(|(i, (&l, &r))| wrd8[i] = gate(self, l, r));
        wrd8
    }

    pub fn u64_unary_op<F>(&mut self, input: &Word64, mut gate: F) -> Word64
    where
        F: FnMut(&mut Self, WireId) -> WireId,
    {
        let mut wrd64 = Word64::default();
        input
            .iter()
            .flat_map(|x| x.iter())
            .zip(iproduct!(0..8, 0..8))
            .for_each(|(&x, (i, j))| wrd64[i][j] = gate(self, x));
        wrd64
    }

    pub fn u8_unary_op<F>(&mut self, input: &Word8, mut gate: F) -> Word8
    where
        F: FnMut(&mut Self, WireId) -> WireId,
    {
        let mut wrd8 = Word8::default();
        input
            .iter()
            .enumerate()
            .for_each(|(i, &x)| wrd8[i] = gate(self, x));
        wrd8
    }

    ////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////// Comparison Functions /////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    /// Requires that both the left and right inputs are either 0 or 1
    fn new_less_than(&mut self, left: WireId, right: WireId) -> WireId {
        let left_not = self.new_not(left);
        self.new_and(left_not, right)
    }

    /// Requires that both the left and right inputs are either 0 or 1
    fn new_greater_than(&mut self, left: WireId, right: WireId) -> WireId {
        let right_not = self.new_not(right);
        self.new_and(left, right_not)
    }

    fn new_equality(&mut self, left: WireId, right: WireId) -> WireId {
        let xor = self.new_xor(left, right);
        self.new_not(xor)
    }

    fn new_word8_greater_than(&mut self, left: Word8, right: Word8) -> WireId {
        let cmp7 = self.new_greater_than(left[7], right[7]);
        let eq7 = self.new_equality(left[7], right[7]);

        let cmp6 = self.new_greater_than(left[6], right[6]);
        let eq6 = self.new_equality(left[6], right[6]);

        let cmp5 = self.new_greater_than(left[5], right[5]);
        let eq5 = self.new_equality(left[5], right[5]);

        let cmp4 = self.new_greater_than(left[4], right[4]);
        let eq4 = self.new_equality(left[4], right[4]);

        let cmp3 = self.new_greater_than(left[3], right[3]);
        let eq3 = self.new_equality(left[3], right[3]);

        let cmp2 = self.new_greater_than(left[2], right[2]);
        let eq2 = self.new_equality(left[2], right[2]);

        let cmp1 = self.new_greater_than(left[1], right[1]);
        let eq1 = self.new_equality(left[1], right[1]);

        let cmp0 = self.new_greater_than(left[0], right[0]);

        let wir6 = self.fan_in(&[cmp6, eq7], Circuit::new_and);
        let wir5 = self.fan_in(&[cmp5, eq7, eq6], Circuit::new_and);
        let wir4 = self.fan_in(&[cmp4, eq7, eq6, eq5], Circuit::new_and);
        let wir3 = self.fan_in(&[cmp3, eq7, eq6, eq5, eq4], Circuit::new_and);
        let wir2 = self.fan_in(&[cmp2, eq7, eq6, eq5, eq4, eq3], Circuit::new_and);
        let wir1 = self.fan_in(&[cmp1, eq7, eq6, eq5, eq4, eq3, eq2], Circuit::new_and);
        let wir0 = self.fan_in(&[cmp0, eq7, eq6, eq5, eq4, eq3, eq2, eq1], Circuit::new_and);

        self.fan_in(
            &[cmp7, wir6, wir5, wir4, wir3, wir2, wir1, wir0],
            Circuit::new_or,
        )
    }

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////// Keccak Functions ////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    fn keccakf_1600(&mut self, a: &mut [Word64; 25]) {
        for i in 0..24 {
            let mut array: [Word64; 5] = [Word64::default(); 5];

            // Theta
            unroll! {
                for x in 0..5 {
                    unroll! {
                        for y_count in 0..5 {
                            let y = y_count * 5;
                            array[x] = self.u64_bitwise_op(&array[x], &a[x + y], Circuit::new_xor);
                        }
                    }
                }
            }

            unroll! {
                for x in 0..5 {
                    unroll! {
                        for y_count in 0..5 {
                            let y = y_count * 5;
                            a[y + x] = self.u64_fan_in([a[y + x], array[(x + 4) % 5],
                                types::rotate_word64_left(array[(x + 1) % 5], 1)].iter(), Circuit::new_xor);
                        }
                    }
                }
            }

            // Rho and pi
            let mut _last = a[1];
            unroll! {
                for x in 0..24 {
                    array[0] = a[types::PI[x]];
                    a[types::PI[x]] = types::rotate_word64_left(_last, types::RHO[x]);
                    _last = array[0];
                }
            }

            // Chi
            unroll! {
                for y_step in 0..5 {
                    let y = y_step * 5;

                    unroll! {
                        for x in 0..5 {
                            array[x] = a[y + x];
                        }
                    }

                    unroll! {
                        for x in 0..5 {
                            let not = self.u64_unary_op(&array[(x + 1) % 5], Circuit::new_not);
                            let and = self.u64_bitwise_op(&not, &(array[(x + 2) % 5]), Circuit::new_and);
                            a[y + x] = self.u64_bitwise_op(&array[x], &and, Circuit::new_xor);

                        }
                    }
                }
            };

            // Iota
            let rc_num = self.const_word64(types::RC[i]);
            a[0] = self.u64_bitwise_op(&a[0], &rc_num, Circuit::new_xor);
        }
    }

    fn squeeze(&mut self, keccak: &mut KeccakInternal, output: &mut [Word8]) {
        fn setout(src: &[Word8], dst: &mut [Word8], len: usize) {
            dst[..len].copy_from_slice(&src[..len]);
        }

        let mut op = 0;
        let mut l = output.len();
        while l >= keccak.rate {
            setout(&keccak.a_bytes(), &mut output[op..], keccak.rate);
            self.keccakf_1600(&mut keccak.a);
            op += keccak.rate;
            l -= keccak.rate;
        }

        setout(&keccak.a_bytes(), &mut output[op..], l);
    }

    fn absorb(&mut self, keccak: &mut KeccakInternal, input: &[Word8]) {
        let mut ip = 0;
        let mut l = input.len();
        let mut rate = keccak.rate - keccak.offset;
        let mut offset = keccak.offset;
        while l >= rate {
            self.xorin(keccak, &input[ip..], None);
            self.keccakf_1600(&mut keccak.a);
            ip += rate;
            l -= rate;
            rate = keccak.rate;
            offset = 0;
        }

        self.xorin(keccak, &input[ip..], Some(l));
        keccak.offset = offset + l;
    }

    fn xorin(&mut self, keccak: &mut KeccakInternal, src: &[Word8], limit: Option<usize>) {
        let l = match limit {
            None => keccak.rate,
            Some(l) => l,
        };
        keccak
            .a
            .iter_mut()
            .flat_map(|wrd64| wrd64.iter_mut())
            .skip(keccak.offset) // Start slice from here
            .take(l) // End slice at position l
            .zip(src.iter())
            .for_each(|(keccak_wrd8, src_wrd8): (&mut Word8, &Word8)| {
                *keccak_wrd8 = self.u8_bitwise_op(keccak_wrd8, src_wrd8, Circuit::new_xor)
            });
    }

    /// This padding idea is also taken from tiny_keccak where
    /// effectively no padding is done before the absorb phase, but is
    /// xor'ed into the internal matrix after absorbing all input.
    ///
    /// NOTE: this is contrary to the way keccak is explained in its
    /// documentation, but is identical in result.
    ///
    fn pad(&mut self, keccak: &mut KeccakInternal) {
        let offset = keccak.offset;
        let rate = keccak.rate;
        let delim = self.const_word8(keccak.delim);

        let tail = self.const_word8(0x80);

        // Offset is out of 200 (25 * 8) which is the number of u8 in
        // the keccak matrix. Here I am selecting the `Word8` at
        // `offest` by first finding which `Word64` and then which
        // `Word8` in that `Word64`.
        //
        keccak.a[offset / 8][offset % 8] =
            self.u8_bitwise_op(&keccak.a[offset / 8][offset % 8], &delim, Circuit::new_xor);

        keccak.a[(rate - 1) / 8][(rate - 1) % 8] = self.u8_bitwise_op(
            &keccak.a[(rate - 1) / 8][(rate - 1) % 8],
            &tail,
            Circuit::new_xor,
        );
    }

    fn finalize(&mut self, keccak: &mut KeccakInternal, output: &mut [Word8]) {
        self.pad(keccak);
        self.keccakf_1600(&mut keccak.a);
        self.squeeze(keccak, output);
    }

    ///
    /// ```
    /// use zksnark::field::z251::Z251;
    /// use zksnark::groth16::circuit::*;
    ///
    /// const BYTES: usize = 56;
    ///
    /// let input: &mut [u8; BYTES] = &mut [
    ///     150, 234, 20, 196, 120, 146, 1, 48, 157, 10, 170, 174, 183, 246, 34, 204, 110, 184, 31,
    ///     155, 70, 130, 115, 205, 179, 165, 27, 165, 104, 31, 7, 16, 157, 242, 34, 232, 56, 161, 8,
    ///     150, 228, 129, 153, 41, 144, 186, 190, 41, 16, 59, 242, 109, 102, 75, 12, 246,
    /// ];
    ///
    /// let mut circuit = Circuit::<Z251>::new();
    /// let circuit_input: &mut [Word8; BYTES] = &mut [Word8::default(); BYTES];
    /// circuit.set_new_word8_array(input.iter(), circuit_input);
    /// let circuit_output: [Word8; 32] = circuit.keccak256(circuit_input);
    ///
    ///
    /// let eval_circuit_output: &mut [u8; 32] = &mut [0; 32];
    /// circuit.evaluate_word8_to_array(circuit_output.iter(), eval_circuit_output);
    ///
    /// assert_eq!(*eval_circuit_output,
    ///     [65, 231, 91, 68, 62, 80, 71, 123, 164, 102, 65, 50, 133
    ///     , 1, 30, 28, 212, 25, 134, 124, 67, 29, 5, 47, 16, 36, 248
    ///     , 235, 214, 168, 145, 209]);
    /// ```
    ///
    pub fn keccak256(&mut self, input: &[Word8]) -> [Word8; 32] {
        let keccak = &mut KeccakInternal {
            a: self.initial_keccakmatrix(),
            offset: 0,
            rate: (200 - (256 / 4)),
            delim: 0x01,
        };
        self.absorb(keccak, input);
        let output: &mut [Word8; 32] = &mut [Word8::default(); 32];
        self.finalize(keccak, output);
        *output
    }

    pub fn keccak256_stream<'a>(
        &mut self,
        input: impl IntoIterator<Item = &'a Word8>,
    ) -> [Word8; 32] {
        let keccak = &mut KeccakInternal {
            a: self.initial_keccakmatrix(),
            offset: 0,
            rate: (200 - (256 / 4)),
            delim: 0x01,
        };
        input
            .into_iter()
            .for_each(|&wrd8| self.absorb(keccak, &[wrd8]));
        let output: &mut [Word8; 32] = &mut [Word8::default(); 32];
        self.finalize(keccak, output);
        *output
    }
}
