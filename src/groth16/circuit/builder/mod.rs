use super::super::super::field::Field;
use std::collections::HashMap;
use std::fmt;
use std::iter::repeat;

extern crate itertools;
use itertools::Itertools;

#[cfg(test)]
mod tests;

pub mod types;
use self::types::*;

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

#[derive(Clone, Copy, Eq, Hash, PartialEq, Default)]
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

impl<T> Circuit<T>
where
    T: Copy + Field,
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

    pub fn new_wire(&mut self) -> WireId {
        let next_wire_id = self.next_wire_id;
        self.next_wire_id.0 += 1;
        self.wire_values.insert(next_wire_id, None);
        next_wire_id
    }

    /// Creates a new u8 "number", but this is not the right way to think about
    /// it. Really it a conduit that accepts a u8 number as input where the
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

    // pub fn new_bitstream(&mut self) -> impl Iterator<Item = WireId> {
    //     repeat_with(|| self.new_wire())
    // }

    // pub fn set_new_bitstream(&mut self, input: &[Binary]) -> impl Iterator<Item = WireId> {
    //     repeat_with(|| self.new_wire()).zip(input.iter()).map(|(w, b)| )
    // }

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

    pub fn const_word8(&mut self, input: u8) -> Word8 {
        let mut n = input;
        let mut wrd8: Word8 = Word8::default();
        (0..8).for_each(|i| {
            if n % 2 == 0 {
                wrd8[i] = self.zero_wire();
            } else {
                wrd8[i] = self.unity_wire();
            }
            n = n >> 1;
        });
        wrd8
    }

    pub fn const_word64(&mut self, input: u64) -> Word64 {
        let mut wrd64: Word64 = Word64::default();
        to_ne_u8(input)
            .iter()
            .enumerate()
            .for_each(|(i, &num)| wrd64[i] = self.const_word8(num));
        wrd64
    }

    /// set the values for a `Word8` from a u8.
    ///
    /// See `new_u8` for example
    ///
    pub fn set_word8(&mut self, u8_wires: &Word8, input: u8) {
        let mut n = input;
        u8_wires.iter().for_each(|&wire_id| {
            if n % 2 == 0 {
                self.set_value(wire_id, T::zero());
            } else {
                self.set_value(wire_id, T::one());
            }
            n = n >> 1;
        });
    }

    /// Set the values for a `Word64` from a u64.
    ///
    /// See `new_u8` for example
    ///
    pub fn set_word64(&mut self, u64_wires: &Word64, input: u64) {
        u64_wires
            .iter()
            .zip(to_ne_u8(input).iter())
            .for_each(|(word, &num)| self.set_word8(word, num));
    }

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

    /// Create a vector of pre-set inputs for a circuit.
    ///
    /// //```
    /// //use zksnark::field::z251::Z251;
    /// //use zksnark::field::*;
    /// //use zksnark::groth16::circuit::*;
    ///
    /// //// Create an empty circuit
    /// //let mut circuit = Circuit::<Z251>::new();
    ///
    /// //let external_input: [u8; 7] = [9, 24, 45, 250, 99, 0, 7];
    /// //let circuit_input: Vec<Word8> = circuit.set_new_word8_stream(external_input.iter());
    ///
    /// //assert_eq!(circuit.evaluate_word8_stream(circuit_input.iter()), external_input.iter().collect());
    /// //```
    ///
    pub fn set_new_word8_stream<'a>(
        &mut self,
        input: impl Iterator<Item = &'a u8>,
        output: &'a mut [Word8],
    ) {
        input
            .zip(0..output.len())
            .for_each(|(num, i)| output[i] = self.set_new_word8(*num));
    }

    pub fn unity_wire(&self) -> WireId {
        WireId(1)
    }

    pub fn num_wires(&self) -> usize {
        self.next_wire_id.0
    }

    pub fn set_value(&mut self, wire: WireId, value: T) {
        self.wire_values.insert(wire, Some(value));
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
        from_ne_u64(arr)
    }

    pub fn evaluate_word64_stream(&mut self, stream: impl Iterator<Item = Word64>) -> Vec<u64> {
        stream.map(|wrd64| self.evaluate_word64(&wrd64)).collect()
    }

    pub fn evaluate_word8_stream<'a>(
        &mut self,
        stream: impl Iterator<Item = &'a Word8>,
        output: &'a mut [u8],
    ) {
        stream
            .zip(0..output.len())
            .for_each(|(wrd8, i)| output[i] = self.evaluate_word8(&wrd8));
    }

    // NOTE: only used internally as a convenience for testing
    fn evaluate_keccakmatrix(&mut self, matrix: &[Word64; 25]) -> [u64; 25] {
        let mut arr: [u64; 25] = [0; 25];
        (0..25).for_each(|x| arr[x] = self.evaluate_word64(&matrix[x]));
        arr
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
    ///
    /// NOTE: I wish Rust would let me define this in terms of `fan_in`, but for
    /// some reason you cannot pass FnMut to inner functions.
    ///
    pub fn u64_fan_in<'a, F>(
        &mut self,
        inputs: impl Iterator<Item = &'a Word64>,
        mut gate: F,
    ) -> Word64
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        let mut wrd64 = Word64::default();
        inputs.fold(wrd64, |acc, next| {
            acc.iter()
                .zip(next.iter())
                .flat_map(|(l, r)| l.iter().zip(r.iter()))
                .zip(iproduct!(0..8, 0..8))
                .for_each(|((&l, &r), (i, j))| wrd64[i][j] = gate(self, l, r));
            wrd64
        });
        wrd64
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
                                rotate_word64_left(array[(x + 1) % 5], 1)].iter(), Circuit::new_xor);
                        }
                    }
                }
            }

            // Rho and pi
            let mut _last = a[1];
            unroll! {
                for x in 0..24 {
                    array[0] = a[PI[x]];
                    a[PI[x]] = rotate_word64_left(_last, RHO[x]);
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
            let rc_num = self.const_word64(RC[i]);
            a[0] = self.u64_bitwise_op(&a[0], &rc_num, Circuit::new_xor);
        }
    }

    fn squeeze(&mut self, keccak: &mut KeccakInternal, output: &mut [Word8]) {
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

    pub fn xorin(&mut self, keccak: &mut KeccakInternal, src: &[Word8], limit: Option<usize>) {
        let l = match limit {
            None => keccak.rate,
            Some(l) => l,
        };
        keccak
            .a
            .iter_mut()
            .flat_map(|wrd64| wrd64.iter_mut())
            .skip(keccak.offset) // Start slice from here
            .take(l - keccak.offset) // End slice at position l
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

    /// TODO: Use a slice instead of a Vec for the argument type.
    pub fn rotate_wires(mut wires: Vec<WireId>, n: usize) -> Vec<WireId> {
        let mut tail = wires.split_off(n);
        tail.append(&mut wires);
        tail
    }

    pub fn wires_from_literal(&self, mut literal: u128) -> Vec<WireId> {
        let mut bits = Vec::with_capacity(128 - literal.leading_zeros() as usize);

        while literal != 0 {
            let wire = match literal % 2 {
                0 => self.zero_wire(),
                1 => self.unity_wire(),
                _ => unreachable!(),
            };

            bits.push(wire);
            literal >>= 1;
        }

        bits
    }
}
