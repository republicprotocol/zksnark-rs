use super::super::super::field::Field;
use std::collections::HashMap;
use std::fmt;
use std::iter::repeat;

extern crate itertools;
use itertools::Itertools;

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
    /// assert_eq!(circuit.evaluate_word8(u8_input), 0b0000_0010);
    ///
    /// // ...
    /// ```
    pub fn new_word8(&mut self) -> Word8 {
        (0..8).map(|_| self.new_wire()).collect()
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
    /// assert_eq!(circuit.evaluate_word64(u64_input), 1);
    /// ```
    pub fn new_word64(&mut self) -> Word64 {
        (0..8).map(|_| self.new_word8()).collect()
    }

    fn new_keccakrow(&mut self) -> KeccakRow {
        (0..5).map(|_| self.new_word64()).collect()
    }

    fn new_keccakmatrix(&mut self) -> KeccakMatrix {
        (0..5).map(|_| self.new_keccakrow()).collect()
    }
    fn new_bitstream(&mut self, size: usize) -> Vec<WireId> {
        repeat(()).take(size).map(|_| self.new_wire()).collect()
    }

    /// set the values for a `Word8` from a u8.
    ///
    /// See `new_u8` for example
    ///
    /// NOTE: This converts the number to big-endian before setting the
    /// number so the right most bit of the u8 is set to the left most bit of
    /// the Word8. That way Word8 is bit little-endian.
    pub fn set_word8(&mut self, u8_wires: &Word8, input: u8) {
        let mut n = input.to_be();
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
    /// NOTE: This converts the number to little-endian before setting the
    /// number
    pub fn set_word64(&mut self, u64_wires: &Word64, input: u64) {
        u64_wires
            .iter()
            .zip(to_le_u8(input).iter())
            .for_each(|(word, &num)| self.set_word8(word, num));
    }

    fn set_keccakrow(&mut self, row: &KeccakRow, input: [u64; 5]) {
        row.iter()
            .zip(input.iter())
            .for_each(|(word, &num)| self.set_word64(word, num));
    }

    fn set_keccakmatrix(&mut self, matrix: &KeccakMatrix, input: [u64; 25]) {
        let mut m: [[u64; 5]; 5] = [[0; 5]; 5];
        input
            .iter()
            .chunks(5)
            .into_iter()
            .enumerate()
            .for_each(|(i, chunk)| {
                chunk
                    .into_iter()
                    .enumerate()
                    .for_each(|(j, &num)| m[i][j] = num)
            });

        matrix
            .iter()
            .zip(m.iter())
            .for_each(|(word, &row)| self.set_keccakrow(word, row));
    }

    // fn set_bitstream(&mut self, bit_stream: &Vec<WireId>, &Vec<

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

    /// NOTE: returns a u8 as big-endian
    pub fn evaluate_word8(&mut self, word: &Word8) -> u8 {
        let mut num: u8 = 0;
        word.iter().enumerate().for_each(|(i, &wire)| {
            let eval = self.evaluate(wire);
            if eval == T::zero() {
                // num.rotate_left(1);
            } else if eval == T::one() {
                num ^= 0b0000_0001 << i;
            } else {
                panic!("evaluate_word8: the evaluation of a wireId is neither 0 or 1");
            }
        });
        num
    }

    /// NOTE: returns a u64 as big-endian
    pub fn evaluate_word64(&mut self, word: &Word64) -> u64 {
        let mut arr: [u8; 8] = [0; 8];
        word.iter()
            .enumerate()
            .for_each(|(i, word8)| arr[i] = self.evaluate_word8(word8));
        from_le_u64(arr)
    }

    pub fn evaluate_keccakrow(&mut self, row: &KeccakRow) -> [u64; 5] {
        let mut arr: [u64; 5] = [0; 5];
        row.iter()
            .enumerate()
            .for_each(|(i, wrd64)| arr[i] = self.evaluate_word64(wrd64));
        arr
    }

    pub fn evaluate_keccakmatrix(&mut self, matrix: &KeccakMatrix) -> [u64; 25] {
        let mut arr: [u64; 25] = [0; 25];

        let rows: Vec<[u64; 5]> = matrix
            .into_iter()
            .map(|row| self.evaluate_keccakrow(row))
            .collect();

        rows.iter().enumerate().for_each(|(i, &row)| {
            row.iter()
                .enumerate()
                .for_each(|(j, &wrd64)| arr[(i * 5) + j] = wrd64)
        });
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
    fn u64_fan_in<'a, F>(&mut self, inputs: impl Iterator<Item = &'a Word64>, mut gate: F) -> Word64
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        inputs.fold(Word64::default(), |acc, next| {
            acc.iter()
                .zip(next.iter())
                .flat_map(|(l, r)| l.iter().zip(r.iter()))
                .map(|(&l, &r)| gate(self, l, r))
                .collect()
        })
    }

    fn u64_bitwise_op<F>(&mut self, left: &Word64, right: &Word64, mut gate: F) -> Word64
    where
        F: FnMut(&mut Self, WireId, WireId) -> WireId,
    {
        left.iter()
            .zip(right.iter())
            .flat_map(|(l, r)| l.iter().zip(r.iter()))
            .map(|(&l, &r)| gate(self, l, r))
            .collect()
    }

    fn u64_unary_op<F>(&mut self, input: &Word64, mut gate: F) -> Word64
    where
        F: FnMut(&mut Self, WireId) -> WireId,
    {
        input
            .iter()
            .flat_map(|x| x.iter())
            .map(|&i| gate(self, i))
            .collect()
    }

    /// # θ step
    /// C[x] = A[x,0] xor A[x,1] xor A[x,2] xor A[x,3] xor A[x,4],   for x in 0…4
    /// D[x] = C[x-1] xor rot(C[x+1],1),                             for x in 0…4
    /// A[x,y] = A[x,y] xor D[x],                           for (x,y) in (0…4,0…4)
    ///
    fn theta(&mut self, a: &mut KeccakMatrix) {
        let mut c: KeccakRow = (0..5)
            .map(|x: isize| {
                self.u64_fan_in(
                    [a[x][0], a[x][1], a[x][2], a[x][3], a[x][4]].iter(),
                    Circuit::new_xor,
                )
            }).collect();

        // NOTE: its rotate_right because Word64 is stored as bit little endian.
        (0..5).for_each(|x: isize| {
            c[x] = self.u64_fan_in(
                [c[x - 1], c[x + 1].rotate_right(1)].iter(),
                Circuit::new_xor,
            )
        });

        iproduct!(0..5, 0..5).for_each(|(x, y): (isize, isize)| {
            a[x][y] = self.u64_fan_in([a[x][y], c[x]].iter(), Circuit::new_xor)
        });
    }

    /// # ρ and π steps
    /// B[y,2*x+3*y] = rot(A[x,y], r[x,y]),                 for (x,y) in (0…4,0…4)
    ///
    /// TODO: what is ρ called
    ///
    /// # χ step
    /// A[x,y] = B[x,y] xor ((not B[x+1,y]) and B[x+2,y]),  for (x,y) in (0…4,0…4)
    ///
    /// TODO: What is χ called
    ///
    /// NOTE: I combined these two steps since the output of step2 is
    /// the input of step3 unlike the usual input of the internal matrix A.
    ///
    fn pi_step3(&mut self, a: &mut KeccakMatrix) {
        let r: KeccakMatrix = self.rotation_offsets();
        let b: KeccakMatrix = iproduct!(0..5, 0..5)
            .map(|(x, y): (isize, isize)| {
                self.u64_fan_in([a[x][y], r[x][y]].iter(), Circuit::new_xor)
            }).collect();

        let not_b: KeccakMatrix = iproduct!(0..5, 0..5)
            .map(|(x, y): (isize, isize)| self.u64_unary_op(&b[x + 1][y], Circuit::new_not))
            .collect();

        let and_not_b: KeccakMatrix = iproduct!(0..5, 0..5)
            .map(|(x, y): (isize, isize)| {
                self.u64_fan_in([not_b[x][y], b[x + 2][y]].iter(), Circuit::new_and)
            }).collect();

        iproduct!(0..5, 0..5).for_each(|(x, y): (isize, isize)| {
            a[x][y] = self.u64_fan_in([b[x][y], and_not_b[x][y]].iter(), Circuit::new_xor)
        });
    }

    /// # ι step
    /// A[0,0] = A[0,0] xor RC
    ///
    /// RC is RC[i] for i in 0..n-1 where n should be 24. In other words it
    /// changes with each iteration of the permutation
    ///
    fn last_step(&mut self, a: &mut KeccakMatrix, rc: u64) {
        let rc_num = self.new_word64();
        self.set_word64(&rc_num, rc);

        a[0][0] = self.u64_bitwise_op(&a[0][0], &rc_num, Circuit::new_xor);
    }

    /// const rotation_offset: KeccakMatrix<u64> = KeccakMatrix([
    ///     [0, 36, 3, 18, 41],
    ///     [1, 44, 10, 45, 2],
    ///     [62, 6, 43, 15, 61],
    ///     [28, 55, 25, 21, 56],
    ///     [27, 20, 39, 8, 14],
    /// ]);
    ///
    fn rotation_offsets(&mut self) -> KeccakMatrix {
        const OFFSET: [u64; 25] = [
            0, 36, 3, 18, 41, 1, 44, 10, 45, 2, 62, 6, 43, 15, 61, 28, 55, 25, 21, 56, 27, 20, 39,
            8, 14,
        ];
        let matrix: KeccakMatrix = self.new_keccakmatrix();
        self.set_keccakmatrix(&matrix, OFFSET);
        matrix
    }

    fn round(&mut self, a: &mut KeccakMatrix, rc: u64) {
        self.theta(a);
        self.pi_step3(a);
        self.last_step(a, rc)
    }

    /// Keccak-f[b](A) {
    ///  for i in 0…n-1
    ///    A = Round[b](A, RC[i])
    ///  return A
    /// }
    ///
    /// - The number of bits `b` is 1600 (there are 1600 wires in the KeccakMatrix)
    /// - the rate `r` + capacity `c` must be equal to 1600
    /// - the width `w` is 64
    /// - the number of rounds `n` is thus 24
    ///
    fn keccak_f1600(&mut self, a: &mut KeccakMatrix) {
        (0..24).for_each(|n| self.round(a, ROUND_CONSTANTS[n]))
    }

    /// Gives back output as long as you want.
    ///
    ///
    /// This is the more general version of keccak that only specifies the
    /// permutation input bits and eventually will handle padding as well. For
    /// reference these variables are implicitly set:
    /// - rate `r` + capacity `r` = 1600
    /// - width `w` is 64
    /// - rounds `n` is 24
    /// - bits `b` is 1600
    ///
    /// TODO: implement padding so you don't have to panic if you get input that
    /// is not exactly a single block length
    ///
    /// Since we do not handle padding now I assume the input is the correct
    /// block length
    ///
    /// `for (x,y) such that x+5*y < r/w`
    /// Where `x` and `y` are integers between 0 and 4 inclusive.
    ///
    /// Since `w` is always 64 for us that means when `r` is 1088 then the block
    /// length is 17 (the number of pairs) * 64 `w` which is 1088 bits.
    ///
    /// ## Pseudo-code
    ///
    /// # Absorbing phase
    /// for each block Pi in P
    ///   S[x,y] = S[x,y] xor Pi[x+5*y],          for (x,y) such that x+5*y < r/w
    ///   S = Keccak-f[r+c](S)
    ///
    /// # Squeezing phase
    /// Z = empty string
    /// while output is requested
    ///   Z = Z || S[x,y],                        for (x,y) such that x+5*y < r/w
    ///   S = Keccak-f[r+c](S)
    ///
    /// return Z
    ///
    /// ## Block size
    ///
    /// I have this here since I found this the most confusing part of keccak.
    /// The Block size is the rate `r`! So, if the rate is 1088 then there are
    /// 1088 bits in the block size. However, the internal state is made up of
    /// u64 numbers, not bits, so you interpret those 1088 bits as an array of
    /// u64 numbers. So, the block size is 17 u64 in this case. (r % 8) is the
    /// number of `Word8`s.
    ///
    /// NOTE: I assume that input is already padded
    ///
    fn keccak(
        &mut self,
        input: impl Iterator<Item = Word8>,
        r: usize,
        output_bit_size: usize,
    ) -> Vec<WireId> {
        assert!(r % 64 == 0);
        assert!(r < 1600);

        // S[x, y] = 0
        let s: &mut KeccakMatrix = &mut repeat(self.zero_wire()).take(1600).collect();;

        // Starts by turning the Word8s into word64s then collects them into
        // blocks based on the block size (unknown size at compile time).
        //
        // NOTE: if you mess up the padding the collect Word64 will fill the
        // blanks with zeros.
        //
        // # Absorbing phase
        // for each block Pi in P
        //   S[x,y] = S[x,y] xor Pi[x+5*y],          for (x,y) such that x+5*y < r/w
        //   S = Keccak-f[r+c](S)
        input
            .chunks(8)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Word64>())
            .chunks(r / 64)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Vec<Word64>>())
            .for_each(|block| {
                iproduct!(0..5, 0..5)
                    .filter(|(y, x)| x + 5 * y < (r as isize) / 64)
                    .for_each(|(x, y)| s[x][y] = block[(x + 5 * y) as usize]);
                self.keccak_f1600(s);
            });

        // # Squeezing phase
        // Z = empty string
        // while output is requested
        //   Z = Z || S[x,y],                        for (x,y) such that x+5*y < r/w
        //   S = Keccak-f[r+c](S)

        let mut z: Vec<WireId> = Vec::with_capacity(output_bit_size);

        while z.len() < output_bit_size {
            iproduct!(0..5, 0..5)
                .filter(|(x, y)| x + 5 * y < (r as isize) / 64)
                .for_each(|(x, y)| {
                    flatten_word64(s[x][y]).iter().for_each(|&x| {
                        if z.len() < output_bit_size {
                            z.push(x)
                        }
                    })
                });
            self.keccak_f1600(s);
        }

        //  return z
        z
    }

    /// TODO: Make input a bit stream and maybe make it a reference
    pub fn sha3_256(&mut self, bit_stream: Vec<Word64>) -> [Word64; 4] {
        unimplemented!();
        // let hash = self.keccak(bit_stream, 1088);
        // let mut tmp = [Word64::default(); 4];
        // hash.iter().enumerate().for_each(|(i, &word)| tmp[i] = word);
        // tmp
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
