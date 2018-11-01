use super::super::super::field::Field;
use std::collections::HashMap;
use std::iter::FromIterator;

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
#[derive(Debug, Default)]
pub struct Word64 {
    pub wire64: WireId,
    pub wire63: WireId,
    pub wire62: WireId,
    pub wire61: WireId,
    pub wire60: WireId,
    pub wire59: WireId,
    pub wire58: WireId,
    pub wire57: WireId,
    pub wire56: WireId,
    pub wire55: WireId,
    pub wire54: WireId,
    pub wire53: WireId,
    pub wire52: WireId,
    pub wire51: WireId,
    pub wire50: WireId,
    pub wire49: WireId,
    pub wire48: WireId,
    pub wire47: WireId,
    pub wire46: WireId,
    pub wire45: WireId,
    pub wire44: WireId,
    pub wire43: WireId,
    pub wire42: WireId,
    pub wire41: WireId,
    pub wire40: WireId,
    pub wire39: WireId,
    pub wire38: WireId,
    pub wire37: WireId,
    pub wire36: WireId,
    pub wire35: WireId,
    pub wire34: WireId,
    pub wire33: WireId,
    pub wire32: WireId,
    pub wire31: WireId,
    pub wire30: WireId,
    pub wire29: WireId,
    pub wire28: WireId,
    pub wire27: WireId,
    pub wire26: WireId,
    pub wire25: WireId,
    pub wire24: WireId,
    pub wire23: WireId,
    pub wire22: WireId,
    pub wire21: WireId,
    pub wire20: WireId,
    pub wire19: WireId,
    pub wire18: WireId,
    pub wire17: WireId,
    pub wire16: WireId,
    pub wire15: WireId,
    pub wire14: WireId,
    pub wire13: WireId,
    pub wire12: WireId,
    pub wire11: WireId,
    pub wire10: WireId,
    pub wire9: WireId,
    pub wire8: WireId,
    pub wire7: WireId,
    pub wire6: WireId,
    pub wire5: WireId,
    pub wire4: WireId,
    pub wire3: WireId,
    pub wire2: WireId,
    pub wire1: WireId,
}

impl FromIterator<WireId> for Word64 {
    fn from_iter<I: IntoIterator<Item = WireId>>(iter: I) -> Self {
        let mut i = iter.into_iter();
        Word64 {
            wire1: i.next().expect("Missing WireId for wire1, (FromIterator)"),
            wire2: i.next().expect("Missing WireId for wire2, (FromIterator)"),
            wire3: i.next().expect("Missing WireId for wire3, (FromIterator)"),
            wire4: i.next().expect("Missing WireId for wire4, (FromIterator)"),
            wire5: i.next().expect("Missing WireId for wire5, (FromIterator)"),
            wire6: i.next().expect("Missing WireId for wire6, (FromIterator)"),
            wire7: i.next().expect("Missing WireId for wire7, (FromIterator)"),
            wire8: i.next().expect("Missing WireId for wire8, (FromIterator)"),
            wire9: i.next().expect("Missing WireId for wire9, (FromIterator)"),
            wire10: i.next().expect("Missing WireId for wire10, (FromIterator)"),
            wire11: i.next().expect("Missing WireId for wire11, (FromIterator)"),
            wire12: i.next().expect("Missing WireId for wire12, (FromIterator)"),
            wire13: i.next().expect("Missing WireId for wire13, (FromIterator)"),
            wire14: i.next().expect("Missing WireId for wire14, (FromIterator)"),
            wire15: i.next().expect("Missing WireId for wire15, (FromIterator)"),
            wire16: i.next().expect("Missing WireId for wire16, (FromIterator)"),
            wire17: i.next().expect("Missing WireId for wire17, (FromIterator)"),
            wire18: i.next().expect("Missing WireId for wire18, (FromIterator)"),
            wire19: i.next().expect("Missing WireId for wire19, (FromIterator)"),
            wire20: i.next().expect("Missing WireId for wire20, (FromIterator)"),
            wire21: i.next().expect("Missing WireId for wire21, (FromIterator)"),
            wire22: i.next().expect("Missing WireId for wire22, (FromIterator)"),
            wire23: i.next().expect("Missing WireId for wire23, (FromIterator)"),
            wire24: i.next().expect("Missing WireId for wire24, (FromIterator)"),
            wire25: i.next().expect("Missing WireId for wire25, (FromIterator)"),
            wire26: i.next().expect("Missing WireId for wire26, (FromIterator)"),
            wire27: i.next().expect("Missing WireId for wire27, (FromIterator)"),
            wire28: i.next().expect("Missing WireId for wire28, (FromIterator)"),
            wire29: i.next().expect("Missing WireId for wire29, (FromIterator)"),
            wire30: i.next().expect("Missing WireId for wire30, (FromIterator)"),
            wire31: i.next().expect("Missing WireId for wire31, (FromIterator)"),
            wire32: i.next().expect("Missing WireId for wire32, (FromIterator)"),
            wire33: i.next().expect("Missing WireId for wire33, (FromIterator)"),
            wire34: i.next().expect("Missing WireId for wire34, (FromIterator)"),
            wire35: i.next().expect("Missing WireId for wire35, (FromIterator)"),
            wire36: i.next().expect("Missing WireId for wire36, (FromIterator)"),
            wire37: i.next().expect("Missing WireId for wire37, (FromIterator)"),
            wire38: i.next().expect("Missing WireId for wire38, (FromIterator)"),
            wire39: i.next().expect("Missing WireId for wire39, (FromIterator)"),
            wire40: i.next().expect("Missing WireId for wire40, (FromIterator)"),
            wire41: i.next().expect("Missing WireId for wire41, (FromIterator)"),
            wire42: i.next().expect("Missing WireId for wire42, (FromIterator)"),
            wire43: i.next().expect("Missing WireId for wire43, (FromIterator)"),
            wire44: i.next().expect("Missing WireId for wire44, (FromIterator)"),
            wire45: i.next().expect("Missing WireId for wire45, (FromIterator)"),
            wire46: i.next().expect("Missing WireId for wire46, (FromIterator)"),
            wire47: i.next().expect("Missing WireId for wire47, (FromIterator)"),
            wire48: i.next().expect("Missing WireId for wire48, (FromIterator)"),
            wire49: i.next().expect("Missing WireId for wire49, (FromIterator)"),
            wire50: i.next().expect("Missing WireId for wire50, (FromIterator)"),
            wire51: i.next().expect("Missing WireId for wire51, (FromIterator)"),
            wire52: i.next().expect("Missing WireId for wire52, (FromIterator)"),
            wire53: i.next().expect("Missing WireId for wire53, (FromIterator)"),
            wire54: i.next().expect("Missing WireId for wire54, (FromIterator)"),
            wire55: i.next().expect("Missing WireId for wire55, (FromIterator)"),
            wire56: i.next().expect("Missing WireId for wire56, (FromIterator)"),
            wire57: i.next().expect("Missing WireId for wire57, (FromIterator)"),
            wire58: i.next().expect("Missing WireId for wire58, (FromIterator)"),
            wire59: i.next().expect("Missing WireId for wire59, (FromIterator)"),
            wire60: i.next().expect("Missing WireId for wire60, (FromIterator)"),
            wire61: i.next().expect("Missing WireId for wire61, (FromIterator)"),
            wire62: i.next().expect("Missing WireId for wire62, (FromIterator)"),
            wire63: i.next().expect("Missing WireId for wire63, (FromIterator)"),
            wire64: i.next().expect("Missing WireId for wire64, (FromIterator)"),
        }
    }
}

impl IntoIterator for Word64 {
    type Item = WireId;
    type IntoIter = Word64Iterator;

    fn into_iter(self) -> Self::IntoIter {
        Word64Iterator {
            curr: 1,
            word64: Word64::default(),
        }
    }
}

pub struct Word64Iterator {
    curr: u8,
    word64: Word64,
}

impl Iterator for Word64Iterator {
    type Item = WireId;

    fn next(&mut self) -> Option<WireId> {
        let current = self.curr;
        self.curr = self.curr + 1;
        match current {
            1 => Some(self.word64.wire1),
            2 => Some(self.word64.wire2),
            3 => Some(self.word64.wire3),
            4 => Some(self.word64.wire4),
            5 => Some(self.word64.wire5),
            6 => Some(self.word64.wire6),
            7 => Some(self.word64.wire7),
            8 => Some(self.word64.wire8),
            9 => Some(self.word64.wire9),
            10 => Some(self.word64.wire10),
            11 => Some(self.word64.wire11),
            12 => Some(self.word64.wire12),
            13 => Some(self.word64.wire13),
            14 => Some(self.word64.wire14),
            15 => Some(self.word64.wire15),
            16 => Some(self.word64.wire16),
            17 => Some(self.word64.wire17),
            18 => Some(self.word64.wire18),
            19 => Some(self.word64.wire19),
            20 => Some(self.word64.wire20),
            21 => Some(self.word64.wire21),
            22 => Some(self.word64.wire22),
            23 => Some(self.word64.wire23),
            24 => Some(self.word64.wire24),
            25 => Some(self.word64.wire25),
            26 => Some(self.word64.wire26),
            27 => Some(self.word64.wire27),
            28 => Some(self.word64.wire28),
            29 => Some(self.word64.wire29),
            30 => Some(self.word64.wire30),
            31 => Some(self.word64.wire31),
            32 => Some(self.word64.wire32),
            33 => Some(self.word64.wire33),
            34 => Some(self.word64.wire34),
            35 => Some(self.word64.wire35),
            36 => Some(self.word64.wire36),
            37 => Some(self.word64.wire37),
            38 => Some(self.word64.wire38),
            39 => Some(self.word64.wire39),
            40 => Some(self.word64.wire40),
            41 => Some(self.word64.wire41),
            42 => Some(self.word64.wire42),
            43 => Some(self.word64.wire43),
            44 => Some(self.word64.wire44),
            45 => Some(self.word64.wire45),
            46 => Some(self.word64.wire46),
            47 => Some(self.word64.wire47),
            48 => Some(self.word64.wire48),
            49 => Some(self.word64.wire49),
            50 => Some(self.word64.wire50),
            51 => Some(self.word64.wire51),
            52 => Some(self.word64.wire52),
            53 => Some(self.word64.wire53),
            54 => Some(self.word64.wire54),
            55 => Some(self.word64.wire55),
            56 => Some(self.word64.wire56),
            57 => Some(self.word64.wire57),
            58 => Some(self.word64.wire58),
            59 => Some(self.word64.wire59),
            60 => Some(self.word64.wire60),
            61 => Some(self.word64.wire61),
            62 => Some(self.word64.wire62),
            63 => Some(self.word64.wire63),
            64 => Some(self.word64.wire64),
            _ => None,
        }
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
    /// assert_eq!(circuit.evaluate(u64_input.wire1), Z251::from(0));
    /// assert_eq!(circuit.evaluate(u64_input.wire2), Z251::from(0));
    /// assert_eq!(circuit.evaluate(u64_input.wire3), Z251::from(0));
    /// assert_eq!(circuit.evaluate(u64_input.wire4), Z251::from(1));
    ///
    /// assert_eq!(circuit.evaluate(u64_input.wire5), Z251::from(1));
    /// assert_eq!(circuit.evaluate(u64_input.wire6), Z251::from(1));
    /// assert_eq!(circuit.evaluate(u64_input.wire7), Z251::from(1));
    /// assert_eq!(circuit.evaluate(u64_input.wire8), Z251::from(0));
    ///
    /// // ...
    /// ```
    pub fn new_u64(&mut self) -> Word64 {
        Word64::default()
            .into_iter()
            .map(|_| self.new_wire())
            .collect()
        // Word64 {
        //     wire64: self.new_wire(),
        //     wire63: self.new_wire(),
        //     wire62: self.new_wire(),
        //     wire61: self.new_wire(),
        //     wire60: self.new_wire(),
        //     wire59: self.new_wire(),
        //     wire58: self.new_wire(),
        //     wire57: self.new_wire(),
        //     wire56: self.new_wire(),
        //     wire55: self.new_wire(),
        //     wire54: self.new_wire(),
        //     wire53: self.new_wire(),
        //     wire52: self.new_wire(),
        //     wire51: self.new_wire(),
        //     wire50: self.new_wire(),
        //     wire49: self.new_wire(),
        //     wire48: self.new_wire(),
        //     wire47: self.new_wire(),
        //     wire46: self.new_wire(),
        //     wire45: self.new_wire(),
        //     wire44: self.new_wire(),
        //     wire43: self.new_wire(),
        //     wire42: self.new_wire(),
        //     wire41: self.new_wire(),
        //     wire40: self.new_wire(),
        //     wire39: self.new_wire(),
        //     wire38: self.new_wire(),
        //     wire37: self.new_wire(),
        //     wire36: self.new_wire(),
        //     wire35: self.new_wire(),
        //     wire34: self.new_wire(),
        //     wire33: self.new_wire(),
        //     wire32: self.new_wire(),
        //     wire31: self.new_wire(),
        //     wire30: self.new_wire(),
        //     wire29: self.new_wire(),
        //     wire28: self.new_wire(),
        //     wire27: self.new_wire(),
        //     wire26: self.new_wire(),
        //     wire25: self.new_wire(),
        //     wire24: self.new_wire(),
        //     wire23: self.new_wire(),
        //     wire22: self.new_wire(),
        //     wire21: self.new_wire(),
        //     wire20: self.new_wire(),
        //     wire19: self.new_wire(),
        //     wire18: self.new_wire(),
        //     wire17: self.new_wire(),
        //     wire16: self.new_wire(),
        //     wire15: self.new_wire(),
        //     wire14: self.new_wire(),
        //     wire13: self.new_wire(),
        //     wire12: self.new_wire(),
        //     wire11: self.new_wire(),
        //     wire10: self.new_wire(),
        //     wire9: self.new_wire(),
        //     wire8: self.new_wire(),
        //     wire7: self.new_wire(),
        //     wire6: self.new_wire(),
        //     wire5: self.new_wire(),
        //     wire4: self.new_wire(),
        //     wire3: self.new_wire(),
        //     wire2: self.new_wire(),
        //     wire1: self.new_wire(),
        // }
    }

    /// set the values for a `Word64` from a u64.
    ///
    /// See `new_u64` for example
    pub fn set_u64(&mut self, u64_wires: &Word64, input: u64) {
        let mut n = input;
        self.set_value(u64_wires.wire1, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire2, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire3, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire4, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire5, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire6, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire7, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire8, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire9, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire10, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire11, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire12, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire13, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire14, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire15, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire16, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire17, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire18, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire19, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire20, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire21, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire22, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire23, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire24, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire25, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire26, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire27, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire28, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire29, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire30, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire31, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire32, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire33, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire34, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire35, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire36, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire37, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire38, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire39, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire40, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire41, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire42, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire43, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire44, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire45, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire46, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire47, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire48, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire49, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire50, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire51, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire52, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire53, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire54, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire55, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire56, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire57, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire58, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire59, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire60, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire61, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire62, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire63, T::from((n % 2) as usize));
        n = n >> 1;
        self.set_value(u64_wires.wire64, T::from((n % 2) as usize));
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

    // /// I assume the input on each wire is a u8
    // pub fn keccak512_72(&mut self, input: &[WireId; 72]) -> &[WireId; 64] {
    //     unimplemented!();
    // }
}
