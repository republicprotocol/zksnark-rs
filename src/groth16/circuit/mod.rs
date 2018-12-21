//! The purpose of this module is to build circuits into a representation that
//! can be turned into a `QAP` (Quadratic Arithmetic Program) which is defined
//! in "groth16/mod.rs". I'll start this off by describing what are the building
//! blocks for a `Circuit` are made of:
//!
//! # Sub Circuit
//!
//! You can think of a circuit as a collection of nodes connected by wires
//! without any cycles (Directed Acyclic Graph). Nodes are made up of
//! "sub circuit" which in reality look like this:
//!
//! ```
//! // `Left` input wires-> \|/ \|  <- `Right` input wires
//! //                       +   +  <- Plus operation
//! //                        \ /   <- (implicit connections, not wires)
//! //                         *    <- Multiplication operation
//! //                         |    <- `Output` wire
//! ```
//!
//! Lets walk through the example "sub circuit" which is the atomic unit of a
//! `Circuit`. In this example the left plus operation has 3 input wires and the
//! right plus operation has 2 input wires. Every plus operation must have at
//! least one input wire, but may have any number of extra input wires. Each
//! "wire" also has an associated weight and takes some input, unless it is an
//! `Output` wire. The way the "sub circuit" is evaluated is by evaluating all
//! input wires, followed by the plus operation and multiplication operation.
//!
//! ```
//! // For Example, given the above "sub circuit" we can evaluate it as follows:
//! //              We have these inputs and weights on the left wires:
//! //                 - [(1,1), (2,0), (1,2)] : [(inputs, weights)]
//!
//! //              And here are the inputs and weights on the right wires:
//! //                 - [(3,0), (0,1)] : [(inputs, weights)]
//!
//! //              To evaluate the wires we multiply their inputs by their weight:
//! //              and then the plus operation adds the results together:
//! //                 - Left plus operation equals => ((1 * 1) + (2 * 0) + (1 * 2))
//! //                 - Right plus operation equals => ((3 * 0) + (0 * 1))
//!
//! //              Finally the multiplication operation multiplies the result
//! //              of the previous two:
//! //                 - (((1 * 1) + (2 * 0) + (1 * 2)) * ((3 * 0) + (0 * 1)))
//! //              
//! //              Thus the output wire would have the value: 0
//! ```
//!
//! Note: A single wire may connect to any number of sub circuits including to the same
//! sub circuit multiple times on both its left and right inputs. (Exception,
//! wires must never form a loop anywhere in the `Circuit`)
//!
//! # `Circuit`
//!
//! A `Circuit` is made up of many connecting sub circuits. To evaluate a
//! `Circuit` means to determine the value of the `Circuit`'s output wires.
//! Which are in turn made up of the sub circuits output wires that do not
//! connect to another sub circuit's input wires. This also means a `Circuit`
//! has some number of input wires which come from the sub circuits with input
//! wires that do not connect to other sub circuit's output wires. `Circuit`s
//! are pure in the sense that the input uniquely determines the output of the
//! `Circuit`.
//!

use super::super::field::*;
use std::collections::HashMap;
use std::str::FromStr;
use std::marker::PhantomData;
use std::collections::hash_set::HashSet;

mod ast;
pub mod builder;
pub mod dummy_rep;

use self::ast::TokenList;
use self::ast::{Expression, ParseErr};
use self::builder::{ConnectionType, SubCircuitId};
use self::dummy_rep::DummyRep;

// This exports flatten_word8 because builder is a private module, so
// there is no other way to export it for lib.rs without making
// builder public.
use self::builder::{Circuit, WireId, Word8};

pub trait SetCircuitInputs<T> 
where
    T: Copy + Field,
{
    type Input;
    /// Sets the Struct (self) that contains `WireId`s with the values
    /// in `set` using the `circuit`
    fn set_inputs(&self, circuit: &mut Circuit<T>, set: &Self::Input);
}

// This is a default instance to keep other basic uses of
// `CircuitInstance` happy. However, in general you should not use
// this for any complex circuits
impl<'a, T> SetCircuitInputs<T> for Vec<&'a WireId> 
where
    T: Copy + Field,
{
    type Input = Vec<T>;

    fn set_inputs(&self, circuit: &mut Circuit<T>, set: &Self::Input) {
        assert_eq!(self.len(), set.len());
        set.iter().zip(self.into_iter()).for_each(|(val, wir)| circuit.set_value(**wir, *val));
    }
}

/// See `circuit_weights_type_check` for example of how to use this
/// without macros.
///
/// TODO: add proper docs here after you write the macro to create the
/// instances
pub struct CircuitInstance<T, F, V, W, I>
where
    T: Copy,
{
    circuit: Circuit<T>,
    verification_wires_len: usize,
    input_wires: W,
    ordered_wires: Vec<WireId>,
    sub_circuit_point: F,

    /// This is here to "constrain" the `I` and `V` generic parameters
    /// to make the compiler happy. There is no run-time overhead for
    /// this.
    ///
    /// The purpose behind `I` is to have `weights` take in some type
    /// `I` as input that maps directly to `W` (the input_wires) using
    /// the `SetCircuitInputs` trait.
    ///
    /// In other words `I` could be `(79u64, 20u8)` and `W` would be
    /// `(Word64, Word8)` then `set_inputs` would set the input
    /// `(Word64, Word8)` with `I`.
    phantom_i: PhantomData<I>,

    /// The purpose behind `V` is essentially the same, its some
    /// structure that we can iterate over its `WireId`.
    phantom_v: PhantomData<V>,
}

impl<'a, T, F, V, W, I> CircuitInstance<T, F, V, W, I>
where
    T: Copy + Field,
    F: Fn(SubCircuitId) -> T,
    V: IntoIterator<Item = &'a WireId>,
    W: IntoIterator<Item = &'a WireId> + SetCircuitInputs<T, Input = I>,
{
    pub fn new(
        circuit: Circuit<T>,
        verification_wires: V,
        input_wires: W,
        sub_circuit_point: F,
    ) -> Self {
        // The goal for ordered_wires is to end up with a structure
        // like this (where n = k and circuit.num_wires() = n):
        //
        // `{ unity_wire, verify0, verify1, ..., verifyN
        //  , witness0, witness1, ..., withnessK }`
        //
        // (Order for verify and witness does not mater, but the
        // verify needs to be before witness wires)
        // 
        // First we add the unity_wire:
        let mut ordered_wires = Vec::with_capacity(circuit.num_wires());
        ordered_wires.push(circuit.unity_wire());

        // Since we only can get`WireId`s from verification_wires `V`
        // and we need to check if a given `WireId` is one of the
        // verification_wires we turn it into a HashSet.
        let verification_wire_set: HashSet<&'a WireId> = verification_wires.into_iter().collect();

        let (mut verification_ids, witness_ids) = circuit
            .wire_assignments() // map will all assigned WireId
            .keys() // All the assigned WireId
            .filter(|w| **w != circuit.unity_wire()) // remove all the unity_wires
            .partition::<Vec<_>, _>(|k| verification_wire_set.contains(k));

        // Sort verification wires first
        verification_ids.sort_unstable();

        // Assign the wires that are to be verified to the lower indices
        ordered_wires.append(
            &mut verification_ids
                .into_iter()
                .chain(witness_ids.into_iter())
                .collect::<Vec<_>>(),
        );

        CircuitInstance {
            circuit,
            verification_wires_len: verification_wire_set.len(),
            input_wires,
            ordered_wires,
            sub_circuit_point,
            phantom_i: PhantomData,
            phantom_v: PhantomData,
        }
    }

    pub fn weights(&mut self, inputs: &I) -> Vec<T> {
        let CircuitInstance {
            ordered_wires,
            circuit,
            input_wires,
            ..
        } = self;

        // Set the values of the input wires of the circuit
        input_wires.set_inputs(circuit, inputs);

        ordered_wires
            .iter()
            .map(|w| circuit.evaluate(*w))
            .collect::<Vec<_>>()
    }
}

impl<'a, T, F, V, W, I> From<&'a CircuitInstance<T, F, V, W, I>> for DummyRep<T>
where
    T: Field + Copy,
    F: Fn(SubCircuitId) -> T,
{
    fn from(instance: &CircuitInstance<T, F, V, W, I>) -> Self {
        use self::ConnectionType::*;

        let mut u = Vec::with_capacity(instance.circuit.num_wires());
        let mut v = Vec::with_capacity(instance.circuit.num_wires());
        let mut w = Vec::with_capacity(instance.circuit.num_wires());
        let roots = instance
            .circuit
            .sub_circuits()
            .map(&instance.sub_circuit_point)
            .collect::<Vec<_>>();

        for wire in instance.ordered_wires.iter() {
            let (mut ui, mut vi, mut wi) = (Vec::new(), Vec::new(), Vec::new());

            for connection in instance.circuit.assignments(wire) {
                match connection {
                    Left(weight, sc_id) => ui.push(((instance.sub_circuit_point)(*sc_id), *weight)),
                    Right(weight, sc_id) => {
                        vi.push(((instance.sub_circuit_point)(*sc_id), *weight))
                    }
                    Output(sc_id) => wi.push(((instance.sub_circuit_point)(*sc_id), T::one())),
                }
            }

            u.push(ui);
            v.push(vi);
            w.push(wi);
        }

        DummyRep {
            u,
            v,
            w,
            roots,
            input: instance.verification_wires_len,
        }
    }
}

pub trait RootRepresentation<F>
where
    F: Field,
{
    type Row: Iterator<Item = Self::Column>;
    type Column: Iterator<Item = (F, F)>;
    type Roots: Iterator<Item = F>;

    fn u(&self) -> Self::Row;
    fn v(&self) -> Self::Row;
    fn w(&self) -> Self::Row;
    fn roots(&self) -> Self::Roots;
    fn input(&self) -> usize;
}

pub trait TryParse<T, F, E>
where
    T: RootRepresentation<F>,
    F: Field,
{
    fn try_parse(&str) -> Result<T, E>;
}

pub struct ASTParser {}

impl<F> TryParse<DummyRep<F>, F, ParseErr> for ASTParser
where
    F: Field + Clone + FromStr + From<usize>,
{
    fn try_parse(code: &str) -> Result<DummyRep<F>, ParseErr> {
        use self::Expression::*;
        use self::ParseErr::*;

        let expressions = ast::expressions(code)?;

        let mut variables: HashMap<String, usize> = HashMap::new();
        let mut gate_number = 0;
        let mut u: Vec<Vec<(F, F)>> = vec![Vec::new()];
        let mut v: Vec<Vec<(F, F)>> = vec![Vec::new()];
        let mut w: Vec<Vec<(F, F)>> = vec![Vec::new()];
        let mut input: usize = 0;

        // Only accept the following format (empty lines don't matter):
        //
        // (in ...)
        // (out ...)
        // (verify ...)
        //
        // (program ...)

        if expressions.len() != 4 {
            return Err(StructureErr(
                Some(gate_number),
                "Expected exactly one each of 'in', 'out', 'verify' and 'program'".to_string(),
            ));
        }

        let mut exp_iter = expressions.clone().into_iter();

        match exp_iter.next() {
            Some(In(_)) => (),
            _ => {
                return Err(StructureErr(
                    Some(gate_number),
                    "Expected first expression to be 'in'".to_string(),
                ))
            }
        }
        match exp_iter.next() {
            Some(Out(_)) => (),
            _ => {
                return Err(StructureErr(
                    Some(gate_number),
                    "Expected second expression to be 'out'".to_string(),
                ))
            }
        }
        if let Some(Verify(vars)) = exp_iter.next() {
            for var in vars.into_iter() {
                match var {
                    Var(vr) => {
                        let index = u.len();
                        variables.insert(vr, index);

                        u.push(Vec::new());
                        v.push(Vec::new());
                        w.push(Vec::new());
                        input += 1;
                    }
                    _ => panic!("parse_expression() did not correctly parse 'verify'"),
                }
            }
        } else {
            return Err(StructureErr(
                Some(gate_number),
                "Expected third expression to be 'verify'".to_string(),
            ));
        }
        if let Some(Program(program)) = exp_iter.next() {
            for assignment in program.into_iter() {
                gate_number += 1;

                if let Assign(left, right) = assignment {
                    if let Var(vr) = *left {
                        // If this is the first appearance of the variable, add it to the list
                        if !variables.contains_key(&vr) {
                            let index = u.len();
                            variables.insert(vr, index);

                            u.push(Vec::new());
                            v.push(Vec::new());
                            w.push(vec![(gate_number.into(), 1.into())]);
                        } else {
                            // We can unwrap because we just checked that the key exists
                            if *variables.get(&vr).unwrap() <= input {
                                let index = variables.get(&vr).unwrap();
                                if w[*index].len() != 0 {
                                    return Err(StructureErr(
                                        Some(gate_number),
                                        "Varify variable cannot be the output of two different gates"
                                            .to_string(),
                                    ));
                                }
                                w[*index].push((gate_number.into(), 1.into()));
                            } else {
                                return Err(StructureErr(
                                    Some(gate_number),
                                    "Already declared variable cannot be the output wire of a gate"
                                        .to_string(),
                                ));
                            }
                        }
                    } else {
                        panic!("parse_expression() did not correctly parse '='");
                    }

                    let right = *right;
                    if let Mul(left, right) = right {
                        // Handle the left inputs
                        match *left {
                            Literal(lit) => u[0].push((gate_number.into(), lit)),
                            Var(vr) => {
                                if !variables.contains_key(&vr) {
                                    let index = u.len();
                                    variables.insert(vr, index);

                                    u.push(vec![(gate_number.into(), 1.into())]);
                                    v.push(Vec::new());
                                    w.push(Vec::new());
                                } else {
                                    // We can unwrap because we just checked that the key exists
                                    let index = variables.get(&vr).unwrap();
                                    u[*index].push((gate_number.into(), 1.into()));
                                }
                            }
                            Add(a) => {
                                for exp in a.into_iter() {
                                    match exp {
                                        Literal(lit) => u[0].push((gate_number.into(), lit)),
                                        Var(vr) => {
                                            if !variables.contains_key(&vr) {
                                                let index = u.len();
                                                variables.insert(vr, index);

                                                u.push(vec![(gate_number.into(), 1.into())]);
                                                v.push(Vec::new());
                                                w.push(Vec::new());
                                            } else {
                                                // We can unwrap because we just checked that the key exists
                                                let index = variables.get(&vr).unwrap();
                                                u[*index].push((gate_number.into(), 1.into()));
                                            }
                                        }
                                        Mul(left, right) => {
                                            let left = match *left {
                                                Literal(lit) => lit,
                                                _ => return Err(StructureErr(
                                                    Some(gate_number),
                                                    "LHS of a '*' expression in a '+' expression must be a literal".to_string()
                                                )),
                                            };
                                            let right = match *right {
                                                Var(vr) => vr,
                                                _ => return Err(StructureErr(
                                                    Some(gate_number),
                                                    "RHS of a '*' expression in a '+' expression must be a variable".to_string()
                                                )),
                                            };

                                            if !variables.contains_key(&right) {
                                                let index = u.len();
                                                variables.insert(right, index);

                                                u.push(vec![(gate_number.into(), left)]);
                                                v.push(Vec::new());
                                                w.push(Vec::new());
                                            } else {
                                                // We can unwrap because we just checked that the key exists
                                                let index = variables.get(&right).unwrap();
                                                u[*index].push((gate_number.into(), left));
                                            }
                                        }
                                        _ => {
                                            return Err(StructureErr(
                                                Some(gate_number),
                                                "Invalid expression found in '+' expression"
                                                    .to_string(),
                                            ))
                                        }
                                    }
                                }
                            }
                            _ => {
                                return Err(StructureErr(
                                    Some(gate_number),
                                    "Invalid expression found in '*' expression".to_string(),
                                ))
                            }
                        }

                        // Handle the right inputs
                        match *right {
                            Literal(lit) => v[0].push((gate_number.into(), lit)),
                            Var(vr) => {
                                if !variables.contains_key(&vr) {
                                    let index = v.len();
                                    variables.insert(vr, index);

                                    u.push(Vec::new());
                                    v.push(vec![(gate_number.into(), 1.into())]);
                                    w.push(Vec::new());
                                } else {
                                    // We can unwrap because we just checked that the key exists
                                    let index = variables.get(&vr).unwrap();
                                    v[*index].push((gate_number.into(), 1.into()));
                                }
                            }
                            Add(a) => {
                                for exp in a.into_iter() {
                                    match exp {
                                        Literal(lit) => v[0].push((gate_number.into(), lit)),
                                        Var(vr) => {
                                            if !variables.contains_key(&vr) {
                                                let index = v.len();
                                                variables.insert(vr, index);

                                                u.push(Vec::new());
                                                v.push(vec![(gate_number.into(), 1.into())]);
                                                w.push(Vec::new());
                                            } else {
                                                // We can unwrap because we just checked that the key exists
                                                let index = variables.get(&vr).unwrap();
                                                v[*index].push((gate_number.into(), 1.into()));
                                            }
                                        }
                                        Mul(left, right) => {
                                            let left = match *left {
                                                Literal(lit) => lit,
                                                _ => return Err(StructureErr(
                                                    Some(gate_number),
                                                    "LHS of a '*' expression in a '+' expression must be a literal".to_string()
                                                )),
                                            };
                                            let right = match *right {
                                                Var(vr) => vr,
                                                _ => return Err(StructureErr(
                                                    Some(gate_number),
                                                    "RHS of a '*' expression in a '+' expression must be a variable".to_string()
                                                )),
                                            };

                                            if !variables.contains_key(&right) {
                                                let index = v.len();
                                                variables.insert(right, index);

                                                u.push(Vec::new());
                                                v.push(vec![(gate_number.into(), left)]);
                                                w.push(Vec::new());
                                            } else {
                                                // We can unwrap because we just checked that the key exists
                                                let index = variables.get(&right).unwrap();
                                                v[*index].push((gate_number.into(), left));
                                            }
                                        }
                                        _ => {
                                            return Err(StructureErr(
                                                Some(gate_number),
                                                "Invalid expression found in '+' expression"
                                                    .to_string(),
                                            ))
                                        }
                                    }
                                }
                            }
                            _ => {
                                return Err(StructureErr(
                                    Some(gate_number),
                                    "Invalid expression found in '*' expression".to_string(),
                                ))
                            }
                        }
                    }
                } else {
                    return Err(StructureErr(
                        Some(gate_number),
                        "Program expression must be a list of '=' expressions".to_string(),
                    ));
                }
            }
        } else {
            return Err(StructureErr(
                Some(gate_number),
                "Expected fourth expression to be 'program'".to_string(),
            ));
        }

        let roots = (1..gate_number + 1).map(|r| r.into()).collect::<Vec<_>>();

        Ok(DummyRep {
            u,
            v,
            w,
            roots,
            input,
        })
    }
}

pub fn weights<F>(code: &str, values: &[F]) -> Result<Vec<F>, ParseErr>
where
    F: Clone + Field + FromStr + PartialEq,
{
    use self::Expression::*;
    use self::ParseErr::*;

    let mut assignments: HashMap<String, F> = HashMap::new();
    let expressions = ast::expressions(code)?;
    let mut exp_iter = expressions.as_slice().iter();
    let token_list: TokenList<F> = ast::try_to_list(code.to_string())?;
    let variables = ast::variable_order(token_list);

    let inputs = match exp_iter.next() {
        Some(In(i)) => i,
        _ => {
            return Err(StructureErr(
                None,
                "Expected first expression to be 'in'".to_string(),
            ))
        }
    };

    if inputs.len() != values.len() {
        return Err(StructureErr(
            None,
            "Wrong number of values supplied".to_string(),
        ));
    }

    inputs.as_slice().iter().zip(values).for_each(|(e, val)| {
        if let Var(var) = e {
            assignments.insert(var.clone(), val.clone());
        } else {
            panic!("Under constained or malformed inputs".to_string());
        }
    });

    match exp_iter.next() {
        Some(Out(_)) => (),
        _ => {
            return Err(StructureErr(
                None,
                "Expected second expression to be 'out'".to_string(),
            ))
        }
    }

    if let Some(Verify(vars)) = exp_iter.next() {
        for var in vars.into_iter() {
            match var {
                Var(_) => (),
                _ => panic!("parse_expression() did not correctly parse 'verify'"),
            }
        }
    } else {
        return Err(StructureErr(
            None,
            "Expected third expression to be 'verify'".to_string(),
        ));
    }

    if let Some(Program(program)) = exp_iter.next() {
        for assignment in program.into_iter() {
            if let Assign(left, right) = assignment {
                if let Var(ref var) = **left {
                    if assignments.contains_key(var) {
                        return Err(StructureErr(
                            None,
                            "Attempted to assign to an already assigned variable".to_string(),
                        ));
                    }

                    match evaluate(right, &assignments) {
                        Some(value) => assignments.insert(var.clone(), value),
                        None => {
                            return Err(StructureErr(
                                None,
                                "Under constrained expression".to_string(),
                            ))
                        }
                    };
                } else {
                    panic!("parse_expression() did not correctly parse '='");
                }
            } else {
                return Err(StructureErr(
                    None,
                    "Program expression must be a list of '=' expressions".to_string(),
                ));
            }
        }
    } else {
        return Err(StructureErr(
            None,
            "Expected fourth expression to be 'program'".to_string(),
        ));
    }

    let weights = variables.into_iter().map(|v| {
        assignments
            .remove(&v)
            .expect("Every variable should have an assignment")
    });

    Ok(::std::iter::once(F::one())
        .chain(weights)
        .collect::<Vec<_>>())
}

fn evaluate<F>(expression: &Expression<F>, assignments: &HashMap<String, F>) -> Option<F>
where
    F: Clone + Field,
{
    use self::Expression::{Add, Literal, Mul, Var};

    match *expression {
        Literal(ref lit) => Some(lit.clone()),
        Var(ref var) => assignments.get(var).cloned(),
        Mul(ref left, ref right) => {
            evaluate(left, assignments).and_then(|l| evaluate(right, assignments).map(|r| l * r))
        }
        Add(ref inputs) => inputs.into_iter().try_fold(F::zero(), |acc, x| {
            evaluate(&x, assignments).map(|v| acc + v)
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::field::z251::Z251;
    use super::dummy_rep::DummyRep;
    use super::*;

    #[test]
    fn try_parse_impl_test() {
        let code = "(in x a b c)
                    (out y)
                    (verify x y)

                    (program
                        (= t1
                            (* x a))
                        (= t2
                            (* x (+ t1 b)))
                        (= y
                            (* 1 (+ t2 c))))";

        // The order of appearance of the variables is (input vairables first):
        // x y t1 a t2 b c

        let expected = DummyRep::<Z251> {
            u: vec![
                vec![(3.into(), 1.into())],                       // 1
                vec![(1.into(), 1.into()), (2.into(), 1.into())], // x
                vec![],                                           // y
                vec![],                                           // t1
                vec![],                                           // a
                vec![],                                           // t2
                vec![],                                           // b
                vec![],                                           // c
            ],
            v: vec![
                vec![],                     // 1
                vec![],                     // x
                vec![],                     // y
                vec![(2.into(), 1.into())], // t1
                vec![(1.into(), 1.into())], // a
                vec![(3.into(), 1.into())], // t2
                vec![(2.into(), 1.into())], // b
                vec![(3.into(), 1.into())], // c
            ],
            w: vec![
                vec![],                     // 1
                vec![],                     // x
                vec![(3.into(), 1.into())], // y
                vec![(1.into(), 1.into())], // t1
                vec![],                     // a
                vec![(2.into(), 1.into())], // t2
                vec![],                     // b
                vec![],                     // c
            ],
            roots: vec![1.into(), 2.into(), 3.into()],
            input: 2,
        };
        let actual = ASTParser::try_parse(code).unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    fn evaluate_test() {
        use self::Expression::*;

        let mut assignments = HashMap::<_, Z251>::new();
        assignments.insert("a".to_string(), 3.into());
        assignments.insert("b".to_string(), 2.into());

        let temp: Expression<Z251> = Mul(
            Box::new(Var("a".to_string())),
            Box::new(Var("b".to_string())),
        );
        let scale_temp = Mul(Box::new(Literal(4.into())), Box::new(temp));
        let six = Mul(Box::new(Literal(6.into())), Box::new(Literal(1.into())));
        let sum = Add(vec![scale_temp, Var("c".to_string()), six]);
        let expression = Mul(Box::new(Literal(1.into())), Box::new(sum));

        // Not all inputs assigned
        assert_eq!(evaluate(&expression, &assignments), None);

        // All inputs assigned
        assignments.insert("c".to_string(), 4.into());
        assert_eq!(evaluate(&expression, &assignments), Some(34.into()));
    }

    #[test]
    fn weights_test() {
        let code = "(in a b c)
                    (out x)
                    (verify b x)

                    (program
                        (= temp
                            (* a b))
                        (= x
                            (* 1 (+ (* 4 temp) c 6))))";

        let assignments = &[3.into(), 2.into(), 4.into()];

        let expected: Vec<Z251> = vec![
            1.into(),  // Unity input
            2.into(),  // b = 2
            34.into(), // x = 34
            6.into(),  // temp = ab = 6
            3.into(),  // a = 3
            4.into(),  // c = 4
        ];

        assert_eq!(Ok(expected), weights(&code, assignments));
    }

    // This is a test to show you can now create `CircuitInstance`
    // with some new struct that has `SetCircuitInputs` implemented
    // for it and you are not able to give `weights` the wrong input
    // type. (Which in this case is show by giving it the right type)
    #[test]
    fn circuit_weights_type_check() {
        let mut circuit = Circuit::<Z251>::new();
        let left = circuit.new_word8(); 
        let right = circuit.new_word8(); 
        let cmp: WireId = circuit.greater_than(&left, &right);

        let verify_wires: Vec<WireId> = circuit.bit_check([left, right].iter().flat_map(|x| x.iter()));

        use std::iter::Chain;
        use std::slice::Iter;

        struct CMP<'a> {
            left: &'a Word8,
            right: &'a Word8,
        }


        impl<'a> IntoIterator for CMP<'a> {
            type Item = &'a WireId;
            type IntoIter = Chain<Iter<'a, WireId>, Iter<'a, WireId>>;

            fn into_iter(self) -> Self::IntoIter {
                self.left.into_iter().chain(self.left.into_iter())
            }
        }

        impl<'a> SetCircuitInputs<Z251> for CMP<'a> {
            type Input = (u8, u8);

            fn set_inputs(&self, circuit: &mut Circuit<Z251>, set: &Self::Input) {
                circuit.set_from_num(self.left, set.0);
                circuit.set_from_num(self.right, set.1);
            }
        }

        let input: CMP = CMP {
            left: &left,
            right: &right,
        };

        let mut instance =
            CircuitInstance::new(circuit, &verify_wires, input, |w| {
                Z251::from(w.inner_id() + 1)
            });

        let _weights: Vec<Z251> = instance.weights(&(26, 11));

        assert_eq!(instance.circuit.evaluate(cmp), Z251::from(1));
    }
}
