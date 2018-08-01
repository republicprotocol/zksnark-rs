use self::ast::{Expression, ParseErr};
use self::dummy_rep::DummyRep;
use super::super::field::*;
use std::collections::HashMap;
use std::ops::Mul;
use std::str::FromStr;

mod arithmetic_circuit;
mod ast;
pub mod dummy_rep;

use self::arithmetic_circuit::{AddGate, Circuit, CircuitGate, MulGate, ScalarGate};
use self::ast::TokenList;

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

fn weights<'a, F>(
    code: &str,
    mut assignments: HashMap<String, F>,
) -> Result<Vec<F>, ParseErr>
where
    F: Clone + Field + FromStr + PartialEq,
{
    use self::Expression::*;
    use self::ParseErr::*;

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

    {
        let constrained = inputs.as_slice().iter().all(|e| {
            if let Var(var) = e {
                assignments.contains_key(var)
            } else {
                false
            }
        });

        if !constrained {
            return Err(StructureErr(
                None,
                "Under constained or malformed inputs".to_string(),
            ));
        }
    }

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
                Var(var) => (),
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

    Ok(::std::iter::once(F::mul_identity()).chain(weights).collect::<Vec<_>>())
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
        Add(ref inputs) => inputs.into_iter().try_fold(F::add_identity(), |acc, x| {
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
    use super::ast;

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

        let mut assignments = HashMap::<_, Z251>::new();
        assignments.insert("a".to_string(), 3.into());
        assignments.insert("b".to_string(), 2.into());
        assignments.insert("c".to_string(), 4.into());

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
}
