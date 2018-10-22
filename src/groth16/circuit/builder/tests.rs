use super::super::super::Z251;
use super::*;

#[test]
fn bit_checker_test() {
    // Bit checker with input 0
    let mut circuit = Circuit::<Z251>::new();
    let input = circuit.new_wire();
    let checker = circuit.new_bit_checker(input);

    circuit.set_value(input, Z251::from(0));
    assert!(circuit.evaluate(checker) == Z251::from(0));

    // Bit checker with input 1
    let mut circuit = Circuit::<Z251>::new();
    let input = circuit.new_wire();
    let checker = circuit.new_bit_checker(input);

    circuit.set_value(input, Z251::from(1));
    assert!(circuit.evaluate(checker) == Z251::from(0));

    // Bit checker with random non-binary input
    for i in 2..251 {
        let mut circuit = Circuit::<Z251>::new();
        let input = circuit.new_wire();
        let checker = circuit.new_bit_checker(input);

        circuit.set_value(input, Z251::from(i));
        assert!(circuit.evaluate(checker) != Z251::from(0));
    }
}

#[test]
fn and_test() {
    let logic_table = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1),
    ];

    for (l, r, l_and_r) in logic_table.iter() {
        let mut circuit = Circuit::<Z251>::new();
        let l_wire = circuit.new_wire();
        let r_wire = circuit.new_wire();
        let and = circuit.new_and(l_wire, r_wire);

        circuit.set_value(l_wire, Z251::from(*l));
        circuit.set_value(r_wire, Z251::from(*r));
        assert!(circuit.evaluate(and) == Z251::from(*l_and_r));
    }
}

#[test]
fn or_test() {
    let logic_table = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ];

    for (l, r, l_or_r) in logic_table.iter() {
        let mut circuit = Circuit::<Z251>::new();
        let l_wire = circuit.new_wire();
        let r_wire = circuit.new_wire();
        let or = circuit.new_or(l_wire, r_wire);

        circuit.set_value(l_wire, Z251::from(*l));
        circuit.set_value(r_wire, Z251::from(*r));
        assert!(circuit.evaluate(or) == Z251::from(*l_or_r));
    }
}