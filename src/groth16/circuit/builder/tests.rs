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
