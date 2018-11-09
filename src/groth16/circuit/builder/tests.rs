use super::super::super::Z251;
use super::*;
use std::ops::{BitAnd, BitOr, BitXor};

extern crate quickcheck;
use self::quickcheck::quickcheck;

fn keccakf(state: &mut [[u64; 5]; 5]) {
    for i in 0..24 {
        round(state, ROUND_CONSTANTS[i]);
    }
}

fn round(state: &mut [[u64; 5]; 5], rc: u64) {
    theta(state);
    rho_and_phi(state);
    iota(state, rc);
}

fn theta(state: &mut [[u64; 5]; 5]) {
    let c = [
        state[0][0] | state[0][1] | state[0][2] | state[0][3] | state[0][4],
        state[1][0] | state[1][1] | state[1][2] | state[1][3] | state[1][4],
        state[2][0] | state[2][1] | state[2][2] | state[2][3] | state[2][4],
        state[3][0] | state[3][1] | state[3][2] | state[3][3] | state[3][4],
        state[4][0] | state[4][1] | state[4][2] | state[4][3] | state[4][4],
    ];
    let d = [
        c[4] | c[1].rotate_left(1),
        c[0] | c[2].rotate_left(1),
        c[1] | c[3].rotate_left(1),
        c[2] | c[4].rotate_left(1),
        c[3] | c[0].rotate_left(1),
    ];

    for (row, d) in state.iter_mut().zip(d.iter()) {
        for col in row.iter_mut() {
            *col ^= d
        }
    }
}

fn rho_and_phi(state: &mut [[u64; 5]; 5]) {
    let mut b = [[0; 5]; 5];
    for x in 0..5 {
        for y in 0..5 {
            let bx = y;
            let by = (2*x + 3*y) % 5;
            b[bx][by] = state[x][y].rotate_left(ROTATION_OFFSETS[x][y]);
        }
    }
    *state = b;
}

fn iota(state: &mut [[u64; 5]; 5], rc: u64) {
    state[0][0] ^= rc;
}

#[test]
fn bit_checker_test() {
    let mut circuit = Circuit::<Z251>::new();
    let input = circuit.new_wire();
    let checker = circuit.new_bit_checker(input);

    // Bit checker with input 0
    circuit.set_value(input, Z251::from(0));
    assert!(circuit.evaluate(checker) == Z251::from(0));

    // Bit checker with input 1
    circuit.reset();
    circuit.set_value(input, Z251::from(1));
    assert!(circuit.evaluate(checker) == Z251::from(0));

    // Bit checker with random non-binary input
    for i in 2..251 {
        circuit.reset();
        circuit.set_value(input, Z251::from(i));
        assert!(circuit.evaluate(checker) != Z251::from(0));
    }
}

#[test]
fn and_test() {
    let logic_table = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)];
    let mut circuit = Circuit::<Z251>::new();
    let l_wire = circuit.new_wire();
    let r_wire = circuit.new_wire();
    let and = circuit.new_and(l_wire, r_wire);

    for (l, r, l_and_r) in logic_table.iter() {
        circuit.reset();
        circuit.set_value(l_wire, Z251::from(*l));
        circuit.set_value(r_wire, Z251::from(*r));
        assert!(circuit.evaluate(and) == Z251::from(*l_and_r));
    }
}

#[test]
fn not_test() {
    let logic_table = [(0, 1), (1, 0)];
    let mut circuit = Circuit::<Z251>::new();
    let wire = circuit.new_wire();
    let not = circuit.new_not(wire);

    for (l, an) in logic_table.iter() {
        circuit.reset();
        circuit.set_value(wire, Z251::from(*l));
        assert!(circuit.evaluate(not) == Z251::from(*an));
    }
}

#[test]
fn or_test() {
    let logic_table = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)];
    let mut circuit = Circuit::<Z251>::new();
    let l_wire = circuit.new_wire();
    let r_wire = circuit.new_wire();
    let or = circuit.new_or(l_wire, r_wire);

    for (l, r, l_or_r) in logic_table.iter() {
        circuit.reset();
        circuit.set_value(l_wire, Z251::from(*l));
        circuit.set_value(r_wire, Z251::from(*r));
        assert!(circuit.evaluate(or) == Z251::from(*l_or_r));
    }
}

#[test]
fn xor_test() {
    let logic_table = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)];
    let mut circuit = Circuit::<Z251>::new();
    let l_wire = circuit.new_wire();
    let r_wire = circuit.new_wire();
    let xor = circuit.new_xor(l_wire, r_wire);

    for (l, r, l_xor_r) in logic_table.iter() {
        circuit.reset();
        circuit.set_value(l_wire, Z251::from(*l));
        circuit.set_value(r_wire, Z251::from(*r));
        assert!(circuit.evaluate(xor) == Z251::from(*l_xor_r));
    }
}

#[test]
fn fan_in_and_test() {
    let mut circuit = Circuit::<Z251>::new();
    let mut wires = [WireId(0); 8];
    for j in 0..8 {
        wires[j] = circuit.new_wire();
    }

    for i in 0..256 {
        circuit.reset();
        for j in 0..8 {
            circuit.set_value(wires[j], Z251::from((i >> j) % 2));
        }

        let output = circuit.fan_in(&wires, Circuit::new_and);
        if i != 255 {
            assert!(circuit.evaluate(output) == Z251::from(0));
        } else {
            assert!(circuit.evaluate(output) == Z251::from(1));
        }
    }
}

#[test]
fn fan_in_or_test() {
    let mut circuit = Circuit::<Z251>::new();
    let mut wires = [WireId(0); 8];
    for j in 0..8 {
        wires[j] = circuit.new_wire();
    }

    for i in 0..256 {
        circuit.reset();
        for j in 0..8 {
            circuit.set_value(wires[j], Z251::from((i >> j) % 2));
        }

        let output = circuit.fan_in(&wires, Circuit::new_or);
        if i != 0 {
            assert!(circuit.evaluate(output) == Z251::from(1));
        } else {
            assert!(circuit.evaluate(output) == Z251::from(0));
        }
    }
}

#[test]
fn fan_in_xor_test() {
    let mut circuit = Circuit::<Z251>::new();
    let mut wires = [WireId(0); 8];
    for j in 0..8 {
        wires[j] = circuit.new_wire();
    }

    for i in 0..256 {
        circuit.reset();
        for j in 0..8 {
            circuit.set_value(wires[j], Z251::from((i >> j) % 2));
        }

        let output = circuit.fan_in(&wires, Circuit::new_xor);
        if i.count_ones() % 2 == 0 {
            assert!(circuit.evaluate(output) == Z251::from(0));
        } else {
            assert!(circuit.evaluate(output) == Z251::from(1));
        }
    }
}

#[test]
fn bitwise_op_test() {
    let mut circuit = Circuit::<Z251>::new();
    let (l_wires, r_wires): (Vec<_>, Vec<_>) = (0..4)
        .map(|_| (circuit.new_wire(), circuit.new_wire()))
        .unzip();

    // let tmp = [|l: usize, r: usize| l ^ r, |l: usize, r: usize| l & r, |l, r| l | r];
    // let tmp2 = [Circuit::<Z251>::new_xor, Circuit::new_and, Circuit::new_or];

    let out_wires = circuit.bitwise_op(&l_wires, &r_wires, Circuit::new_xor);

    (0..256).map(|n| (n >> 8, n % 16)).for_each(|(l, r)| {
        circuit.reset();
        for j in 0..4 {
            circuit.set_value(l_wires[j], Z251::from((l >> j) % 2));
            circuit.set_value(r_wires[j], Z251::from((r >> j) % 2));
        }
        assert_eq!(
            out_wires
                .iter()
                .map(|&x| circuit.evaluate(x))
                .map(|x| x.into())
                .rev()
                .fold(0, |acc, x: usize| (acc << 1) + x),
            l ^ r
        );
    });
}

quickcheck! {
    // fn step0_prop(inputs: [u64; 25]) -> bool {
    //     let mut word64: Word64 = (0..64).map(WireId).collect();
    //     word64.rotate_right_mut(rotate_by);
    //     word64.rotate_left_mut(rotate_by);
    //     word64 == (0..64).map(WireId).collect()
    // }
}
