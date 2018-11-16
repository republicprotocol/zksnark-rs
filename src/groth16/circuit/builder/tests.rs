use super::super::super::Z251;
use super::*;
use field::FieldIdentity;
use std::iter::repeat;
use std::ops::{BitAnd, BitOr, BitXor};

extern crate quickcheck;
use self::quickcheck::quickcheck;

extern crate tiny_keccak;
use self::tiny_keccak::keccakf;

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

// /// I guessed the [0][0][0] value was zero
// #[test]
// fn keccak_f_basic() {
//     let mut circuit = Circuit::<Z251>::new();

//     let data: [[u64; 5]; 5] = [
//         [0, 1, 2, 3, 4],
//         [5, 6, 7, 8, 9],
//         [10, 11, 12, 13, 14],
//         [15, 16, 17, 18, 19],
//         [20, 21, 22, 23, 24],
//     ];
//     let matrix: KeccakMatrix = circuit.new_keccakmatrix();
//     circuit.set_keccakmatrix(&matrix, data);
//     let output: KeccakMatrix = circuit.keccak_f1600(matrix);
//     assert_eq!(circuit.evaluate_keccakmatrix(output)[0][0][0], Z251::zero());
// }

// fn sha3_256_basic() {
//     let mut circuit = Circuit::<Z251>::new();

//     let input: Vec<Word64> = vec![
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//         circuit.new_word64(),
//     ];
//     input
//         .iter()
//         .enumerate()
//         .for_each(|(i, word)| circuit.set_word64(word, (i as u64)));
//     let output = circuit.sha3_256(input);
//     assert_eq!(circuit.evaluate_word64(output[0])[0], Z251::zero());
//     // FIXME: after you write a bit_stream evaluate function
// }

#[test]
fn word64_set_eval() {
    let mut circuit = Circuit::<Z251>::new();
    let u64_input = circuit.new_word64();
    circuit.set_word64(&u64_input, 1);
    assert_eq!(circuit.evaluate_word64(&u64_input), 1);
}

#[test]
fn set_word8_is_bit_little_endian() {
    let mut circuit = Circuit::<Z251>::new();
    let u8_input = circuit.new_word8();
    circuit.set_word8(&u8_input, 0b0000_0100);
    assert_eq!(circuit.evaluate(u8_input[0]), Z251::zero());
    assert_eq!(circuit.evaluate(u8_input[1]), Z251::zero());
    assert_eq!(circuit.evaluate(u8_input[2]), Z251::one());
    assert_eq!(circuit.evaluate(u8_input[3]), Z251::zero());

    assert_eq!(circuit.evaluate(u8_input[4]), Z251::zero());
    assert_eq!(circuit.evaluate(u8_input[5]), Z251::zero());
    assert_eq!(circuit.evaluate(u8_input[6]), Z251::zero());
    assert_eq!(circuit.evaluate(u8_input[7]), Z251::zero());
}

#[test]
fn set_word64_is_little_endian() {
    let mut circuit = Circuit::<Z251>::new();
    let o = circuit.new_word8();
    let k = circuit.new_word8();
    circuit.set_word8(&o, 0b0100_1111); // becomes 1111 0010
    circuit.set_word8(&k, 0b0100_1011); // becomes 1101 0010
    let w64: Word64 = [o, k].iter().collect();
    assert_eq!(circuit.evaluate(w64[0][0]), Z251::one());
    assert_eq!(circuit.evaluate(w64[0][1]), Z251::one());
    assert_eq!(circuit.evaluate(w64[0][2]), Z251::one());
    assert_eq!(circuit.evaluate(w64[0][3]), Z251::one());

    assert_eq!(circuit.evaluate(w64[0][4]), Z251::zero());
    assert_eq!(circuit.evaluate(w64[0][5]), Z251::zero());
    assert_eq!(circuit.evaluate(w64[0][6]), Z251::one());
    assert_eq!(circuit.evaluate(w64[0][7]), Z251::zero());

    assert_eq!(circuit.evaluate(w64[1][0]), Z251::one());
    assert_eq!(circuit.evaluate(w64[1][1]), Z251::one());
    assert_eq!(circuit.evaluate(w64[1][2]), Z251::zero());
    assert_eq!(circuit.evaluate(w64[1][3]), Z251::one());

    assert_eq!(circuit.evaluate(w64[1][4]), Z251::zero());
    assert_eq!(circuit.evaluate(w64[1][5]), Z251::zero());
    assert_eq!(circuit.evaluate(w64[1][6]), Z251::one());
    assert_eq!(circuit.evaluate(w64[1][7]), Z251::zero());

    iproduct!(2..8, 0..8).for_each(|(x, y)| assert_eq!(circuit.evaluate(w64[x][y]), Z251::zero()));
}

#[test]
fn iproduct_macro_single_test() {
    assert_eq!(
        iproduct!(0..5, 0..5).collect::<Vec<_>>(),
        vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4)
        ]
    );
}

fn theta_part_single_test() {
    let mut input: [u64; 25] = [0; 25];
    let mut tiny: &mut [u64; 25] = &mut [0; 25];
    vec![0, 9, 54, 21].iter().zip(0..25).for_each(|(&num, i)| {
        input[i] = num;
        tiny[i] = num;
    });

    let mut array: [u64; 5] = [0; 5];

    // Theta
    for x in 0..5 {
        for y_count in 0..5 {
            let y = y_count * 5;
            array[x] ^= tiny[x + y];
        }
    }

    let mut circuit = Circuit::<Z251>::new();
    let a = circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(&a, input);

    let mut c: KeccakRow = (0..5)
        .map(|x: isize| {
            circuit.u64_fan_in(
                [a[x][0], a[x][1], a[x][2], a[x][3], a[x][4]].iter(),
                Circuit::new_xor,
            )
        }).collect();

    // NOTE: its rotate_right because Word64 is stored as bit little endian.
    (0..5).for_each(|x: isize| {
        c[x] = circuit.u64_fan_in(
            [c[x - 1], c[x + 1].rotate_right(1)].iter(),
            Circuit::new_xor,
        )
    });

    assert_eq!(circuit.evaluate_keccakrow(&c), array);
}

#[test]
fn theta_single_test() {
    let mut input: [u64; 25] = [0; 25];
    let mut a: &mut [u64; 25] = &mut [0; 25];
    vec![0, 1].iter().zip(0..25).for_each(|(&num, i)| {
        input[i] = num;
        a[i] = num;
    });

    let mut array: [u64; 5] = [0; 5];

    // Theta
    for x in 0..5 {
        for y_count in 0..5 {
            let y = y_count * 5;
            array[x] ^= a[x + y];
        }
    }

    for x in 0..5 {
        for y_count in 0..5 {
            let y = y_count * 5;
            a[y + x] ^= array[(x + 4) % 5] ^ array[(x + 1) % 5].rotate_left(1);
        }
    }

    let mut circuit = Circuit::<Z251>::new();
    let mut matrix = circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(&matrix, input);
    circuit.theta(&mut matrix);

    assert_eq!(circuit.evaluate_keccakmatrix(&matrix), *a);
}

#[test]
fn keccak_f1600_single_test() {
    let mut input: [u64; 25] = [0; 25];
    let mut tiny_keccak: &mut [u64; 25] = &mut [0; 25];

    let mut circuit = Circuit::<Z251>::new();

    keccakf(tiny_keccak);

    let mut matrix = &mut circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(&matrix, input);

    circuit.keccak_f1600(matrix);
    assert_eq!(circuit.evaluate_keccakmatrix(matrix), *tiny_keccak);
}

fn u64_fan_in_single_test() {
    let mut input: [u64; 5] = [1, 0, 35, 5, 6];

    let mut circuit = Circuit::<Z251>::new();

    let row = circuit.new_keccakrow();
    circuit.set_keccakrow(&row, input);

    let complete_circuit = circuit.u64_fan_in(row.iter(), Circuit::new_xor);

    assert_eq!(
        circuit.evaluate_word64(&complete_circuit),
        1 ^ 0 ^ 35 ^ 5 ^ 6
    );
}

quickcheck! {
    fn u64_fan_in_prop(rand: Vec<u64>) -> bool {
        let mut input: [u64; 5] = [0; 5];
        rand.iter().zip(0..5).for_each(|(&num, i)| input[i] = num);

        let mut circuit = Circuit::<Z251>::new();

        let row = circuit.new_keccakrow();
        circuit.set_keccakrow(&row, input);

        let complete_circuit = circuit.u64_fan_in(row.iter(), Circuit::new_xor);

        circuit.evaluate_word64(&complete_circuit) == input.iter().skip(1).fold(input[0], |acc, x| acc ^ x)
    }
    fn keccak_f1600_equiv_prop(rand: Vec<u64>) -> bool {
        let mut input: [u64; 25] = [0; 25];
        let mut tiny_keccak: &mut [u64; 25] = &mut [0; 25];
        rand.iter().zip(0..25).for_each(|(&num, i)| {input[i] = num; tiny_keccak[i] = num;});

        let mut circuit = Circuit::<Z251>::new();

        keccakf(tiny_keccak);
        let a_copy = *tiny_keccak;

        let mut matrix = &mut circuit.new_keccakmatrix();
        circuit.set_keccakmatrix(&matrix, input);

        circuit.keccak_f1600(matrix);
        circuit.evaluate_keccakmatrix(matrix) == a_copy
    }
    fn theta_prop(rand: Vec<u64>) -> bool {
        let mut input: [u64; 25] = [0; 25];
        let mut a: &mut [u64; 25] = &mut [0; 25];
        rand.iter().zip(0..25).for_each(|(&num, i)| {input[i] = num; a[i] = num;});

        let mut array: [u64; 5] = [0; 5];

        // Theta
        for x in 0..5 {
            for y_count in 0..5 {
                let y = y_count * 5;
                array[x] ^= a[x + y];
            }
        }

        for x in 0..5 {
            for y_count in 0..5 {
                let y = y_count * 5;
                a[y + x] ^= array[(x + 4) % 5] ^ array[(x + 1) % 5].rotate_left(1);
            }
        }
        let a_copy = *a;

        let mut circuit = Circuit::<Z251>::new();
        let mut matrix = &mut circuit.new_keccakmatrix();
        circuit.set_keccakmatrix(&matrix, input);

        circuit.theta(matrix);
        circuit.evaluate_keccakmatrix(&matrix) == a_copy
    }
    fn set_keccackmatrix_prop(rand: Vec<u64>) -> bool {
        let mut input: [u64; 25] =
            [   15, 468, 45, 647, 567, 4, 95, 267, 48, 465
            , 5468, 567, 25,   1,   0, 0,  9,   1,  3,   4
            , 5, 7, 786, 564, 9999];
        rand.iter().zip(0..25).for_each(|(&num, i)| input[i] = num);
        let mut circuit = Circuit::<Z251>::new();
        let matrix = circuit.new_keccakmatrix();
        circuit.set_keccakmatrix(&matrix, input);

        circuit.evaluate_keccakmatrix(&matrix) == input
    }
    fn word8_prop(num: u8) -> bool {
        let mut circuit = Circuit::<Z251>::new();
        let u8_input = circuit.new_word8();
        circuit.set_word8(&u8_input, num);
        circuit.evaluate_word8(&u8_input) == num
    }
    fn word64_prop(num: u64) -> bool {
        let mut circuit = Circuit::<Z251>::new();
        let u64_input = circuit.new_word64();
        circuit.set_word64(&u64_input, num);
        circuit.evaluate_word64(&u64_input) == num
    }
}
