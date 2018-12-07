use super::super::super::Z251;
use super::*;
use field::FieldIdentity;
use std::time::{Duration, Instant};

extern crate quickcheck;
use self::quickcheck::quickcheck;

extern crate tiny_keccak;
use self::tiny_keccak::keccak256;
use self::tiny_keccak::keccakf;
use self::tiny_keccak::Keccak;

// TODO: Replace all instances of Z251 with FrLocal
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
fn less_than_test() {
    let logic_table = [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0)];
    let mut circuit = Circuit::<Z251>::new();
    let l_wire = circuit.new_wire();
    let r_wire = circuit.new_wire();
    let less_than = circuit.new_less_than(l_wire, r_wire);

    for (l, r, l_less_r) in logic_table.iter() {
        circuit.reset();
        circuit.set_value(l_wire, Z251::from(*l));
        circuit.set_value(r_wire, Z251::from(*r));
        assert!(circuit.evaluate(less_than) == Z251::from(*l_less_r));
    }
}

#[test]
fn greater_than_test() {
    let logic_table = [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0)];
    let mut circuit = Circuit::<Z251>::new();
    let l_wire = circuit.new_wire();
    let r_wire = circuit.new_wire();
    let greater_than = circuit.new_greater_than(l_wire, r_wire);

    for (l, r, l_greater_r) in logic_table.iter() {
        circuit.reset();
        circuit.set_value(l_wire, Z251::from(*l));
        circuit.set_value(r_wire, Z251::from(*r));
        assert!(circuit.evaluate(greater_than) == Z251::from(*l_greater_r));
    }
}

fn test_test() {
    let mut circuit = Circuit::<Z251>::new();
    let l_wire = circuit.new_word8();
    let r_wire = circuit.new_word8();
    let greater_than = circuit.new_word8_greater_than(l_wire, r_wire);
    circuit.set_word8(&l_wire, 0);
    circuit.set_word8(&r_wire, 1);
    assert!(circuit.evaluate(greater_than) == Z251::from(0));
}

#[test]
fn word8_greater_than_test() {
    let mut circuit = Circuit::<Z251>::new();
    let l_wire = circuit.new_word8();
    let r_wire = circuit.new_word8();
    let greater_than = circuit.new_word8_greater_than(l_wire, r_wire);

    for (l_num, r_num) in iproduct!(0..u8::max_value(), 0..u8::max_value()) {
        circuit.reset();
        circuit.set_word8(&l_wire, l_num);
        circuit.set_word8(&r_wire, r_num);
        // println!("this was tried: ({}, {})", l_num, r_num);
        if l_num < r_num {
            assert!(circuit.evaluate(greater_than) == Z251::from(0));
        } else if l_num == r_num {
            assert!(circuit.evaluate(greater_than) == Z251::from(0));
        } else {
            assert!(circuit.evaluate(greater_than) == Z251::from(1));
        }
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

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Word8/64 Tests ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#[test]
fn const_word64_sanity_check() {
    let mut circuit = Circuit::<Z251>::new();
    let const_u64 = circuit.const_word64(0b0000_0100);
    assert_eq!(circuit.evaluate_word64(&const_u64), 0b000_0100);
    assert_eq!(circuit.evaluate(const_u64[0][0]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u64[0][1]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u64[0][2]), Z251::one());
    assert_eq!(circuit.evaluate(const_u64[0][3]), Z251::zero());

    assert_eq!(circuit.evaluate(const_u64[0][4]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u64[0][5]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u64[0][6]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u64[0][7]), Z251::zero());
}

#[test]
fn const_word8_sanity_check() {
    let mut circuit = Circuit::<Z251>::new();
    let const_u8 = circuit.const_word8(0b0000_0100);
    assert_eq!(circuit.evaluate_word8(&const_u8), 0b000_0100);
    assert_eq!(circuit.evaluate(const_u8[0]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u8[1]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u8[2]), Z251::one());
    assert_eq!(circuit.evaluate(const_u8[3]), Z251::zero());

    assert_eq!(circuit.evaluate(const_u8[4]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u8[5]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u8[6]), Z251::zero());
    assert_eq!(circuit.evaluate(const_u8[7]), Z251::zero());
}

#[test]
fn word64_set_eval() {
    let mut circuit = Circuit::<Z251>::new();
    let u64_input = circuit.new_word64();
    circuit.set_word64(&u64_input, 1);
    assert_eq!(circuit.evaluate_word64(&u64_input), 1);
}

#[test]
fn set_word8_sanity_check() {
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
fn set_word64_sanity_check() {
    let mut circuit = Circuit::<Z251>::new();
    let w64 = circuit.new_word64();
    circuit.set_word64(&w64, 0b0100_1011_0100_1111);
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

#[test]
fn u64_fan_in_single_test() {
    let input: [u64; 5] = [1, 0, 35, 5, 6];

    let mut circuit = Circuit::<Z251>::new();

    let mut elems: [Word64; 5] = [Word64::default(); 5];
    input.iter().enumerate().for_each(|(i, &num)| {
        let wrd64 = circuit.new_word64();
        circuit.set_word64(&wrd64, num);
        elems[i] = wrd64;
    });

    let complete_circuit = circuit.u64_fan_in(elems.iter(), Circuit::new_xor);

    assert_eq!(
        circuit.evaluate_word64(&complete_circuit),
        1 ^ 0 ^ 35 ^ 5 ^ 6
    );
}

#[test]
fn set_new_word8_array() {
    let mut circuit = Circuit::<Z251>::new();

    let external_input: &mut [u8; 7] = &mut [9, 24, 45, 250, 99, 0, 7];
    let circuit_input: &mut [Word8; 7] = &mut [Word8::default(); 7];
    circuit.set_new_word8_array(external_input.iter(), circuit_input);

    let eval_circuit: &mut [u8; 7] = &mut [0; 7];
    circuit.evaluate_word8_to_array(circuit_input.iter(), eval_circuit);

    assert_eq!(eval_circuit, external_input);
}

/// Check that the circuit builder for u64_fan_in has the semantics of
/// taking `[a, b, c, d, ... z]` into `a xor b xor c xor d ... xor z`.
#[test]
#[ignore]
fn u64_fan_in_prop() {
    fn prop(rand: Vec<u64>) -> bool {
        let mut input: [u64; 25] = [0; 25];
        rand.iter().zip(0..25).for_each(|(&num, i)| input[i] = num);

        let mut circuit = Circuit::<Z251>::new();

        let row = circuit.new_keccakmatrix();
        circuit.set_keccakmatrix(&row, &input);

        let complete_circuit = circuit.u64_fan_in(row.iter(), Circuit::new_xor);

        circuit.evaluate_word64(&complete_circuit)
            == input.iter().skip(1).fold(input[0], |acc, x| acc ^ x)
    }
    quickcheck(prop as fn(Vec<u64>) -> bool);
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Keccak Tests //////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////// Single Tests //////////////////////////////

#[test]
fn keccakf_1600_theta_rotate_test() {
    let tiny_keccak_input: &mut [u64; 25] = &mut [0; 25];
    let tiny_keccak_array: [u64; 5] = [0; 5];

    let circuit_input: &mut [u64; 25] = &mut [0; 25];
    let array: [Word64; 5] = [Word64::default(); 5];

    let rand = [0, 9546, 6264, 57, 0, 0, 0, 99, 1];
    rand.iter().zip(0..25).for_each(|(&num, i)| {
        tiny_keccak_input[i] = num;
        circuit_input[i] = num;
    });

    let mut circuit = Circuit::<Z251>::new();
    let a = &mut circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(a, circuit_input);

    unroll! {
        for x in 0..5 {
            unroll! {
                for y_count in 0..5 {
                    let y = y_count * 5;
                    a[y + x] = circuit.u64_fan_in([a[y + x], array[(x + 4) % 5],
                    types::rotate_word64_left(array[(x + 1) % 5], 1)].iter(), Circuit::new_xor);
                }
            }
        }
    }

    fn theta_rotate_part(a: &mut [u64; 25], array: [u64; 5]) {
        unroll! {
            for x in 0..5 {
                unroll! {
                    for y_count in 0..5 {
                        let y = y_count * 5;
                        a[y + x] ^= array[(x + 4) % 5] ^ array[(x + 1) % 5].rotate_left(1);
                    }
                }
            }
        }
    }

    theta_rotate_part(tiny_keccak_input, tiny_keccak_array);

    assert_eq!(circuit.evaluate_keccakmatrix(a), *tiny_keccak_input);
}

#[test]
fn keccakf_1600_theta_test() {
    let tiny_keccak_input: &mut [u64; 25] = &mut [0; 25];
    let circuit_input: &mut [u64; 25] = &mut [0; 25];
    let rand = [
        0, 9546, 6264, 57, 0, 0, 86798, 99, 1, 987978, 4568798, 555, 22222, 0,
    ];
    rand.iter().zip(0..25).for_each(|(&num, i)| {
        tiny_keccak_input[i] = num;
        circuit_input[i] = num;
    });

    let mut circuit = Circuit::<Z251>::new();
    let a = &mut circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(a, circuit_input);

    let mut array: [Word64; 5] = [Word64::default(); 5];

    // Theta
    unroll! {
        for x in 0..5 {
            unroll! {
                for y_count in 0..5 {
                    let y = y_count * 5;
                    array[x] = circuit.u64_bitwise_op(&array[x], &a[x + y], Circuit::new_xor);
                }
            }
        }
    }

    unroll! {
        for x in 0..5 {
            unroll! {
                for y_count in 0..5 {
                    let y = y_count * 5;
                    a[y + x] = circuit.u64_fan_in([a[y + x], array[(x + 4) % 5],
                    types::rotate_word64_left(array[(x + 1) % 5], 1)].iter(), Circuit::new_xor);
                }
            }
        }
    }

    fn theta(a: &mut [u64; 25]) {
        let mut array: [u64; 5] = [0; 5];

        // Theta
        unroll! {
            for x in 0..5 {
                unroll! {
                    for y_count in 0..5 {
                        let y = y_count * 5;
                        array[x] ^= a[x + y];
                    }
                }
            }
        }

        unroll! {
            for x in 0..5 {
                unroll! {
                    for y_count in 0..5 {
                        let y = y_count * 5;
                        a[y + x] ^= array[(x + 4) % 5] ^ array[(x + 1) % 5].rotate_left(1);
                    }
                }
            }
        }
    }

    theta(tiny_keccak_input);

    assert_eq!(circuit.evaluate_keccakmatrix(a), *tiny_keccak_input);
}

#[test]
fn keccakf_1600_single_test() {
    let tiny_keccak_input: &mut [u64; 25] = &mut [0; 25];
    // rand.iter().zip(0..25).for_each(|(&num, i)| tiny_keccak_input[i] = num );

    let mut circuit = Circuit::<Z251>::new();

    let build = Instant::now();
    let matrix = &mut circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(matrix, tiny_keccak_input);

    circuit.keccakf_1600(matrix);

    keccakf(tiny_keccak_input);
    let build_time = build.elapsed().subsec_micros();

    let eval = Instant::now();
    assert_eq!(circuit.evaluate_keccakmatrix(matrix), *tiny_keccak_input);
    let eval_time = eval.elapsed().subsec_micros();
    println!(
        "keccaf_1600 circuit took {} microseconds to build and {} microseconds to evaluate",
        build_time, eval_time
    );
}

#[test]
fn keccak_absorb_squeeze_prop() {
    fn prop(rand: Vec<u8>) -> bool {
        const LEN: usize = 67;

        let mut keccak = Keccak::new_keccak256();
        let mut to_be_absorbed: [u8; LEN] = [0; LEN];

        rand.iter()
            .zip(0..to_be_absorbed.len())
            .for_each(|(from_rand, i)| to_be_absorbed[i] = *from_rand);
        keccak.absorb(&to_be_absorbed);

        let mut keccak_output: [u8; 32] = [0; 32];
        keccak.squeeze(&mut keccak_output);

        let mut circuit_output: [Word8; 32] = [Word8::default(); 32];
        let mut circuit = Circuit::<Z251>::new();
        let matrix = &mut circuit.new_keccakmatrix();
        circuit.set_keccakmatrix(matrix, &[0; 25]);
        let circuit_keccak_struct = &mut KeccakInternal {
            a: *matrix,
            offset: 0,
            rate: (200 - (256 / 4)),
            delim: 0x01,
        };

        let mut circuit_to_be_absorbed: [Word8; LEN] = [Word8::default(); LEN];
        rand.iter()
            .zip(0..circuit_to_be_absorbed.len())
            .for_each(|(from_rand, i)| {
                circuit_to_be_absorbed[i] = circuit.set_new_word8(*from_rand)
            });

        circuit.absorb(circuit_keccak_struct, &circuit_to_be_absorbed);
        circuit.squeeze(circuit_keccak_struct, &mut circuit_output);

        let mut circuit_converted_output: [u8; 32] = [0; 32];
        circuit_output
            .iter()
            .enumerate()
            .for_each(|(i, wrd8)| circuit_converted_output[i] = circuit.evaluate_word8(wrd8));

        circuit_converted_output == keccak_output
    }
    quickcheck(prop as fn(Vec<u8>) -> bool);
}

#[test]
fn keccak_absorb_squeeze_single_test() {
    let mut keccak = Keccak::new_keccak256();
    keccak.absorb(&[25, 26, 26]);
    let mut keccak_output: [u8; 32] = [0; 32];
    keccak.squeeze(&mut keccak_output);

    let mut circuit_output: [Word8; 32] = [Word8::default(); 32];
    let mut circuit = Circuit::<Z251>::new();
    let matrix = &mut circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(matrix, &[0; 25]);
    let circuit_keccak_struct = &mut KeccakInternal {
        a: *matrix,
        offset: 0,
        rate: (200 - (256 / 4)),
        delim: 0x01,
    };
    let circuit_25 = circuit.set_new_word8(25);
    let circuit_26 = circuit.set_new_word8(26);
    circuit.absorb(circuit_keccak_struct, &[circuit_25, circuit_26, circuit_26]);
    circuit.squeeze(circuit_keccak_struct, &mut circuit_output);

    let mut circuit_converted_output: [u8; 32] = [0; 32];
    circuit_output
        .iter()
        .enumerate()
        .for_each(|(i, wrd8)| circuit_converted_output[i] = circuit.evaluate_word8(wrd8));

    assert_eq!(circuit_converted_output, keccak_output);
}

#[test]
fn keccak_absorb_pad_squeeze_single_test() {
    let mut keccak = Keccak::new_keccak256();
    const LEN: usize = 137;
    let input: [u8; LEN] = [79; LEN];
    keccak.absorb(&input);
    let mut keccak_output: [u8; 32] = [0; 32];
    keccak.pad();
    keccak.squeeze(&mut keccak_output);

    let mut circuit = Circuit::<Z251>::new();
    let matrix = &mut circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(matrix, &[0; 25]);
    let circuit_keccak_struct = &mut KeccakInternal {
        a: *matrix,
        offset: 0,
        rate: (200 - (256 / 4)),
        delim: 0x01,
    };

    let mut circuit_input: [Word8; LEN] = [Word8::default(); LEN];
    input
        .iter()
        .enumerate()
        .for_each(|(i, &x)| circuit_input[i] = circuit.set_new_word8(x));

    circuit.absorb(circuit_keccak_struct, &circuit_input);
    circuit.pad(circuit_keccak_struct);
    let mut circuit_output: [Word8; 32] = [Word8::default(); 32];
    circuit.squeeze(circuit_keccak_struct, &mut circuit_output);

    let mut circuit_converted_output: [u8; 32] = [0; 32];

    circuit_output
        .iter()
        .enumerate()
        .for_each(|(i, wrd8)| circuit_converted_output[i] = circuit.evaluate_word8(wrd8));

    assert_eq!(circuit_converted_output, keccak_output);
}

#[test]
fn keccak_pad_and_squeeze_single_test() {
    let keccak = Keccak::new_keccak256();
    let mut keccak_output: [u8; 32] = [0; 32];
    keccak.finalize(&mut keccak_output);

    let mut circuit_output: [Word8; 32] = [Word8::default(); 32];
    let mut circuit = Circuit::<Z251>::new();
    let matrix = &mut circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(matrix, &[0; 25]);
    let circuit_keccak_struct = &mut KeccakInternal {
        a: *matrix,
        offset: 0,
        rate: (200 - (256 / 4)),
        delim: 0x01,
    };
    circuit.pad(circuit_keccak_struct);
    circuit.keccakf_1600(&mut circuit_keccak_struct.a);
    circuit.squeeze(circuit_keccak_struct, &mut circuit_output);

    let mut circuit_converted_output: [u8; 32] = [0; 32];
    circuit_output
        .iter()
        .enumerate()
        .for_each(|(i, wrd8)| circuit_converted_output[i] = circuit.evaluate_word8(wrd8));

    assert_eq!(circuit_converted_output, keccak_output);
}

#[test]
fn keccak256_equiv_fixed_size_single() {
    let rand = vec![1];
    let rand_offset = 0;

    const LEN: usize = 1067;

    let input: &mut [u8; LEN] = &mut [0; LEN];
    rand.iter()
        .zip(0..input.len())
        .for_each(|(&num, i)| input[(i * (rand_offset + 1)) % input.len()] = num);

    let tiny_output: [u8; 32] = keccak256(input);

    let mut circuit = Circuit::<Z251>::new();
    let circuit_input: &mut [Word8; LEN] = &mut [Word8::default(); LEN];
    circuit.set_new_word8_array(input.iter(), circuit_input);

    let circuit_output: [Word8; 32] = circuit.keccak256(circuit_input);
    let eval_circuit_output: &mut [u8; 32] = &mut [0; 32];
    circuit.evaluate_word8_to_array(circuit_output.iter(), eval_circuit_output);

    assert_eq!(*eval_circuit_output, tiny_output);
}

/////////////////////////////////// Performance Tests //////////////////////////////

/// This function will not fail, instead it is meant to be run
/// with: `cargo test keccak256_metrics -- --nocapture` to print out the
/// metrics until we setup real metrics testing.
#[test]
fn keccak256_metrics() {
    const BYTES: usize = 56;

    let input: &mut [u8; BYTES] = &mut [
        150, 234, 20, 196, 120, 146, 1, 48, 157, 10, 170, 174, 183, 246, 34, 204, 110, 184, 31,
        155, 70, 130, 115, 205, 179, 165, 27, 165, 104, 31, 7, 16, 157, 242, 34, 232, 56, 161, 8,
        150, 228, 129, 153, 41, 144, 186, 190, 41, 16, 59, 242, 109, 102, 75, 12, 246,
    ];

    let build = Instant::now();
    let mut circuit = Circuit::<Z251>::new();
    let circuit_input: &mut [Word8; BYTES] = &mut [Word8::default(); BYTES];
    circuit.set_new_word8_array(input.iter(), circuit_input);
    let circuit_output: [Word8; 32] = circuit.keccak256(circuit_input);
    let build_time = build.elapsed().subsec_micros();

    let eval = Instant::now();
    let eval_circuit_output: &mut [u8; 32] = &mut [0; 32];
    circuit.evaluate_word8_to_array(circuit_output.iter(), eval_circuit_output);
    let eval_time = eval.elapsed().subsec_micros();

    println!(
        "keccak256 circuit took {} microseconds to build and {} microseconds to evaluate with 7 u64 as input",
        build_time, eval_time
    );
}

//////////////////////////////////// Quickcheck Tests //////////////////////////////
#[test]
#[ignore]
fn keccak256_stream_equiv_prop() {
    fn prop(rand: Vec<u8>, rand_offset: usize) -> bool {
        const LEN: usize = 79;

        let input: &mut [u8; LEN] = &mut [0; LEN];
        rand.iter()
            .zip(0..input.len())
            .for_each(|(&num, i)| input[(i * (rand_offset + 1)) % input.len()] = num);

        let mut circuit_stream = Circuit::<Z251>::new();
        let keccak_stream_input: Vec<Word8> = circuit_stream.set_new_word8_vec(input.iter());
        let keccak_stream_circuit: [Word8; 32] =
            circuit_stream.keccak256_stream(keccak_stream_input.iter());
        let keccak_stream_result: Vec<u8> =
            circuit_stream.evaluate_word8_to_vec(keccak_stream_circuit.iter());

        let mut circuit = Circuit::<Z251>::new();
        let circuit_input: &mut [Word8; LEN] = &mut [Word8::default(); LEN];
        circuit.set_new_word8_array(input.iter(), circuit_input);

        let circuit_output: [Word8; 32] = circuit.keccak256(circuit_input);
        let eval_circuit_output: &mut [u8; 32] = &mut [0; 32];
        circuit.evaluate_word8_to_array(circuit_output.iter(), eval_circuit_output);

        eval_circuit_output.iter().cloned().collect::<Vec<u8>>() == keccak_stream_result
    }
    quickcheck(prop as fn(Vec<u8>, usize) -> bool);
}

/// check if tiny_keccak's keccak256 is the same as the circuit's
/// implementation.
#[test]
#[ignore]
fn keccak256_equiv_fixed_size_prop() {
    fn prop(rand: Vec<u8>, rand_offset: usize) -> bool {
        const LEN: usize = 79;

        let input: &mut [u8; LEN] = &mut [0; LEN];
        rand.iter()
            .zip(0..input.len())
            .for_each(|(&num, i)| input[(i * (rand_offset + 1)) % input.len()] = num);

        let tiny_output: [u8; 32] = keccak256(input);

        let mut circuit = Circuit::<Z251>::new();
        let circuit_input: &mut [Word8; LEN] = &mut [Word8::default(); LEN];
        circuit.set_new_word8_array(input.iter(), circuit_input);

        let circuit_output: [Word8; 32] = circuit.keccak256(circuit_input);
        let eval_circuit_output: &mut [u8; 32] = &mut [0; 32];
        circuit.evaluate_word8_to_array(circuit_output.iter(), eval_circuit_output);

        *eval_circuit_output == tiny_output
    }
    quickcheck(prop as fn(Vec<u8>, usize) -> bool);
}

/// check if tiny_keccak's permutation function is the same as the circuit's
/// implementation.
#[test]
#[ignore]
fn keccakf_1600_equiv_prop() {
    fn prop(rand: Vec<u64>) -> bool {
        let tiny_keccak_input: &mut [u64; 25] = &mut [0; 25];
        rand.iter()
            .zip(0..25)
            .for_each(|(&num, i)| tiny_keccak_input[i] = num);

        let mut circuit = Circuit::<Z251>::new();

        let matrix = &mut circuit.new_keccakmatrix();
        circuit.set_keccakmatrix(matrix, tiny_keccak_input);

        circuit.keccakf_1600(matrix);

        keccakf(tiny_keccak_input);

        circuit.evaluate_keccakmatrix(matrix) == *tiny_keccak_input
    }
    quickcheck(prop as fn(Vec<u64>) -> bool);
}

quickcheck! {
    /// Checks that a rotation combined with a bitwise xor works as expected
    fn rotate_and_u64_bitwise_op(left: u64, right: u64) -> bool {
        let mut circuit = Circuit::<Z251>::new();

        let left_word = circuit.new_word64();
        let right_word = circuit.new_word64();
        circuit.set_word64(&left_word, left);
        circuit.set_word64(&right_word, right);

        let complete_circuit = circuit.u64_bitwise_op(&left_word,
                    &types::rotate_word64_left(right_word, 1), Circuit::new_xor);

        circuit.evaluate_word64(&complete_circuit) == left ^ right.rotate_left(1)
    }

    /// Checks that xor of Word64 is done correctly
    fn u64_bitwise_op_prop(left: u64, right: u64) -> bool {
        let mut circuit = Circuit::<Z251>::new();

        let left_word = circuit.new_word64();
        let right_word = circuit.new_word64();
        circuit.set_word64(&left_word, left);
        circuit.set_word64(&right_word, right);

        let complete_circuit = circuit.u64_bitwise_op(&left_word, &right_word, Circuit::new_xor);

        circuit.evaluate_word64(&complete_circuit) == left ^ right

    }


    /// I wanted to check that creating a new KeccakMatrix, setting it from an
    /// array and then evaluating that KeccakMatrix would result in the same
    /// array. Its somewhat a sanity check and to make sure the setting /
    /// evaluating are not messing up the overall result.
    fn set_keccackmatrix_prop(rand: Vec<u64>) -> bool {
        let mut input: [u64; 25] =
            [   15, 468, 45, 647, 567, 4, 95, 267, 48, 465
            , 5468, 567, 25,   1,   0, 0,  9,   1,  3,   4
            , 5, 7, 786, 564, 9999];
        rand.iter().zip(0..25).for_each(|(&num, i)| input[i] = num);
        let mut circuit = Circuit::<Z251>::new();
        let matrix = &mut circuit.new_keccakmatrix();
        circuit.set_keccakmatrix(matrix, &input);

        circuit.evaluate_keccakmatrix(matrix) == input
    }

    /// Like a smaller version of KeccakMatrix, just need to make sure the new,
    /// set, evaluate of Word8 does not change the value of the initial u8
    /// number.
    fn word8_prop(num: u8) -> bool {
        let mut circuit = Circuit::<Z251>::new();
        let u8_input = circuit.new_word8();
        circuit.set_word8(&u8_input, num);
        circuit.evaluate_word8(&u8_input) == num
    }

    /// Like a smaller version of KeccakMatrix, just need to make sure the new,
    /// set, evaluate of Word64 does not change the value of the initial u64
    /// number.
    fn word64_prop(num: u64) -> bool {
        let mut circuit = Circuit::<Z251>::new();
        let u64_input = circuit.new_word64();
        circuit.set_word64(&u64_input, num);
        circuit.evaluate_word64(&u64_input) == num
    }
}
