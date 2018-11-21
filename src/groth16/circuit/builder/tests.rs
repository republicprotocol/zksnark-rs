use super::super::super::Z251;
use super::*;
use field::FieldIdentity;

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
fn keccak_f1600_theta_rotate_test() {
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
                    rotate_word64_left(array[(x + 1) % 5], 1)].iter(), Circuit::new_xor);
                }
            }
        }
    }

    theta_rotate_part(tiny_keccak_input, tiny_keccak_array);

    assert_eq!(circuit.evaluate_keccakmatrix(a), *tiny_keccak_input);
}

#[test]
fn keccak_f1600_theta_test() {
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
                    rotate_word64_left(array[(x + 1) % 5], 1)].iter(), Circuit::new_xor);
                }
            }
        }
    }

    theta(tiny_keccak_input);

    assert_eq!(circuit.evaluate_keccakmatrix(a), *tiny_keccak_input);
}

#[test]
fn keccak_f1600_single_test() {
    let tiny_keccak_input: &mut [u64; 25] = &mut [0; 25];
    // rand.iter().zip(0..25).for_each(|(&num, i)| tiny_keccak_input[i] = num );

    let mut circuit = Circuit::<Z251>::new();

    let matrix = &mut circuit.new_keccakmatrix();
    circuit.set_keccakmatrix(matrix, tiny_keccak_input);

    circuit.keccakf_1600(matrix);

    keccakf(tiny_keccak_input);

    assert_eq!(circuit.evaluate_keccakmatrix(matrix), *tiny_keccak_input);
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

quickcheck! {

    /// check if tiny_keccak's permutation function is the same as the circuit's
    /// implementation.

    fn rotate_and_u64_bitwise_op(left: u64, right: u64) -> bool {
        let mut circuit = Circuit::<Z251>::new();

        let left_word = circuit.new_word64();
        let right_word = circuit.new_word64();
        circuit.set_word64(&left_word, left);
        circuit.set_word64(&right_word, right);

        let complete_circuit = circuit.u64_bitwise_op(&left_word,
                    &rotate_word64_left(right_word, 1), Circuit::new_xor);

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

//  # θ step
//  C[x] = A[x,0] xor A[x,1] xor A[x,2] xor A[x,3] xor A[x,4],   for x in 0…4
//  D[x] = C[x-1] xor rot(C[x+1],1),                             for x in 0…4
//  A[x,y] = A[x,y] xor D[x],                           for (x,y) in (0…4,0…4)
// fn theta(a: &mut [[u64; 5]; 5]) {
//     let mut c: [u64; 5] = [0; 5];
//     (0..5).for_each(|x| c[x] = a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4]);

//     let mut d: [u64; 5] = [0; 5];
//     (0..5).for_each(|x| d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1));

//     iproduct!(0..5, 0..5).for_each(|(x, y)| a[x][y] = a[x][y] ^ d[x]);
// }

// # ρ and π steps
// B[y,2*x+3*y] = rot(A[x,y], r[x,y]),                 for (x,y) in (0…4,0…4)
//
// # χ step
// A[x,y] = B[x,y] xor ((not B[x+1,y]) and B[x+2,y]),  for (x,y) in (0…4,0…4)
//
// fn rho_pi_chi(a: &mut [[u64; 5]; 5]) {
//     let mut b: [[u64; 5]; 5] = [[0; 5]; 5];
//     iproduct!(0..5, 0..5)
//         .for_each(|(x, y)| b[y][(2 * x + 3 * y) % 5] = a[x][y].rotate_left(R[x][y]));

//     iproduct!(0..5, 0..5)
//         .for_each(|(x, y)| a[x][y] = b[x][y] ^ (!(b[(x + 1) % 5][y]) & b[(x + 2) % 5][y]));
// }

// fn last_step(a: &mut [[u64; 5]; 5], rc: u64) {
//     a[0][0] = a[0][0] ^ rc;
// }

// fn round(a: &mut [[u64; 5]; 5], rc: u64) {
//     theta(a);
//     pi_step3(a);
//     last_step(a, rc);
// }

// fn keccak_f1600(a: &mut [[u64; 5]; 5]) {
//     (0..24).for_each(|n| round(a, ROUND_CONSTANTS[n]))
// }

// fn flatten(a: &[[u64; 5]; 5]) -> [u64; 25] {
//     let mut arr: [u64; 25] = [0; 25];

//     iproduct!(0..5, 0..5).for_each(|(x, y)| arr[(x * 5) + y] = a[x][y]);
//     arr
// }

// fn to_matrix(a: &[u64; 25]) -> [[u64; 5]; 5] {
//     let mut matrix = [[0; 5]; 5];

//     iproduct!(0..5, 0..5).for_each(|(x, y)| matrix[x][y] = a[(x * 5) + y]);
//     matrix
// }
