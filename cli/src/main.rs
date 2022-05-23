
use clap::{Parser, Subcommand};

use std::str::FromStr;

use std::io::{stdout, Write};
use std::fs::{File};

use std::string::String;

use zksnark::{CoefficientPoly, ASTParser, QAP, FrLocal, TryParse, SigmaG1, SigmaG2};
use zksnark::groth16::fr::{G1Local, G2Local, Proof};

extern crate borsh;
use borsh::{BorshSerialize, BorshDeserialize};

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Debug, Parser)]
struct Cli {
    /// The command to execute
    #[clap(subcommand)]
    command: Commands
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[clap(arg_required_else_help = true)]
    Setup {
        #[clap(long, parse(from_os_str))]
        zk_path: Option<std::path::PathBuf>,
        #[clap(long, parse(from_os_str))]
        output_path: Option<std::path::PathBuf>
    },
    Proof {
        #[clap(long)]
        assignments: Option<String>,
        #[clap(long, parse(from_os_str))]
        setup_path: Option<std::path::PathBuf>,
        #[clap(long, parse(from_os_str))]
        output_path: Option<std::path::PathBuf>
    },
    Verify {
        #[clap(long)]
        assignments: Option<String>,
        #[clap(long, parse(from_os_str))]
        setup_path: Option<std::path::PathBuf>,
        #[clap(long, parse(from_os_str))]
        proof_path: Option<std::path::PathBuf>
    }
}

fn do_string_output(output_path: Option<std::path::PathBuf>, output_string: String) {

    let mut out_writer = match output_path {
        Some(x) => {
            Box::new(File::create(&x).unwrap()) as Box<dyn Write>
        }
        None => Box::new(stdout()) as Box<dyn Write>,
    };
    
    out_writer.write_all(output_string.as_bytes());
}

fn do_binary_output(output_path: std::path::PathBuf,  buf: Vec<u8>) -> File {
    let mut file = File::create(&output_path).unwrap();
    file.write_all(&buf);
    return file
}

fn read_bin_file<V: Decodable>(setup_path: std::path::PathBuf) -> V {
    let setup_bin = &*::std::fs::read(setup_path).unwrap();
    return decode::<V>(setup_bin).unwrap();
}

// arbitrary check value addeed to the *File structs so that we can ensure they are deserialized correctly
// in unit tests
const CHECK: u32 = 0xABAD1DEA;

#[derive(RustcDecodable, RustcEncodable)]
struct SetupFile {
    check: u32,
    code: String,
    qap: QAP<CoefficientPoly<FrLocal>>,
    sigmag1: SigmaG1<G1Local>,
    sigmag2: SigmaG2<G2Local>
}

fn setup(zk_path: std::path::PathBuf, output_path: std::path::PathBuf) {

    let code = &*::std::fs::read_to_string(zk_path).unwrap();
    let qap: QAP<CoefficientPoly<FrLocal>> = ASTParser::try_parse(code).unwrap().into();

    let (sigmag1, sigmag2) = zksnark::groth16::setup(&qap);

    let setup_file_object = SetupFile {check: CHECK, qap: qap, code: String::from(code), sigmag1: sigmag1, sigmag2: sigmag2};

    // do_string_output(output_path, json::encode(&setup_file_object).unwrap());
    let encoded =  encode(&setup_file_object, Infinite).unwrap();
    do_binary_output(output_path, encoded);
}

#[derive(RustcDecodable, RustcEncodable)]
struct ProofFile {
    check: u32,
    proof: Proof<G1Local, G2Local>
}

fn proof(assignments: &[FrLocal], setup_path: std::path::PathBuf, output_path: std::path::PathBuf) 
    // where F: Clone + zksnark::field::Field + FromStr + PartialEq, 
    {

    let setup: SetupFile = read_bin_file(setup_path);
    let weights = zksnark::groth16::weights(&setup.code, assignments).unwrap();

    let proof = zksnark::groth16::prove(&setup.qap, (&setup.sigmag1, &setup.sigmag2), &weights);
    let proof_file = ProofFile {check: CHECK, proof: proof};
    let encoded =  encode(&proof_file, Infinite).unwrap();
    do_binary_output(output_path, encoded);
}

fn verify(assignments: &[FrLocal], setup_path: std::path::PathBuf, proof_path: std::path::PathBuf) -> bool {
    let setup: SetupFile = read_bin_file(setup_path);
    let proof: ProofFile = read_bin_file(proof_path);
    return zksnark::groth16::verify::<CoefficientPoly<FrLocal>, _, _, _, _> (
        (setup.sigmag1, setup.sigmag2),
        assignments,
        proof.proof
    );
}

fn parse_assignment_string(s: &str) -> Vec<FrLocal> {
    return s.split(',').map(|item| FrLocal::from_str(item).unwrap()).into_iter().collect::<Vec<FrLocal>>();
}

fn read_setup_file<'a>(setup_path: std::path::PathBuf) -> Result<SetupFile, self::rustc_serialize::json::DecoderError> {
    let setup_json = &*::std::fs::read_to_string(setup_path).unwrap();
    let setup: Result< SetupFile, self::rustc_serialize::json::DecoderError> = json::decode(setup_json);

    return setup;
}

fn proof(setup_path: std::path::PathBuf, output_path: Option<std::path::PathBuf>) {
}

// command line example from https://github.com/clap-rs/clap/blob/v3.1.18/examples/git-derive.rs

fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Setup { zk_path, output_path }  => setup(zk_path.unwrap(), output_path.unwrap()),
        Commands::Proof { assignments, setup_path, output_path }  => {
            proof(&parse_assignment_string(&assignments.unwrap()[..]), setup_path.unwrap(), output_path.unwrap());
        },
        Commands::Verify { assignments, setup_path, proof_path }  => {
            verify(&parse_assignment_string(&assignments.unwrap()[..]), setup_path.unwrap(), proof_path.unwrap());
        },
        _ => println!("unknown command!"),
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn input_assignments() -> [FrLocal; 3] {
        return [
            FrLocal::from(3), // a
            FrLocal::from(2), // b
            FrLocal::from(4) // c
        ];
    } 

    fn output_assignments() -> [FrLocal; 2] {
        return [
            FrLocal::from(2),
            FrLocal::from(34)
        ];
    } 

    #[test]
    fn try_setup_test() {
        setup(PathBuf::from("../test_programs/simple.zk"), PathBuf::from("simple.setup.bin"));
        assert!(true);
    }

    #[test]
    fn try_read_setup_test() {
        let setup: SetupFile = read_bin_file(PathBuf::from("simple.setup.bin"));
        assert!(setup.check == CHECK)
    }

    #[test]
    fn try_proof_test() {
        proof(&input_assignments(), PathBuf::from("simple.setup.bin"), PathBuf::from("simple.proof.bin"));
        assert!(true);
    }

    #[test]
    fn try_verify_test() {
        assert!(verify(&output_assignments(), PathBuf::from("simple.setup.bin"), PathBuf::from("simple.proof.bin")));
    }

    #[test]
    fn try_read_proof_test() {
        let setup: ProofFile = read_bin_file(PathBuf::from("simple.proof.bin"));
        assert!(setup.check == CHECK)
    }

    #[test]
    fn complete_test() {
        extern crate zksnark;

        // from test_programs/simple.zk
        // x = 4ab + c + 6
        let code = r#"(in a b c)
            (out x)
            (verify b x)
            
            (program
                (= temp
                    (* a b))
                (= x
                    (* 1 (+ (* 4 temp) c 6))))"#;

        let qap: QAP<CoefficientPoly<FrLocal>> =
            ASTParser::try_parse(code)
                .unwrap()
                .into();

        let weights = zksnark::groth16::weights(code, &input_assignments()).unwrap();

        let (sigmag1, sigmag2) = zksnark::groth16::setup(&qap);

        let proof = zksnark::groth16::prove(&qap, (&sigmag1, &sigmag2), &weights);

        assert!(zksnark::groth16::verify::<CoefficientPoly<FrLocal>, _, _, _, _>(
            (sigmag1, sigmag2),
            &output_assignments(),
            proof
        ));
    }
}
