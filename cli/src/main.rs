
use clap::{Parser, Subcommand};

use std::str::FromStr;

use std::io::{stdout, Write};
use std::fs::{File};

use std::string::String;

use zksnark::{CoefficientPoly, ASTParser, QAP, FrLocal, TryParse, SigmaG1, SigmaG2};
use zksnark::groth16::fr::{G1Local, G2Local, Proof};

extern crate rustc_serialize;
extern crate bincode;

use bincode::SizeLimit::Infinite;
use bincode::rustc_serialize::{encode, decode};

use self::rustc_serialize::{Decodable};

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

    // let qap_json = &*::std::fs::read_to_string(qap_path).unwrap();
    // let qap: Result< QAP<CoefficientPoly<FrLocal>>, _> = json::decode(qap_json);

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

// command line example from https://github.com/clap-rs/clap/blob/v3.1.18/examples/git-derive.rs

fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Setup { zk_path, output_path }  => setup(zk_path.unwrap(), output_path.unwrap()),
        Commands::Proof { assignments, setup_path, output_path }  => {
            proof(&assignments.unwrap().split(',').map(|item| FrLocal::from_str(item).unwrap()).into_iter().collect::<Vec<FrLocal>>(), setup_path.unwrap(), output_path.unwrap());
        },
        _ => println!("unknown command!"),
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn try_setup_test() {
        setup(PathBuf::from("../test_programs/simple.zk"), PathBuf::from("simple.setup.bin"));
    }

    #[test]
    fn try_read_setup_test() {
        let setup: SetupFile = read_bin_file(PathBuf::from("simple.setup.bin"));
        assert!(setup.check == CHECK)
    }

    #[test]
    fn try_proof_test() {
        let assignments = [
            FrLocal::from(3), // a
            FrLocal::from(2), // b
            FrLocal::from(4) // c
    ];
        proof(&assignments, PathBuf::from("simple.setup.bin"), PathBuf::from("simple.proof.bin"));
        assert!(true);
    }

    #[test]
    fn try_read_proof_test() {
        let setup: ProofFile = read_bin_file(PathBuf::from("simple.proof.bin"));
        assert!(setup.check == CHECK)
    }
}