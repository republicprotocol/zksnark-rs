
use clap::{Parser, Subcommand};

use std::io::{stdout, Write, BufWriter};
use std::fs::{File};

use zksnark::{CoefficientPoly, ASTParser, QAP, FrLocal, TryParse, SigmaG1, SigmaG2};
use zksnark::groth16::fr::{G1Local, G2Local};

extern crate rustc_serialize;
use self::rustc_serialize::json;

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
        #[clap(long, parse(from_os_str))]
        setup_path: Option<std::path::PathBuf>,
        #[clap(long, parse(from_os_str))]
        output_path: Option<std::path::PathBuf>
    }
}

fn do_output(output_path: Option<std::path::PathBuf>, output_string: String) {

    let mut out_writer = match output_path {
        Some(x) => {
            Box::new(File::create(&x).unwrap()) as Box<dyn Write>
        }
        None => Box::new(stdout()) as Box<dyn Write>,
    };
    
    out_writer.write_all(output_string.as_bytes());
}

#[derive(RustcDecodable, RustcEncodable)]
struct SetupFile {
    sigmag1: SigmaG1<G1Local>,
    sigmag2: SigmaG2<G2Local>
}

fn setup(zk_path: std::path::PathBuf, output_path: Option<std::path::PathBuf>) {

    let code = &*::std::fs::read_to_string(zk_path).unwrap();
    let qap: QAP<CoefficientPoly<FrLocal>> = ASTParser::try_parse(code).unwrap().into();

    // let qap_json = &*::std::fs::read_to_string(qap_path).unwrap();
    // let qap: Result< QAP<CoefficientPoly<FrLocal>>, _> = json::decode(qap_json);

    let (sigmag1, sigmag2) = zksnark::groth16::setup(&qap);

    let setup_file_object = SetupFile {sigmag1: sigmag1, sigmag2: sigmag2};

    do_output(output_path, json::encode(&setup_file_object).unwrap());
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
        Commands::Setup { zk_path, output_path }  => setup(zk_path.unwrap(), output_path),
        Commands::Proof { setup_path, output_path }  => proof(setup_path.unwrap(), output_path),
        _ => println!("unknown command!"),
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn try_read_setup_test() {
        let result = read_setup_file(PathBuf::from("simple.json"));
        assert!(!result.is_err(), "setup file did not parse")
    }}