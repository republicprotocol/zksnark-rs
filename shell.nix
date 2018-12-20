let
  moz_overlay = import ((import <nixpkgs> {}).fetchFromGitHub 
    { owner = "mozilla";
      repo = "nixpkgs-mozilla";
      inherit 
       ({ url = "https://github.com/mozilla/nixpkgs-mozilla";
          rev = "c985206e160204707e15a45f0b9df4221359d21c";
          sha256 = "0k0p3nfzr3lfgp1bb52bqrbqjlyyiysf8lq2rnrmn759ijxy2qmq";
          fetchSubmodules = false;
	}) rev sha256;
    });
  
  nixpkgs = import <nixpkgs> { overlays = [ moz_overlay ]; };

  stable-rust = 
    (nixpkgs.rustChannelOf 
      { date = "2018-12-06"; 
        channel = "stable";
      }
    );

	nightly-rust = 
		(nixpkgs.rustChannelOf 
			{ date = "2018-11-08";
				channel = "nightly"; 
			}
		);

  nightlyBuildRustPlatform = with nixpkgs; recurseIntoAttrs 
  		(makeRustPlatform 
  			(
  				{
  					rustc = nightly-rust.rustc;
            cargo = nightly-rust.cargo;
          }
  			)
  		);
  
  tarpaulin = with nixpkgs; nightlyBuildRustPlatform.buildRustPackage rec {
  	name = "tarpaulin-${version}";
  	version = "0.6.10";
  
  	src = fetchFromGitHub {
  		owner = "xd009642";
  		repo = "tarpaulin";
  		rev = "${version}";
  		sha256 = "1jw4jlrgv5an53q3idamkybv8i29426hmmf7x8kvr8lzbfcpsjc1";
  	};
  	
  	buildInputs = [ nixpkgs.openssl nixpkgs.zlib ];

    nativeBuildInputs = [ nixpkgs.pkgconfig nixpkgs.cmake nightly-rust.rust-std ];

    RUSTFLAGS="--cfg procmacro2_semver_exempt";

  	doCheck = false;
  
  	cargoSha256 = "0r0r633l07dnkbj77358asp6180d0y6psqfnrq2l5pkdshb0gb0w";
  
  	meta = with stdenv.lib; {
  		description = "A code coverage tool for Rust projects";
  		homepage = https://github.com/xd009642/tarpaulin;
  		license = licenses.mit;
  		maintainers = [ maintainers.tailhook ];
  		platforms = platforms.all;
  	};
  };
in
  with nixpkgs;
  stdenv.mkDerivation {
    name = "rust-env";
    buildInputs = [
      stable-rust.rust 
      stable-rust.rls-preview
      stable-rust.rustfmt-preview
      stable-rust.rust-analysis
      stable-rust.clippy-preview

      rustracer carnix 
    ];

    # Set Environment Variables
    RUST_BACKTRACE = 1;
    RUST_RACER_PATH = "${rustracer}/bin/racer";
    RUST_SRC_PATH = "${stable-rust.rust-src}/lib/rustlib/src/rust/src/";
    RUST_STD_DOCS_PATH = "${stable-rust.rust-docs}";
    RUST_STD_PATH = "${stable-rust.rust-std}";
  }
