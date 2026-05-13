//! Hanzo Engine CLI binary.
//!
//! Two modes:
//!  * `hanzo-engine serve [--port PORT] [--host HOST]` — forwards to
//!    `mistralrs-server` for the full HTTP server experience.
//!  * `hanzo-engine version` — prints version info from the library.
//!
//! The library (`hanzo_engine`) is what consumers integrate against
//! programmatically. This binary is a thin wrapper for quick local serving.

use std::env;
use std::process::Command;

const DEFAULT_PORT: &str = "36900";
const DEFAULT_HOST: &str = "0.0.0.0";

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 1 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        print_version();
        return;
    }

    let port = arg_value(&args, "--port").unwrap_or(DEFAULT_PORT);
    let host = arg_value(&args, "--host").unwrap_or(DEFAULT_HOST);

    println!(
        "Starting Hanzo Engine v{} on {}:{}",
        env!("CARGO_PKG_VERSION"),
        host,
        port
    );
    println!("  Embeddings API: http://{host}:{port}/v1/embeddings");
    println!("  Chat API:       http://{host}:{port}/v1/chat/completions");

    // Show inference / embedding registration state at startup so it's
    // obvious whether a model was wired in.
    println!(
        "  Inference engine registered: {}",
        hanzo_engine::inference_engine_registered()
    );
    println!(
        "  Embedding engine registered: {}",
        hanzo_engine::embedding_engine_registered()
    );

    let status = Command::new("mistralrs-server")
        .arg("--port")
        .arg(port)
        .arg("--log")
        .arg("info")
        .status();

    match status {
        Ok(exit_status) if exit_status.success() => {}
        Ok(exit_status) => {
            eprintln!("Hanzo Engine exited with error");
            std::process::exit(exit_status.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("Failed to start Hanzo Engine: {e}");
            eprintln!();
            eprintln!("Make sure mistralrs-server is installed:");
            eprintln!("    cargo build --release --bin mistralrs-server");
            std::process::exit(1);
        }
    }
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn print_version() {
    println!("Hanzo Engine v{}", env!("CARGO_PKG_VERSION"));
}

fn print_help() {
    println!(
        "Hanzo Engine v{} - canonical inference + embedding engine",
        env!("CARGO_PKG_VERSION")
    );
    println!();
    println!("USAGE:");
    println!("    hanzo-engine [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --port <PORT>          Port to listen on [default: {DEFAULT_PORT}]");
    println!("    --host <HOST>          Host to bind to    [default: {DEFAULT_HOST}]");
    println!("    -h, --help             Print help");
    println!("    -V, --version          Print version");
    println!();
    println!("LIBRARY:");
    println!("    For programmatic use (precompiles, agents, RPC handlers),");
    println!("    depend on the `hanzo_engine` crate and call:");
    println!("        hanzo_engine::register_inference_engine(...)");
    println!("        hanzo_engine::infer(model_id, prompt)");
    println!("        hanzo_engine::embed(dim, text)");
}
