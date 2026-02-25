// Hanzo Engine - Wrapper around mistral.rs server
// This provides a simplified CLI interface for running Hanzo AI inference engine

use std::env;
use std::process::Command;

fn main() {
    // Get the arguments
    let args: Vec<String> = env::args().collect();

    // Default port for Hanzo Engine
    let default_port = "36900";

    // If no arguments or help is requested, show help
    if args.len() == 1 || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        println!("Hanzo Engine v0.6.0 - High-performance AI inference engine");
        println!();
        println!("USAGE:");
        println!("    hanzo-engine [OPTIONS]");
        println!();
        println!("OPTIONS:");
        println!("    --port <PORT>          Port to listen on [default: 36900]");
        println!("    --host <HOST>          Host to bind to [default: 0.0.0.0]");
        println!("    -h, --help             Print help");
        println!("    -V, --version          Print version");
        println!();
        println!("EXAMPLES:");
        println!("    # Start server on default port (36900)");
        println!("    hanzo-engine");
        println!();
        println!("    # Start server on custom port");
        println!("    hanzo-engine --port 8080");
        println!();
        println!("NOTE: Hanzo Engine uses mistral.rs under the hood.");
        println!("      For advanced options, use mistralrs-server directly.");
        return;
    }

    if args.contains(&"--version".to_string()) || args.contains(&"-V".to_string()) {
        println!("Hanzo Engine v0.6.0");
        return;
    }

    // Parse port from arguments
    let port = args
        .iter()
        .position(|arg| arg == "--port")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or(default_port);

    // Parse host from arguments
    let host = args
        .iter()
        .position(|arg| arg == "--host")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("0.0.0.0");

    println!("🚀 Starting Hanzo Engine on {}:{}", host, port);
    println!("📊 Embeddings API: http://{}:{}/v1/embeddings", host, port);
    println!("💬 Chat API: http://{}:{}/v1/chat/completions", host, port);
    println!();

    // Call mistralrs-server with appropriate arguments
    let status = Command::new("mistralrs-server")
        .arg("--port")
        .arg(port)
        .arg("--log")
        .arg("info")
        .status();

    match status {
        Ok(exit_status) => {
            if !exit_status.success() {
                eprintln!("❌ Hanzo Engine exited with error");
                std::process::exit(exit_status.code().unwrap_or(1));
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to start Hanzo Engine: {}", e);
            eprintln!();
            eprintln!("Make sure mistralrs-server is installed:");
            eprintln!("    cargo build --release --bin mistralrs-server");
            std::process::exit(1);
        }
    }
}
