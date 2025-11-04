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
        println!("    hanzo-engine [OPTIONS] [MODEL_ID]");
        println!();
        println!("ARGS:");
        println!("    <MODEL_ID>    HuggingFace model ID or local path [optional]");
        println!();
        println!("OPTIONS:");
        println!("    --port <PORT>          Port to listen on [default: 36900]");
        println!("    --host <HOST>          Host to bind to [default: 0.0.0.0]");
        println!("    -h, --help             Print help");
        println!("    -V, --version          Print version");
        println!();
        println!("EXAMPLES:");
        println!("    # Start server on default port (36900) in basic mode");
        println!("    hanzo-engine");
        println!();
        println!("    # Start with a specific embedding model");
        println!("    hanzo-engine Qwen/Qwen3-Embedding-8B");
        println!();
        println!("    # Start on custom port");
        println!("    hanzo-engine --port 8080");
        println!();
        println!("NOTE: Hanzo Engine uses mistral.rs under the hood.");
        println!("      Set MISTRALRS_DEBUG=1 for detailed logging.");
        println!();
        println!("ENDPOINTS:");
        println!("    POST /v1/embeddings           Generate embeddings");
        println!("    POST /v1/chat/completions     Chat completions");
        println!("    GET  /v1/models               List models");
        println!("    GET  /health                  Health check");
        return;
    }
    
    if args.contains(&"--version".to_string()) || args.contains(&"-V".to_string()) {
        println!("Hanzo Engine v0.6.0");
        return;
    }
    
    // Parse port from arguments
    let port = args.iter()
        .position(|arg| arg == "--port")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or(default_port);
    
    // Parse host from arguments  
    let host = args.iter()
        .position(|arg| arg == "--host")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("0.0.0.0");

    // Check if model ID is provided (first non-option arg)
    let model_id = args.iter()
        .skip(1)
        .find(|arg| !arg.starts_with('-') && !port.contains(*arg) && !host.contains(*arg));
    
    println!("🚀 Starting Hanzo Engine on {}:{}", host, port);
    println!("📊 Embeddings API: http://{}:{}/v1/embeddings", host, port);
    println!("💬 Chat API: http://{}:{}/v1/chat/completions", host, port);
    
    if let Some(model) = model_id {
        println!("📦 Model: {}", model);
    } else {
        println!("⚠️  No model specified - server will start but needs model for inference");
    }
    println!();
    
    // Build command based on whether model is specified
    let mut cmd = Command::new("mistralrs-server");
    
    if let Some(model) = model_id {
        // Use 'run' subcommand for auto-loading a model
        cmd.arg("run")
            .arg("--model-id")
            .arg(model);
    } else {
        // Use 'plain' subcommand with a dummy model for basic server
        // This allows the server to start and accept requests
        cmd.arg("plain")
            .arg("--model-id")
            .arg("microsoft/Phi-3-mini-4k-instruct"); // Lightweight default model
    }
    
    cmd.arg("--port")
        .arg(port)
        .arg("--serve-ip")
        .arg(host)
        .arg("--log")
        .arg("info");
    
    let status = cmd.status();
    
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
