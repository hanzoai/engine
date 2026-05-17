// Hanzo Engine - thin CLI front for mistralrs-server
//
// M1 of the multimodal router (`MULTIMODAL_ROUTER_DESIGN.md`) puts the
// real `--register` clap parsing in `mistralrs-server` itself (where the
// model lifecycle lives). This binary stays a wrapper: it owns the
// hanzo-flavored help text and default port, then forwards every other
// argument — including any number of `--register ID:KIND:LOCATION` flags —
// straight through to `mistralrs-server`.
//
// Pass-through rules:
//   * `--port <p>`         consumed here, defaulted to 36900, re-emitted.
//   * `--host <h>`         consumed here, defaulted to 0.0.0.0, displayed.
//   * `--help` / `-h`      handled locally, prints hanzo-flavored help.
//   * `--version` / `-V`   handled locally.
//   * everything else      forwarded verbatim.

use std::env;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();

    let default_port = "36900";

    if args.len() == 1 || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_help();
        return;
    }

    if args.contains(&"--version".to_string()) || args.contains(&"-V".to_string()) {
        println!("Hanzo Engine v0.6.0");
        return;
    }

    // Parse --port locally (defaulted) and --host locally (display only).
    // Every other arg, *including any --register flags*, is forwarded.
    let port = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or(default_port)
        .to_string();

    let host = args
        .iter()
        .position(|a| a == "--host")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("0.0.0.0")
        .to_string();

    // Build the forwarded args. Skip argv[0] (our binary name) and skip
    // any --host pair (mistralrs-server takes --serve-ip, not --host) but
    // keep --port pair. Pass --serve-ip explicitly so the bind addr matches
    // what we printed.
    let mut forwarded: Vec<String> = Vec::new();
    forwarded.push("--serve-ip".to_string());
    forwarded.push(host.clone());
    forwarded.push("--port".to_string());
    forwarded.push(port.clone());

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                // Already handled.
                i += 2;
            }
            "--host" => {
                // Already mapped to --serve-ip.
                i += 2;
            }
            "--log" if i + 1 < args.len() => {
                // Honor an explicit --log if present.
                forwarded.push("--log".to_string());
                forwarded.push(args[i + 1].clone());
                i += 2;
            }
            other => {
                forwarded.push(other.to_string());
                i += 1;
            }
        }
    }

    // Default log level if not provided.
    if !forwarded.iter().any(|a| a == "--log") {
        forwarded.push("--log".to_string());
        forwarded.push("info".to_string());
    }

    // List any --register flags for the operator's benefit.
    let register_specs: Vec<&String> = args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            if a == "--register" {
                args.get(i + 1)
            } else {
                None
            }
        })
        .collect();

    println!("Hanzo Engine starting on {host}:{port}");
    println!("  Chat:        http://{host}:{port}/v1/chat/completions");
    println!("  Models:      http://{host}:{port}/v1/models");
    if !register_specs.is_empty() {
        println!("  Registered experts:");
        for spec in &register_specs {
            println!("    - {spec}");
        }
    } else {
        println!("  Registered experts: (default:inprocess:auto auto-added)");
    }
    println!();

    let status = Command::new("mistralrs-server").args(&forwarded).status();

    match status {
        Ok(exit_status) => {
            if !exit_status.success() {
                eprintln!("Hanzo Engine exited with error");
                std::process::exit(exit_status.code().unwrap_or(1));
            }
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

fn print_help() {
    println!("Hanzo Engine v0.6.0 - High-performance AI inference engine");
    println!();
    println!("USAGE:");
    println!("    hanzo-engine [OPTIONS] [-- <model-selector args...>]");
    println!();
    println!("OPTIONS:");
    println!("    --port <PORT>                    Port to listen on [default: 36900]");
    println!("    --host <HOST>                    Host to bind to   [default: 0.0.0.0]");
    println!("    --register <ID:KIND:LOCATION>    Register an expert in the ModelRegistry.");
    println!("                                     KIND ∈ {{inprocess, proxy, subprocess}}.");
    println!("                                     LOCATION = `auto` (inprocess), URL (proxy),");
    println!("                                     or path (subprocess). Repeatable.");
    println!("    --log <LEVEL>                    Log level [default: info]");
    println!("    -h, --help                       Print help");
    println!("    -V, --version                    Print version");
    println!();
    println!("All other arguments are forwarded verbatim to mistralrs-server,");
    println!("including the model-selector subcommand (plain, gguf, vision-plain, ...).");
    println!();
    println!("EXAMPLES:");
    println!("    # Default port, in-process model only (auto-registers default).");
    println!("    hanzo-engine plain -m meta-llama/Llama-3.2-1B-Instruct");
    println!();
    println!("    # M1 router stub: register a remote sibling alongside the local model.");
    println!("    hanzo-engine \\");
    println!("        --register default:inprocess:auto \\");
    println!("        --register vision:proxy:http://127.0.0.1:8001/v1 \\");
    println!("        plain -m meta-llama/Llama-3.2-1B-Instruct");
    println!();
    println!("NOTE: Hanzo Engine uses mistral.rs under the hood.");
}
