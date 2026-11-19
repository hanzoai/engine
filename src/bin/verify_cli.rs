use clap::Parser;
use std::fs;
use hanzo::verify::{Verifier, AttestationEvidence};

#[derive(Parser, Debug)]
#[command(author, version, about = "Hanzo Remote Attestation Client", long_about = None)]
struct Args {
    #[arg(short, long)]
    evidence: String,

    #[arg(short, long)]
    expected_mrenclave: String,

    #[arg(short, long)]
    expected_model: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let data = fs::read_to_string(&args.evidence)?;
    let evidence: AttestationEvidence = serde_json::from_str(&data)?;

    let verifier = Verifier::new(&args.expected_mrenclave, &args.expected_model);

    match verifier.verify(&evidence) {
        Ok(report) => {
            println!("✅ Verification Successful!");
            println!("Provider: {}", report.provider);
            println!("Timestamp: {}", report.timestamp);
        }
        Err(e) => {
            eprintln!("❌ Verification Failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
