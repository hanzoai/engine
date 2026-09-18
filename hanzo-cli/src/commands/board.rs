//! `hanzo-engine board`: score a run, file it as evidence, pin its manifest.


use anyhow::Result;
use hanzo_bench::board;

use crate::args::BoardCommand;

pub fn run_board(cmd: BoardCommand) -> Result<()> {
    match cmd {
        BoardCommand::Score { run } => print!("{}", board::markdown(&board::score(&run)?)),
        BoardCommand::Publish { runs, to } => {
            let runs: Vec<_> = runs.iter().map(|r| board::read(r)).collect::<Result<_>>()?;
            let evidence = board::evidence(&runs)?;
            match to {
                Some(url) => board::post(&url, &evidence)?,
                None => println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({"experiments": evidence, "attempts": []}))?
                ),
            }
        }
        BoardCommand::Manifest { out, pins } => {
            std::fs::write(&out, serde_json::to_string_pretty(&board::pin(pins)?)?)?;
            println!("manifest -> {}", out.display());
        }
    }
    Ok(())
}

