//! The public `hanzo-engine` binary: the command with no extensions.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    hanzo_cli::run(Vec::new()).await
}
