//! NEREIDS command-line interface.

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "nereids",
    version,
    about = "NEutron REsonance Imaging Diagnostic Suite"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Display version information.
    Version,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Version) | None => {
            println!("nereids {}", env!("CARGO_PKG_VERSION"));
        }
    }
}
