use std::{env, path::Path, process::Command};

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/benchmark/run_a100.py");
    let status = Command::new("python3")
        .arg(&script)
        .args(env::args_os().skip(1))
        .status()
        .with_context(|| format!("launch benchmark runner `{}`", script.display()))?;
    std::process::exit(status.code().unwrap_or(1));
}
