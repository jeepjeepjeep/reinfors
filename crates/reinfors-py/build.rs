use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn main() {
    // Bake source identity into the wheel so run manifests can record the exact commit.
    // Builds from tarballs (no .git) fall back to "unknown".
    let sha = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let tag = git(&["describe", "--tags", "--exact-match"]).unwrap_or_default();
    println!("cargo:rustc-env=REINFORS_GIT_SHA={sha}");
    println!(
        "cargo:rustc-env=REINFORS_GIT_DIRTY={}",
        if dirty { "1" } else { "0" }
    );
    println!("cargo:rustc-env=REINFORS_GIT_TAG={tag}");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs");
}
