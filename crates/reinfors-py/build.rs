use std::path::PathBuf;
use std::process::Command;

// Bake source identity into the wheel so run manifests can record the exact commit.
// Anchored to THIS crate's repository root (never a parent repo an archive might sit
// inside); builds without a git checkout report sha "unknown" and dirty "unknown".
// Dirty state is best-effort at build time: manifests additionally hash the built
// extension, which is the stronger evidence of the executable actually used.

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo_root())
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()
        .map(|s| s.trim().to_string())
}

fn main() {
    // Confirm the anchored root IS a repository whose top-level is our root (rejects
    // inheriting identity from an enclosing unrelated repository).
    let toplevel = git(&["rev-parse", "--show-toplevel"]).map(PathBuf::from);
    let anchored = toplevel
        .and_then(|t| t.canonicalize().ok())
        .zip(repo_root().canonicalize().ok())
        .map(|(t, r)| t == r)
        .unwrap_or(false);

    let (sha, dirty, tag) = if anchored {
        (
            git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
            match git(&["status", "--porcelain"]) {
                Some(s) if s.is_empty() => "false",
                Some(_) => "true",
                None => "unknown",
            }
            .to_string(),
            git(&["describe", "--tags", "--exact-match"]).unwrap_or_default(),
        )
    } else {
        ("unknown".into(), "unknown".into(), String::new())
    };
    println!("cargo:rustc-env=REINFORS_GIT_SHA={sha}");
    println!("cargo:rustc-env=REINFORS_GIT_DIRTY={dirty}");
    println!("cargo:rustc-env=REINFORS_GIT_TAG={tag}");
    // Rerun tracking. A commit on a branch moves the branch's loose ref (HEAD itself is
    // unchanged), a new tag lands under refs/ or packed-refs — watch all of them, plus
    // HEAD (checkout switches, detached commits) and the index (staged edits). Unstaged
    // edits touch none of these: the dirty flag is refreshed via the REINFORS_BUILD_NONCE
    // hook below (set it to force a rebuild), and preflight cross-checks the wheel's
    // identity against the checkout it was built from.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
        println!("cargo:rerun-if-changed={git_dir}/index");
    }
    if let Some(common) = git(&["rev-parse", "--path-format=absolute", "--git-common-dir"]) {
        println!("cargo:rerun-if-changed={common}/packed-refs");
        println!("cargo:rerun-if-changed={common}/refs");
    }
    if let Some(head_ref) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&[
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            &head_ref,
        ]) {
            // A packed-only ref has no loose file; cargo treats the missing path as
            // always-changed, which errs toward rebuilding — the safe direction.
            println!("cargo:rerun-if-changed={path}");
        }
    }
    println!("cargo:rerun-if-env-changed=REINFORS_BUILD_NONCE");
}
