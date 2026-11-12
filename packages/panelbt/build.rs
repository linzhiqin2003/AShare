//! Embed a digest of every Rust source used by the extension.
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
};
fn sources(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, files);
        } else if path.extension().is_some_and(|x| x == "rs") {
            files.push(path);
        }
    }
}
fn dependency(manifest: &Path, name: &str) -> PathBuf {
    let data: toml::Value = fs::read_to_string(manifest.join("Cargo.toml"))
        .unwrap()
        .parse()
        .unwrap();
    manifest.join(data["dependencies"][name]["path"].as_str().unwrap())
}
fn main() {
    let binding = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let core = dependency(&binding, "panelbt-core");
    let trade = dependency(&core, "tradecore");
    let mut digest = Sha256::new();
    for root in [&binding, &core, &trade] {
        let mut files = Vec::new();
        sources(&root.join("src"), &mut files);
        if root.join("build.rs").exists() {
            files.push(root.join("build.rs"));
        }
        files.sort();
        for file in files {
            println!("cargo:rerun-if-changed={}", file.display());
            digest.update(fs::read(file).unwrap());
        }
        println!(
            "cargo:rerun-if-changed={}",
            root.join("Cargo.toml").display()
        );
    }
    println!(
        "cargo:rustc-env=PANELBT_SOURCE_SHA256={:x}",
        digest.finalize()
    );
}
