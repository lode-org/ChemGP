use std::env;
use std::path::PathBuf;
use std::process::Command;

fn first_glob(root: &PathBuf, pattern: &str) -> Option<PathBuf> {
    let mut current = vec![root.clone()];
    for part in pattern.split('/') {
        let mut next = Vec::new();
        if part.contains('*') {
            let prefix = part.trim_end_matches('*');
            for base in &current {
                if let Ok(entries) = std::fs::read_dir(base) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|name| name.starts_with(prefix))
                        {
                            next.push(path);
                        }
                    }
                }
            }
        } else {
            for base in &current {
                next.push(base.join(part));
            }
        }
        current = next;
    }
    current.into_iter().find(|path| path.exists())
}

fn query_rgpot_libdirs(rgpot_root: &PathBuf) -> Vec<PathBuf> {
    let python = rgpot_root.join(".pixi/envs/metatomicbld/bin/python");
    if !python.exists() {
        return Vec::new();
    }

    let script = r#"
from pathlib import Path
import os
import torch, metatensor, metatensor.torch, metatomic.torch, vesin

print(Path(torch.__file__).resolve().parent / "lib")
print(Path(metatensor.__file__).resolve().parent / "lib")
print(Path(metatensor.torch.utils.cmake_prefix_path).resolve().parent)
print(Path(metatomic.torch.utils.cmake_prefix_path).resolve().parent)
print(Path(vesin.__file__).resolve().parent / "lib")
"#;

    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .output()
        .expect("Failed to query rgpot metatomic environment");
    if !output.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .collect()
}

fn main() {
    println!("cargo:rerun-if-env-changed=RGPOT_BUILD_DIR");

    let use_rpc = env::var_os("CARGO_FEATURE_RGPOT").is_some();
    let use_local = env::var_os("CARGO_FEATURE_RGPOT_LOCAL").is_some();

    if use_rpc && use_local {
        panic!("chemgp-core supports either 'rgpot' or 'rgpot_local' in one build, not both");
    }

    if !use_local {
        return;
    }

    let build_dir = PathBuf::from(
        env::var("RGPOT_BUILD_DIR")
            .expect("RGPOT_BUILD_DIR must point to the rgpot Meson build directory"),
    );
    let rgpot_root = build_dir
        .parent()
        .expect("RGPOT_BUILD_DIR should have a parent directory")
        .to_path_buf();

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("CppCore/rgpot/MetatomicPot").display()
    );
    for subdir in ["cargo-target/release", "cargo-target/debug"] {
        let path = build_dir.join(subdir);
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
    let mut queried = query_rgpot_libdirs(&rgpot_root);
    if queried.is_empty() {
        for pattern in [
            ".pixi/envs/metatomicbld/lib/python*/site-packages/torch/lib",
            ".pixi/envs/metatomicbld/lib/python*/site-packages/metatensor/lib",
            ".pixi/envs/metatomicbld/lib/python*/site-packages/metatensor/torch/torch-*/lib",
            ".pixi/envs/metatomicbld/lib/python*/site-packages/metatomic/torch/torch-*/lib",
            ".pixi/envs/metatomicbld/lib/python*/site-packages/vesin/lib",
        ] {
            if let Some(path) = first_glob(&rgpot_root, pattern) {
                queried.push(path);
            }
        }
    }
    for path in queried {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    println!("cargo:rustc-link-lib=dylib=rgpot_core");
    println!("cargo:rustc-link-lib=dylib=metatomicpot");
    println!("cargo:rustc-link-lib=dylib=c10");
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=metatensor");
    println!("cargo:rustc-link-lib=dylib=metatensor_torch");
    println!("cargo:rustc-link-lib=dylib=metatomic_torch");
    println!("cargo:rustc-link-lib=dylib=vesin");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=dl");
}
