use std::{os::unix::process::CommandExt, process::Command};

fn main() {
    println!("cargo::rerun-if-changed=shaders/shader.vert,shaders/shader.frag");

    Command::new("make").current_dir("shaders").exec();
}
