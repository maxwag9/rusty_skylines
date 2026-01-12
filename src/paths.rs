use std::fs;
use std::path::{Path, PathBuf};

fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

pub fn data_dir(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();

    // 1. Try runtime path beside the executable
    let runtime = exe_dir().join("data").join(path);
    if runtime.exists() {
        return runtime;
    }

    // 2. Fallback: use project directory during development
    // (CARGO_MANIFEST_DIR only exists in debug / IDE)
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(path);

    dev
}
/// Ensures the shader directory exists and returns its absolute path.
pub fn shader_dir() -> PathBuf {
    let dir = data_dir("shaders");
    let _ = fs::create_dir_all(&dir);
    dir
}

pub fn texture_dir() -> PathBuf {
    let dir = shader_dir().join("textures");
    let _ = fs::create_dir_all(&dir);
    dir
}
