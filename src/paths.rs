use std::fs;
use std::path::{Path, PathBuf};

/// Returns an absolute path inside the project directory.
pub fn project_path(path: impl AsRef<Path>) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

/// Ensures the shader directory exists and returns its absolute path.
pub fn shader_dir() -> PathBuf {
    let dir = project_path("data/shaders");
    let _ = fs::create_dir_all(&dir);
    dir
}
