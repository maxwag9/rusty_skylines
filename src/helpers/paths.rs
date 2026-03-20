use std::fs;
use std::path::{Path, PathBuf};

fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .expect("Failed to get executable path")
        .parent()
        .expect("Executable has no parent directory")
        .to_path_buf()
}

/// Finds the data directory by searching multiple candidate locations.
fn find_data_root() -> PathBuf {
    let exe = exe_dir();

    // Check candidates in priority order
    let candidates: &[PathBuf] = &[
        exe.join("data"),          // Distribution: data/ beside exe
        exe.join("../data"),       // Distribution: data/ one level up
        exe.join("../../data"),    // Dev: target/release/ or target/debug/
        exe.join("../../../data"), // Dev: nested workspace crate
    ];

    for candidate in candidates {
        if candidate.is_dir() {
            // Canonicalize resolves ".." and returns absolute path
            if let Ok(resolved) = candidate.canonicalize() {
                println!("[data_path] Found data directory: {}", resolved.display());
                return resolved;
            }
        }
    }

    // Log what we tried (helps debugging distribution issues)
    eprintln!("[data_path] ERROR: Could not find data directory!");
    eprintln!("[data_path] Executable directory: {}", exe.display());
    eprintln!("[data_path] Searched:");
    for c in candidates {
        eprintln!("  - {} (exists: {})", c.display(), c.exists());
    }

    // Return most likely distribution path for meaningful error messages
    exe.join("data")
}

fn rusty_skylines_root() -> PathBuf {
    // 1. Prefer portable folder if it exists
    let portable = exe_dir().join("RustySkylines");
    if portable.exists() {
        return portable;
    }

    // 2. Otherwise fallback to OS data dir
    let base = dirs::data_local_dir().expect("Failed to get local data directory");
    let dir = base.join("RustySkylines");

    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[data_path] Failed to create app dir: {}", e);
    }
    dir
}

/// Cache the data root to avoid repeated filesystem checks
fn data_root() -> &'static PathBuf {
    use std::sync::OnceLock;
    static DATA_ROOT: OnceLock<PathBuf> = OnceLock::new();
    DATA_ROOT.get_or_init(find_data_root)
}

pub fn data_dir(path: impl AsRef<Path>) -> PathBuf {
    data_root().join(path.as_ref())
}
pub fn rusty_skylines_dir(path: impl AsRef<Path>) -> PathBuf {
    rusty_skylines_root().join(path.as_ref())
}
pub fn shader_dir() -> PathBuf {
    let dir = data_dir("shaders");
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[data_path] Failed to create shader dir: {}", e);
    }
    dir
}

pub fn texture_dir() -> PathBuf {
    let dir = shader_dir().join("textures");
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[data_path] Failed to create texture dir: {}", e);
    }
    dir
}

pub fn compute_shader_dir() -> PathBuf {
    let dir = shader_dir().join("compute");
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[data_path] Failed to create compute shader dir: {}", e);
    }
    dir
}

pub fn saves_dir() -> PathBuf {
    let dir = rusty_skylines_dir("saves");
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[data_path] Failed to create saves dir: {}", e);
    }
    dir
}

pub fn screenshots_dir() -> PathBuf {
    let dir = rusty_skylines_dir("screenshots");
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[data_path] Failed to create screenshots dir: {}", e);
    }
    dir
}
pub fn next_screenshot_path() -> PathBuf {
    let dir = screenshots_dir();

    let now = chrono::Local::now();
    let base = now.format("%Y-%m-%d__%Hh-%Mm-%Ss").to_string(); // "2026-10-13__13h-56m-03s"

    // Try without suffix first
    let path = dir.join(format!("{}.png", base));

    if !path.exists() {
        return path;
    }

    // If exists, append _2, _3, ...
    for i in 2.. {
        let candidate = dir.join(format!("{}_{}.png", base, i));
        if !candidate.exists() {
            return candidate;
        }
    }

    unreachable!()
}
