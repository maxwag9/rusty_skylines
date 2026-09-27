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
    let portable = exe_dir().join("RustySkylines");
    if portable.exists() {
        return portable;
    }

    let base = dirs::document_dir()
        .or_else(dirs::data_local_dir)
        .expect("Failed to get documents or local data directory");

    let dir = base.join("RustySkylines");
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[data_path] Failed to create app dir: {}", e);
    }
    dir
}

fn mods_root_impl() -> PathBuf {
    rusty_skylines_root().join("mods")
}

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

pub fn mods_root() -> PathBuf {
    mods_root_impl()
}

pub fn mods_dir() -> PathBuf {
    mods_root()
}

pub fn mods_config_path() -> PathBuf {
    mods_root().join("mods.toml")
}

pub fn user_mod_cache_dir() -> PathBuf {
    mods_root().join(".mod_cache")
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
    let base = now.format("RS_%Y-%m-%d_%H.%M.%S").to_string();
    let path = dir.join(format!("{}.png", base));

    if !path.exists() {
        return path;
    }

    for i in 2..69 {
        let candidate = dir.join(format!("{}_{}.png", base, i));
        if !candidate.exists() {
            return candidate;
        }
    }

    unreachable!()
}
