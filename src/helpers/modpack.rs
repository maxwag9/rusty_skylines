use crate::app::GAME_VERSION;
use crate::helpers::paths::{data_dir, mods_config_path, mods_root, user_mod_cache_dir};
use crate::renderer::shader_watcher::ShaderWatcher;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use tracing::{error, info, warn};
use unarc_rs::unified::{ArchiveFormat, ArchiveOptions};

const MOD_CONFIG_VERSION: u32 = 1;
const MOD_MANIFEST: &str = "mod.yaml";
const BUILTIN_MOD_ID: &str = "Base Mod";
const MAX_NESTED_ARCHIVE_DEPTH: usize = 6;
const MAX_EXTRACTED_ARCHIVES: usize = 256;
const MAX_ARCHIVE_FILE_SIZE: u64 = 4 * 1024 * 1024 * 1024;
const MAX_TOTAL_EXTRACTED_SIZE: u64 = 16 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
pub enum ModSource {
    BuiltIn,
    Directory(PathBuf),
    Archive(PathBuf),
}

#[derive(Debug, Clone)]
pub struct Mod {
    pub manifest: ModManifest,
    pub path: PathBuf,
    pub enabled: bool,
    pub source: ModSource,
}
impl Mod {
    pub fn id(&self) -> &String {
        &self.manifest.id
    }
    pub fn is_builtin(&self) -> bool {
        matches!(self.source, ModSource::BuiltIn)
    }
}
#[derive(Debug, Clone, Copy)]
pub enum ModFileKind {
    Menus,
    AdvancedPrimitives,
    Buildings,
    Utilities,
    Sounds,
    Shaders,
    Textures,
    Fonts,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModManifestYaml {
    pub id: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub version: Option<String>,
    pub author: Option<String>,
    pub dependencies: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ModManifest {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub author: String,
    pub dependencies: Vec<String>,
}

impl ModManifest {
    fn base() -> ModManifest {
        ModManifest {
            id: BUILTIN_MOD_ID.to_string(),
            name: "Rusty Skylines fallback".to_string(),
            version: GAME_VERSION.to_string(),
            description: "This is the Base Mod. This is an internal mod, located in the data folder in the game files, which I made. This contains basically everything in the game. All UI Menus, shaders, buildings, sfx, utilities, textures and more. You can edit it in the data folder like any other mod, because it is 'just another mod'™.".to_string(),
            author: "maxwag9 (Guy who made this whole game)".to_string(),
            dependencies: vec![],
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct ModConfig {
    #[serde(default = "config_version")]
    version: u32,
    #[serde(default)]
    load_order: Vec<String>,
    #[serde(default)]
    disabled: Vec<String>,
}

fn config_version() -> u32 {
    MOD_CONFIG_VERSION
}

struct ExtractionBudget {
    archive_count: usize,
    total_written: u64,
}

impl ExtractionBudget {
    fn new() -> Self {
        Self {
            archive_count: 0,
            total_written: 0,
        }
    }

    fn start_archive(&mut self) -> Result<()> {
        self.archive_count += 1;
        if self.archive_count > MAX_EXTRACTED_ARCHIVES {
            return Err(anyhow!(
                "archive extraction limit exceeded: more than {} archives",
                MAX_EXTRACTED_ARCHIVES
            ));
        }
        Ok(())
    }

    fn reserve(&self, bytes: u64) -> Result<()> {
        if bytes > MAX_ARCHIVE_FILE_SIZE {
            return Err(anyhow!(
                "archive entry is too large: {} bytes, limit is {} bytes",
                bytes,
                MAX_ARCHIVE_FILE_SIZE
            ));
        }
        if self.total_written.saturating_add(bytes) > MAX_TOTAL_EXTRACTED_SIZE {
            return Err(anyhow!(
                "archive extraction would exceed total limit of {} bytes",
                MAX_TOTAL_EXTRACTED_SIZE
            ));
        }
        Ok(())
    }
}

struct BudgetWriter<'a, W> {
    inner: W,
    budget: &'a mut ExtractionBudget,
    file_written: u64,
}

impl<'a, W: Write> BudgetWriter<'a, W> {
    fn new(inner: W, budget: &'a mut ExtractionBudget) -> Self {
        Self {
            inner,
            budget,
            file_written: 0,
        }
    }
}

impl<W: Write> Write for BudgetWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len() as u64;
        if self.file_written.saturating_add(len) > MAX_ARCHIVE_FILE_SIZE {
            return Err(io::Error::other("archive file size limit exceeded"));
        }
        if self.budget.total_written.saturating_add(len) > MAX_TOTAL_EXTRACTED_SIZE {
            return Err(io::Error::other(
                "total archive extraction size limit exceeded",
            ));
        }

        let written = self.inner.write(buf)?;
        let written = written as u64;
        self.file_written = self.file_written.saturating_add(written);
        self.budget.total_written = self.budget.total_written.saturating_add(written);
        Ok(written as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[derive(Default)]
pub struct ModManager {
    pub mods: Vec<Mod>,
    config: ModConfig,
    resource_index: HashMap<String, PathBuf>,
}

impl ModManager {
    pub fn new() -> Self {
        let mut manager = Self::default();
        manager.reload(None);
        manager
    }

    pub fn reload(&mut self, shader_watcher: Option<&mut ShaderWatcher>) {
        let config = load_config();
        let mut discovered = Vec::new();
        let mut seen_roots = HashSet::new();
        let mut active_cache_dirs = HashSet::new();

        let base_path = data_dir("Rusty Skylines Mod");
        let base_mod = match load_manifest(base_path.as_path(), "Base Mod") {
            Ok(manifest) => Mod {
                manifest,
                path: base_path,
                enabled: true,
                source: ModSource::BuiltIn,
            },
            Err(e) => {
                error!(
                    "[Mods] Failed to load Base Mod Manifest: {e}, using internal manifest instead."
                );
                Mod {
                    manifest: ModManifest::base(),
                    path: base_path,
                    enabled: true,
                    source: ModSource::BuiltIn,
                }
            }
        };

        let user_root = mods_root();
        if let Err(e) = fs::create_dir_all(&user_root) {
            error!(
                "[Mods] Failed to create user mod directory {}: {}",
                user_root.display(),
                e
            );
        }

        if let Err(e) = scan_user_root(
            &user_root,
            &mut discovered,
            &mut seen_roots,
            &mut active_cache_dirs,
        ) {
            error!("[Mods] Failed while scanning user mods: {}", e);
        }

        discovered.sort_by(|a, b| a.id().cmp(&b.id()));

        let disabled: HashSet<&str> = config.disabled.iter().map(String::as_str).collect();
        let mut by_id = HashMap::with_capacity(discovered.len());

        for mut m in discovered {
            if m.manifest.id == BUILTIN_MOD_ID {
                error!(
                    "[Mods] Mod '{}' is reserved by the game, skipping {}",
                    BUILTIN_MOD_ID,
                    m.path.display()
                );
                continue;
            }

            if by_id.contains_key(&m.manifest.id) {
                error!(
                    "[Mods] Duplicate mod id '{}', skipping {}",
                    m.manifest.id,
                    m.path.display()
                );
                continue;
            }

            m.enabled = !disabled.contains(m.manifest.id.as_str());
            by_id.insert(m.manifest.id.clone(), m);
        }

        let mut ordered = Vec::with_capacity(by_id.len() + 1);
        ordered.push(base_mod);

        for id in &config.load_order {
            if let Some(m) = by_id.remove(id) {
                ordered.push(m);
            }
        }

        let mut remaining: Vec<_> = by_id.into_values().collect();
        remaining.sort_by(|a, b| a.id().cmp(&b.id()));
        ordered.extend(remaining);

        self.mods = ordered;
        self.config = config;
        self.rebuild_resource_index();
        self.sync_config();

        if let Err(e) = save_config(&self.config) {
            error!("[Mods] Failed to save mod configuration: {}", e);
        }

        if let Err(e) = cleanup_mod_cache(&active_cache_dirs) {
            warn!("[Mods] Failed to clean mod cache: {}", e);
        }

        info!("[Mods] Loaded {} mod(s)", self.mods.len());
        for (index, m) in self.mods.iter().enumerate() {
            info!(
                "[Mods] Load order {}: id: {} name: ({}){}",
                index,
                m.manifest.name,
                m.id(),
                if m.enabled { "" } else { " [disabled]" }
            );
        }
    }

    fn rebuild_resource_index(&mut self) {
        let mut index = HashMap::new();

        for m in self.mods.iter().filter(|m| m.enabled) {
            let mut files = Vec::new();

            if let Err(e) = collect_files(&m.path, &mut files) {
                warn!("[Mods] Failed to scan mod '{}': {}", m.manifest.id, e);
                continue;
            }

            for path in files {
                let Ok(relative) = path.strip_prefix(&m.path) else {
                    continue;
                };

                let key = relative.to_string_lossy().replace('\\', "/");
                index.insert(key, path);
            }
        }

        self.resource_index = index;
    }

    pub fn mods(&self) -> &[Mod] {
        &self.mods
    }

    pub fn base_mod(&self) -> &Mod {
        &self.mods[0]
    }

    pub fn user_mods(&self) -> impl Iterator<Item = &Mod> {
        self.mods.iter().filter(|m| !m.is_builtin())
    }

    pub fn enabled_mods(&self) -> impl Iterator<Item = &Mod> {
        self.mods.iter().filter(|m| m.enabled)
    }

    pub fn get(&self, id: &str) -> Option<&Mod> {
        self.mods.iter().find(|m| m.manifest.id == id)
    }

    pub fn set_enabled(&mut self, id: &str, enabled: bool) -> bool {
        let Some(index) = self.mods.iter().position(|m| m.manifest.id == id) else {
            return false;
        };
        if self.mods[index].is_builtin() {
            return false;
        }

        self.mods[index].enabled = enabled;
        self.sync_config();
        self.save_config();
        true
    }

    pub fn move_up(&mut self, id: &str) -> bool {
        let Some(index) = self.mods.iter().position(|m| m.manifest.id == id) else {
            return false;
        };
        if index <= 1 {
            return false;
        }

        self.mods.swap(index, index - 1);
        self.sync_config();
        self.save_config();
        true
    }

    pub fn move_down(&mut self, id: &str) -> bool {
        let Some(index) = self.mods.iter().position(|m| m.manifest.id == id) else {
            return false;
        };
        if index == 0 || index + 1 >= self.mods.len() {
            return false;
        }

        self.mods.swap(index, index + 1);
        self.sync_config();
        self.save_config();
        true
    }

    pub fn move_to(&mut self, id: &str, target_index: usize) -> bool {
        let Some(index) = self.mods.iter().position(|m| m.manifest.id == id) else {
            return false;
        };
        if index == 0 || self.mods.len() <= 1 {
            return false;
        }

        let target_index = target_index.clamp(1, self.mods.len() - 1);
        if index == target_index {
            return false;
        }

        let m = self.mods.remove(index);
        self.mods.insert(target_index, m);
        self.sync_config();
        self.save_config();
        true
    }

    pub fn reset_order(&mut self) {
        if self.mods.len() <= 1 {
            return;
        }

        self.mods[1..].sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
        self.sync_config();
        self.save_config();
    }

    pub fn paths(&self, kind: ModFileKind) -> Vec<PathBuf> {
        match kind {
            ModFileKind::Menus => self.menu_paths(),
            ModFileKind::AdvancedPrimitives => self.advanced_primitive_paths(),
            ModFileKind::Buildings => self.building_paths(),
            ModFileKind::Utilities => self.utility_paths(),
            ModFileKind::Sounds => self.sound_paths(),
            ModFileKind::Shaders => self.shader_paths(),
            ModFileKind::Textures => self.texture_paths(),
            ModFileKind::Fonts => self.font_paths(),
        }
    }

    pub fn menu_paths(&self) -> Vec<PathBuf> {
        self.effective_paths("ui/menus", Some(&["yaml"]), |relative| {
            !relative.starts_with(Path::new("ui/menus/advanced_primitives"))
        })
    }

    pub fn advanced_primitive_paths(&self) -> Vec<PathBuf> {
        self.effective_paths("ui/menus/advanced_primitives", Some(&["yaml"]), |_| true)
    }

    pub fn building_paths(&self) -> Vec<PathBuf> {
        self.effective_paths("simulation/buildings", Some(&["yaml"]), |_| true)
    }

    pub fn utility_paths(&self) -> Vec<PathBuf> {
        self.effective_paths("simulation/utilities", Some(&["yaml"]), |_| true)
    }

    pub fn sound_paths(&self) -> Vec<PathBuf> {
        self.effective_paths("sounds", Some(&["yaml"]), |_| true)
    }

    pub fn shader_paths(&self) -> Vec<PathBuf> {
        self.effective_paths("shaders", Some(&["wgsl"]), |_| true)
    }

    pub fn texture_paths(&self) -> Vec<PathBuf> {
        self.effective_paths("textures", None, |_| true)
    }

    pub fn font_paths(&self) -> Vec<PathBuf> {
        self.effective_paths("ui/ttf", Some(&["ttf", "otf", "ttc"]), |_| true)
    }

    pub fn paths_in(&self, relative_root: impl AsRef<Path>, extensions: &[&str]) -> Vec<PathBuf> {
        self.effective_paths(relative_root.as_ref(), Some(extensions), |_| true)
    }

    pub fn all_paths_in(&self, relative_root: impl AsRef<Path>) -> Vec<PathBuf> {
        self.effective_paths(relative_root.as_ref(), None, |_| true)
    }

    #[inline]
    pub fn resource_path(&self, relative_path: &str) -> Option<&Path> {
        self.resource_index.get(relative_path).map(PathBuf::as_path)
    }

    #[inline]
    pub fn shader_path(&self, relative_path: &str) -> Option<&Path> {
        let path = format!("shaders/{relative_path}");

        let Some(path) = self.resource_path(&path) else {
            error!("[Mods] Missing shader '{path}'");
            return None;
        };

        Some(path)
    }

    fn effective_paths<F>(
        &self,
        relative_root: impl AsRef<Path>,
        extensions: Option<&[&str]>,
        filter: F,
    ) -> Vec<PathBuf>
    where
        F: Fn(&Path) -> bool,
    {
        let relative_root = relative_root.as_ref();
        let mut winning: HashMap<PathBuf, (usize, PathBuf)> = HashMap::new();

        for (mod_index, m) in self.mods.iter().enumerate() {
            if !m.enabled {
                continue;
            }

            let root = m.path.join(relative_root);
            let mut files = Vec::new();
            if let Err(e) = collect_files(&root, &mut files) {
                warn!(
                    "[Mods] Failed to scan {} in mod '{}': {}",
                    relative_root.display(),
                    m.manifest.id,
                    e
                );
                continue;
            }

            files.sort();

            for path in files {
                let Ok(relative) = path.strip_prefix(&m.path) else {
                    continue;
                };
                if !filter(relative) || !extension_matches(relative, extensions) {
                    continue;
                }

                winning.insert(relative.to_path_buf(), (mod_index, path));
            }
        }

        let mut result: Vec<_> = winning.into_iter().collect();
        result.sort_by(|(a, (index_a, _)), (b, (index_b, _))| {
            index_a.cmp(index_b).then_with(|| a.cmp(b))
        });
        result.into_iter().map(|(_, (_, path))| path).collect()
    }

    fn user_mod_ids(&self) -> Vec<String> {
        self.mods
            .iter()
            .filter(|m| !m.is_builtin())
            .map(|m| m.id().clone())
            .collect()
    }

    fn sync_config(&mut self) {
        self.config.version = MOD_CONFIG_VERSION;
        self.config.load_order = self.user_mod_ids();
        self.config.disabled = self
            .mods
            .iter()
            .filter(|m| !m.is_builtin() && !m.enabled)
            .map(|m| m.id().clone())
            .collect();
    }

    fn save_config(&self) {
        if let Err(e) = save_config(&self.config) {
            error!("[Mods] Failed to save mod configuration: {}", e);
        }
    }
}

fn scan_user_root(
    root: &Path,
    discovered: &mut Vec<Mod>,
    seen_roots: &mut HashSet<PathBuf>,
    active_cache_dirs: &mut HashSet<String>,
) -> Result<()> {
    for path in sorted_dir_entries(root)? {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        if name == "mods.toml" || name == ".mod_cache" || name == "saves" || name == "screenshots" {
            continue;
        }

        scan_path(&path, discovered, seen_roots, active_cache_dirs)?;
    }

    Ok(())
}

fn scan_path(
    path: &Path,
    discovered: &mut Vec<Mod>,
    seen_roots: &mut HashSet<PathBuf>,
    active_cache_dirs: &mut HashSet<String>,
) -> Result<()> {
    if path.is_dir() {
        if is_mod_root(path)? {
            add_directory_mod(path, discovered, seen_roots)?;
            return Ok(());
        }

        for child in sorted_dir_entries(path)? {
            scan_path(&child, discovered, seen_roots, active_cache_dirs)?;
        }
        return Ok(());
    }

    if !path.is_file() || !is_archive_candidate(path)? {
        return Ok(());
    }

    let cache_dir = materialize_archive(path)?;
    if let Some(name) = cache_dir.file_name().and_then(|n| n.to_str()) {
        active_cache_dirs.insert(name.to_string());
    }

    discover_mod_roots(
        &cache_dir,
        Some(&archive_stem(path)),
        ModSource::Archive(path.to_path_buf()),
        discovered,
        seen_roots,
    )?;

    Ok(())
}

fn add_directory_mod(
    path: &Path,
    discovered: &mut Vec<Mod>,
    seen_roots: &mut HashSet<PathBuf>,
) -> Result<()> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize mod path {}", path.display()))?;

    if !seen_roots.insert(canonical) {
        return Ok(());
    }

    let fallback = dir_name(path);
    let manifest = load_manifest(path, &fallback)?;

    info!("[Mods] Found mod folder: {}", path.display());
    discovered.push(Mod {
        manifest,
        path: path.to_path_buf(),
        enabled: true,
        source: ModSource::Directory(path.to_path_buf()),
    });

    Ok(())
}

fn discover_mod_roots(
    root: &Path,
    fallback_name: Option<&str>,
    source: ModSource,
    discovered: &mut Vec<Mod>,
    seen_roots: &mut HashSet<PathBuf>,
) -> Result<()> {
    if is_mod_root(root)? {
        add_discovered_mod(root, fallback_name, source, discovered, seen_roots)?;
        return Ok(());
    }

    for path in sorted_dir_entries(root)? {
        if path.is_dir() {
            discover_mod_roots(&path, fallback_name, source.clone(), discovered, seen_roots)?;
        }
    }

    Ok(())
}

fn add_discovered_mod(
    path: &Path,
    fallback_name: Option<&str>,
    source: ModSource,
    discovered: &mut Vec<Mod>,
    seen_roots: &mut HashSet<PathBuf>,
) -> Result<()> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize mod path {}", path.display()))?;

    if !seen_roots.insert(canonical) {
        return Ok(());
    }

    let fallback = fallback_name
        .map(str::to_owned)
        .unwrap_or_else(|| dir_name(path));
    let manifest = load_manifest(path, &fallback)?;

    info!(
        "[Mods] Found packaged mod: {} from {}",
        manifest.name,
        source_label(&source)
    );

    discovered.push(Mod {
        manifest,
        path: path.to_path_buf(),
        enabled: true,
        source,
    });

    Ok(())
}

#[derive(Debug, thiserror::Error)]
enum ModManifestError {
    #[error("manifest path is a directory: {0}")]
    ManifestIsDirectory(String),

    #[error("failed to read manifest {path}: {source}")]
    Read { path: String, source: io::Error },

    #[error("failed to parse manifest {path}: {source}")]
    Parse {
        path: String,
        source: serde_yaml::Error,
    },

    #[error("invalid mod ID: {0}")]
    InvalidModId(String),
}

fn load_manifest(path: &Path, fallback_name: &str) -> Result<ModManifest, ModManifestError> {
    let manifest_path = path.join(MOD_MANIFEST);

    if manifest_path.is_dir() {
        return Err(ModManifestError::ManifestIsDirectory(
            manifest_path.display().to_string(),
        ));
    }

    let text = fs::read_to_string(&manifest_path).map_err(|source| ModManifestError::Read {
        path: manifest_path.display().to_string(),
        source,
    })?;

    let manifest: ModManifestYaml =
        serde_yaml::from_str(&text).map_err(|source| ModManifestError::Parse {
            path: manifest_path.display().to_string(),
            source,
        })?;

    let name = manifest.name.unwrap_or_else(|| fallback_name.to_string());

    let id = manifest
        .id
        .map(|id| {
            validate_mod_id(&id).map_err(|_| ModManifestError::InvalidModId(id.clone()))?;
            Ok(id)
        })
        .transpose()?
        .unwrap_or_else(|| slugify_id(&name));

    let version = manifest.version.unwrap_or_else(|| "unknown".to_string());

    let manifest = ModManifest {
        id,
        name,
        description: manifest.description.unwrap_or_default(),
        version,
        author: manifest.author.unwrap_or_default(),
        dependencies: manifest.dependencies.unwrap_or_default(),
    };

    Ok(manifest)
}

fn validate_mod_id(id: &str) -> Result<()> {
    if id.is_empty() || id == "." || id == ".." {
        return Err(anyhow!("mod id cannot be empty or a path component"));
    }
    if id.chars().any(|c| c.is_control()) || id.contains('/') || id.contains('\\') {
        return Err(anyhow!("mod id '{}' contains invalid characters", id));
    }
    Ok(())
}

fn slugify_id(name: &str) -> String {
    let mut result = String::new();
    let mut separator = false;

    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            if separator && !result.is_empty() {
                result.push('-');
            }
            result.push(c.to_ascii_lowercase());
            separator = false;
        } else if c == '-' || c == '_' || c.is_whitespace() {
            separator = true;
        }
    }

    if result.is_empty() {
        "unnamed-mod".to_string()
    } else {
        result
    }
}

fn is_mod_root(path: &Path) -> Result<bool> {
    if path.join(MOD_MANIFEST).is_file() {
        return Ok(true);
    }

    for marker in ["ui", "simulation", "sounds", "shaders", "textures"] {
        if path.join(marker).is_dir() {
            return Ok(true);
        }
    }

    Ok(false)
}

fn materialize_archive(source: &Path) -> Result<PathBuf> {
    let cache_root = user_mod_cache_dir();
    fs::create_dir_all(&cache_root)?;

    let fingerprint = archive_fingerprint(source)?;
    let final_dir = cache_root.join(format!("{:016x}", fingerprint));
    let complete_marker = final_dir.join(".complete");

    if complete_marker.is_file() {
        return Ok(final_dir);
    }

    if final_dir.exists() {
        fs::remove_dir_all(&final_dir).with_context(|| {
            format!(
                "failed to remove incomplete archive cache {}",
                final_dir.display()
            )
        })?;
    }

    let temp_dir = cache_root.join(format!(".{:016x}.partial", fingerprint));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;

    let mut budget = ExtractionBudget::new();
    let mut processed = HashSet::new();

    if let Err(e) = extract_archive_tree(source, &temp_dir, 0, &mut budget, &mut processed) {
        let _ = fs::remove_dir_all(&temp_dir);
        return Err(e).with_context(|| format!("failed to extract {}", source.display()));
    }

    fs::write(temp_dir.join(".complete"), b"Rusty Skylines mod cache\n")?;
    fs::rename(&temp_dir, &final_dir).with_context(|| {
        format!(
            "failed to finalize archive cache {} -> {}",
            temp_dir.display(),
            final_dir.display()
        )
    })?;

    Ok(final_dir)
}

fn extract_archive_tree(
    source: &Path,
    destination: &Path,
    depth: usize,
    budget: &mut ExtractionBudget,
    processed: &mut HashSet<PathBuf>,
) -> Result<()> {
    if depth > MAX_NESTED_ARCHIVE_DEPTH {
        return Err(anyhow!(
            "nested archive depth exceeded maximum of {}",
            MAX_NESTED_ARCHIVE_DEPTH
        ));
    }

    budget.start_archive()?;
    extract_one_archive(source, destination, budget)?;
    expand_nested_archives(destination, depth + 1, budget, processed)?;
    Ok(())
}

fn expand_nested_archives(
    root: &Path,
    depth: usize,
    budget: &mut ExtractionBudget,
    processed: &mut HashSet<PathBuf>,
) -> Result<()> {
    if depth > MAX_NESTED_ARCHIVE_DEPTH {
        return Ok(());
    }

    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    files.sort();

    for archive in files {
        if archive.file_name().and_then(|n| n.to_str()) == Some(".complete") {
            continue;
        }
        if archive.starts_with(root.join(".rs_extract_")) {
            continue;
        }
        if !is_archive_candidate(&archive)? {
            continue;
        }

        let canonical = match archive.canonicalize() {
            Ok(path) => path,
            Err(_) => continue,
        };
        if !processed.insert(canonical) {
            continue;
        }

        let extraction_root = nested_extraction_root(&archive);
        if extraction_root.exists() {
            continue;
        }
        fs::create_dir_all(&extraction_root)?;

        if let Err(e) = extract_archive_tree(&archive, &extraction_root, depth, budget, processed) {
            let _ = fs::remove_dir_all(&extraction_root);
            return Err(e).with_context(|| {
                format!("failed to extract nested archive {}", archive.display())
            });
        }
    }

    Ok(())
}

fn nested_extraction_root(archive: &Path) -> PathBuf {
    let hash = xxhash_rust::xxh3::xxh3_64(archive.to_string_lossy().as_bytes());
    archive
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(".rs_extract_{:016x}", hash))
}

fn extract_one_archive(
    source: &Path,
    destination: &Path,
    budget: &mut ExtractionBudget,
) -> Result<()> {
    if special_stream_kind(source).is_ok() {
        return extract_special_stream(source, destination, budget);
    }

    let mut file = File::open(source)
        .with_context(|| format!("failed to open archive {}", source.display()))?;
    let format = ArchiveFormat::detect(&mut file, Some(source))
        .map_err(|e| anyhow!("archive format detection failed: {}", e))?
        .ok_or_else(|| anyhow!("unsupported archive format for {}", source.display()))?;
    file.seek(SeekFrom::Start(0))?;

    let options = ArchiveOptions::new().with_verify_crc(true);
    let mut archive = format
        .open_with_options(file, options)
        .map_err(|e| anyhow!("failed to open {} archive: {}", format.name(), e))?;

    if matches!(
        format,
        ArchiveFormat::Gz | ArchiveFormat::Bz2 | ArchiveFormat::Z
    ) {
        archive.set_single_file_name(decompressed_single_file_name(source));
    }

    while let Some(entry) = archive
        .next_entry()
        .map_err(|e| anyhow!("failed to read archive entry: {}", e))?
    {
        let relative = safe_archive_path(entry.name())?;
        if relative.as_os_str().is_empty() {
            continue;
        }

        let output = destination.join(&relative);
        if entry.is_directory() {
            fs::create_dir_all(&output)?;
            continue;
        }

        budget.reserve(entry.original_size())?;
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut output_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&output)
            .with_context(|| format!("failed to create {}", output.display()))?;

        let mut writer = BudgetWriter::new(&mut output_file, budget);
        if let Err(e) = archive.read_to(&entry, &mut writer) {
            let _ = fs::remove_file(&output);
            return Err(anyhow!("failed to extract {}: {}", entry.name(), e));
        }
        writer.flush()?;
    }

    Ok(())
}

fn extract_special_stream(
    source: &Path,
    destination: &Path,
    budget: &mut ExtractionBudget,
) -> Result<()> {
    let output = destination.join(decompressed_output_path(source));
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    let input = File::open(source)?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&output)?;
    let mut writer = BudgetWriter::new(file, budget);

    match special_stream_kind(source)? {
        SpecialStream::Xz => {
            let mut decoder = xz2::read::XzDecoder::new(input);
            io::copy(&mut decoder, &mut writer)?;
        }
        SpecialStream::Zstd => {
            let mut decoder = zstd::stream::read::Decoder::new(input)?;
            io::copy(&mut decoder, &mut writer)?;
        }
        SpecialStream::Lz4 => {
            let mut decoder = lz4_flex::frame::FrameDecoder::new(input);
            io::copy(&mut decoder, &mut writer)?;
        }
    }

    writer.flush()?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum SpecialStream {
    Xz,
    Zstd,
    Lz4,
}

fn special_stream_kind(path: &Path) -> Result<SpecialStream> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 8];
    let count = file.read(&mut magic)?;

    if count >= 6 && magic[..6] == [0xFD, b'7', b'z', b'X', b'Z', 0x00] {
        return Ok(SpecialStream::Xz);
    }
    if count >= 4 && magic[..4] == [0x28, 0xB5, 0x2F, 0xFD] {
        return Ok(SpecialStream::Zstd);
    }
    if count >= 4 && magic[..4] == [0x04, 0x22, 0x4D, 0x18] {
        return Ok(SpecialStream::Lz4);
    }

    Err(anyhow!("not a supported special compression stream"))
}

fn is_archive_candidate(path: &Path) -> Result<bool> {
    if special_stream_kind(path).is_ok() {
        return Ok(true);
    }
    if ArchiveFormat::from_path(path).is_some() {
        return Ok(true);
    }

    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return Ok(false),
    };

    Ok(ArchiveFormat::detect(&mut file, Some(path))
        .map_err(|e| anyhow!("archive format detection failed: {}", e))?
        .is_some())
}

fn safe_archive_path(raw: &str) -> Result<PathBuf> {
    if raw.contains('\0') {
        return Err(anyhow!("archive entry contains NUL byte"));
    }

    let normalized = raw.replace('\\', "/");
    let mut result = PathBuf::new();

    for component in Path::new(&normalized).components() {
        match component {
            Component::Normal(value) => result.push(value),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(anyhow!(
                    "archive entry escapes extraction directory: {}",
                    raw
                ));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(anyhow!("archive entry has an absolute path: {}", raw));
            }
        }
    }

    Ok(result)
}

fn collect_files(root: &Path, result: &mut Vec<PathBuf>) -> io::Result<()> {
    if !root.is_dir() {
        return Ok(());
    }

    for entry in sorted_dir_entries(root)? {
        let file_type = fs::symlink_metadata(&entry)?.file_type();
        if file_type.is_dir() {
            collect_files(&entry, result)?;
        } else if file_type.is_file() {
            result.push(entry);
        }
    }

    Ok(())
}

fn sorted_dir_entries(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut entries: Vec<_> = fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .collect();
    entries.sort();
    Ok(entries)
}

fn extension_matches(path: &Path, extensions: Option<&[&str]>) -> bool {
    let Some(extensions) = extensions else {
        return true;
    };
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };

    extensions
        .iter()
        .any(|candidate| ext.eq_ignore_ascii_case(candidate))
}

fn normalize_resource_path(path: &Path) -> Option<PathBuf> {
    let mut result = PathBuf::new();

    for component in path.components() {
        match component {
            Component::Normal(value) => result.push(value),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }

    Some(result)
}

fn archive_fingerprint(path: &Path) -> Result<u64> {
    let metadata = fs::metadata(path)?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());

    let mut fingerprint_data = Vec::with_capacity(128);
    fingerprint_data.extend_from_slice(path.to_string_lossy().as_bytes());
    fingerprint_data.push(0);
    fingerprint_data.extend_from_slice(&metadata.len().to_le_bytes());

    if let Some(modified) = modified {
        fingerprint_data.extend_from_slice(&modified.as_secs().to_le_bytes());
        fingerprint_data.extend_from_slice(&modified.subsec_nanos().to_le_bytes());
    }

    let mut file = File::open(path)?;
    let mut sample = [0u8; 64 * 1024];
    let read = file.read(&mut sample)?;
    fingerprint_data.extend_from_slice(&sample[..read]);

    if metadata.len() > sample.len() as u64 {
        file.seek(SeekFrom::End(-(sample.len() as i64)))?;
        let read = file.read(&mut sample)?;
        fingerprint_data.extend_from_slice(&sample[..read]);
    }

    Ok(xxhash_rust::xxh3::xxh3_64(&fingerprint_data))
}

fn archive_stem(path: &Path) -> String {
    let mut name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("mod")
        .to_string();
    let lower = name.to_ascii_lowercase();

    for suffix in [
        ".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tar.lz4", ".7z", ".zip", ".jar", ".rar",
        ".tar", ".tgz", ".tbz2", ".txz", ".tzst", ".tlz4", ".xz", ".zst", ".lz4", ".gz", ".bz2",
    ] {
        if lower.ends_with(suffix) {
            name.truncate(name.len() - suffix.len());
            break;
        }
    }

    name
}

fn decompressed_output_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("decompressed");
    let lower = file_name.to_ascii_lowercase();

    let stripped = if lower.ends_with(".tar.xz") {
        file_name[..file_name.len() - 3].to_string()
    } else if lower.ends_with(".txz") {
        format!("{}.tar", &file_name[..file_name.len() - 4])
    } else if lower.ends_with(".tar.zst") {
        file_name[..file_name.len() - 4].to_string()
    } else if lower.ends_with(".tzst") {
        format!("{}.tar", &file_name[..file_name.len() - 5])
    } else if lower.ends_with(".tar.lz4") {
        file_name[..file_name.len() - 4].to_string()
    } else if lower.ends_with(".tlz4") {
        format!("{}.tar", &file_name[..file_name.len() - 5])
    } else if lower.ends_with(".xz") {
        file_name[..file_name.len() - 3].to_string()
    } else if lower.ends_with(".zst") {
        file_name[..file_name.len() - 4].to_string()
    } else if lower.ends_with(".lz4") {
        file_name[..file_name.len() - 4].to_string()
    } else {
        file_name.to_string()
    };

    PathBuf::from(stripped)
}

fn decompressed_single_file_name(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("decompressed");
    let lower = name.to_ascii_lowercase();

    for suffix in [".tar.gz", ".tar.bz2", ".tar.z", ".gz", ".bz2", ".z"] {
        if lower.ends_with(suffix) {
            return name[..name.len() - suffix.len()].to_string();
        }
    }

    format!("{}.decompressed", name)
}

fn dir_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("mod")
        .to_string()
}

fn source_label(source: &ModSource) -> String {
    match source {
        ModSource::BuiltIn => "built-in".to_string(),
        ModSource::Directory(path) => path.display().to_string(),
        ModSource::Archive(path) => path.display().to_string(),
    }
}

fn load_config() -> ModConfig {
    let path = mods_config_path();

    match fs::read_to_string(&path) {
        Ok(text) => match serde_yaml::from_str::<ModConfig>(&text) {
            Ok(config) => config,
            Err(e) => {
                error!("[Mods] Failed to parse {}: {}", path.display(), e);
                ModConfig::default()
            }
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => ModConfig::default(),
        Err(e) => {
            error!("[Mods] Failed to read {}: {}", path.display(), e);
            ModConfig::default()
        }
    }
}

fn save_config(config: &ModConfig) -> Result<()> {
    let path = mods_config_path();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let text = serde_yaml::to_string(config)?;
    let temp = path.with_extension("yaml.tmp");

    fs::write(&temp, text)?;
    fs::rename(temp, path)?;

    Ok(())
}

fn cleanup_mod_cache(active: &HashSet<String>) -> Result<()> {
    let root = user_mod_cache_dir();
    if !root.is_dir() {
        return Ok(());
    }

    for path in sorted_dir_entries(&root)? {
        if !path.is_dir() {
            continue;
        }

        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        if name.starts_with('.') {
            continue;
        }

        if !active.contains(name) {
            let _ = fs::remove_dir_all(&path);
        }
    }

    Ok(())
}
