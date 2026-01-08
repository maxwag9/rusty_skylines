use crate::data::BendMode;
use crate::ui::vertex::*;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

/// Simple deterministic PRNG based on splitmix64.
struct SimpleRng {
    state: u64,
}
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        let v = (self.next_u64() as f64) / (u64::MAX as f64);
        (lo as f64 + v * ((hi - lo) as f64)) as f32
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() as usize) % bound
        }
    }

    fn next_bool(&mut self) -> bool {
        (self.next_u64() & 1) == 1
    }

    fn next_i32_range(&mut self, lo: i32, hi: i32) -> i32 {
        let r = self.next_u64() as i128;
        let span = (hi - lo) as i128;
        (lo as i128 + (r % (span + 1))) as i32
    }
}

/// FNV-1a 64-bit hash for seeding RNG from raw bytes.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001B3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Load from legacy single-file format
pub fn load_legacy_gui_layout_legacy(path: &PathBuf, mode: &BendMode) -> Vec<MenuYaml> {
    if !path.exists() {
        println!("Legacy file not found: {}", path.display());
        return vec![];
    }

    match load_gui_from_file_legacy(path.clone(), mode) {
        Ok(layout) => {
            println!("Loaded legacy layout with {} menus", layout.menus.len());
            layout.menus
        }
        Err(e) => {
            eprintln!("Failed to load legacy GUI layout: {e}");
            vec![]
        }
    }
}

/// Legacy single-file loader
pub fn load_gui_from_file_legacy(
    path: PathBuf,
    mode: &BendMode,
) -> Result<GuiLayout, Box<dyn Error>> {
    let bytes = fs::read(&path)?;
    match mode {
        BendMode::Strict => {
            let parsed: GuiLayout = serde_yaml::from_slice(&bytes)?;
            Ok(parsed)
        }
        BendMode::Bent => {
            let seed = fnv1a_64(&bytes);
            let mut rng = SimpleRng::new(seed);
            let layout = synthesize_layout_from_bytes(&bytes, &mut rng);
            Ok(layout)
        }
    }
}
pub fn load_menus_from_directory(
    menus_dir: &PathBuf,
    mode: &BendMode,
) -> Result<Vec<MenuYaml>, Box<dyn Error>> {
    let mut menus = Vec::new();

    if !menus_dir.is_dir() {
        println!("Menus directory not found: {}", menus_dir.display());
        return Ok(menus);
    }

    for entry in fs::read_dir(menus_dir)? {
        let entry = entry?;
        let path = entry.path();

        let is_yaml = path
            .extension()
            .map_or(false, |e| e == "yaml" || e == "yml" || e == "Yaml");

        if is_yaml && path.is_file() {
            match load_menu_from_file(&path, mode) {
                Ok(menu) => {
                    //println!("Loaded menu: {}", menu.name);
                    menus.push(menu);
                }
                Err(e) => {
                    eprintln!("Failed to load {}: {}", path.display(), e);
                }
            }
        }
    }

    println!(
        "📂 Loaded {} menus from {}",
        menus.len(),
        menus_dir.display()
    );
    Ok(menus)
}

pub fn load_menu_from_file(path: &PathBuf, mode: &BendMode) -> Result<MenuYaml, Box<dyn Error>> {
    let bytes = fs::read(path)?;

    match mode {
        BendMode::Strict => {
            let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("yaml");
            let parsed: MenuYaml = match extension {
                "Yaml" => serde_yaml::from_slice(&bytes)?,
                _ => serde_yaml::from_slice(&bytes)?,
            };
            Ok(parsed)
        }
        BendMode::Bent => {
            let seed = fnv1a_64(&bytes);
            let mut rng = SimpleRng::new(seed);
            let menu = synthesize_menu_from_bytes(&bytes, &mut rng, path);
            Ok(menu)
        }
    }
}

pub fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
            c if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' => c,
            _ => '_',
        })
        .collect()
}

fn synthesize_layout_from_bytes(bytes: &[u8], rng: &mut SimpleRng) -> GuiLayout {
    // mix bytes into rng for more entropy
    for chunk in bytes.chunks(8) {
        let mut v: u64 = 0;
        for (i, &b) in chunk.iter().enumerate().take(8) {
            v |= (b as u64) << (i * 8);
        }
        rng.state = rng.state.wrapping_add(v);
        let _ = rng.next_u64();
    }

    let menus_n = 1 + (rng.next_u64() % 3) as usize; // 1..3 menus
    let mut menus = Vec::with_capacity(menus_n);
    for mi in 0..menus_n {
        let layers_n = 1 + (rng.next_u64() % 4) as usize; // 1..4 layers
        let mut layers = Vec::with_capacity(layers_n);
        for li in 0..layers_n {
            let layer = synth_layer(rng, mi, li);
            layers.push(layer);
        }
        menus.push(MenuYaml {
            name: format!("menu_{}_{}", mi, rng.next_u64()),
            layers,
        });
    }

    GuiLayout { menus }
}
fn synthesize_menu_from_bytes(bytes: &[u8], rng: &mut SimpleRng, path: &PathBuf) -> MenuYaml {
    // Mix bytes into rng for more entropy
    for chunk in bytes.chunks(8) {
        let mut v: u64 = 0;
        for (i, &b) in chunk.iter().enumerate().take(8) {
            v |= (b as u64) << (i * 8);
        }
        rng.state = rng.state.wrapping_add(v);
        let _ = rng.next_u64();
    }

    // Derive menu name from filename
    let menu_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("menu_{}", rng.next_u64()));

    let layers_n = 1 + (rng.next_u64() % 4) as usize;
    let mut layers = Vec::with_capacity(layers_n);

    for li in 0..layers_n {
        let layer = synth_layer(rng, 0, li);
        layers.push(layer);
    }

    MenuYaml {
        name: menu_name,
        layers,
    }
}

fn synth_layer(rng: &mut SimpleRng, menu_idx: usize, layer_idx: usize) -> UiLayerYaml {
    // ensure active true always, visible
    let name = format!("layer_{}_{}_{}", menu_idx, layer_idx, rng.next_u64());
    let order = (rng.next_u64() % 100) as i32;

    // counts
    let texts_n = rng.next_usize(4); // 0..2 texts
    let circles_n = rng.next_usize(4); // 0..2 circles
    let outlines_n = rng.next_usize(3); // 0..1 outlines
    let handles_n = rng.next_usize(3); // 0..1 handles
    let polys_n = rng.next_usize(4); // 0..2 polygons

    let mut texts = Vec::with_capacity(texts_n);
    for _ in 0..texts_n {
        texts.push(synth_text(rng));
    }

    let mut circles = Vec::with_capacity(circles_n);
    for _ in 0..circles_n {
        circles.push(synth_circle(rng));
    }

    let mut outlines = Vec::with_capacity(outlines_n);
    for _ in 0..outlines_n {
        outlines.push(synth_outline(rng));
    }

    let mut handles = Vec::with_capacity(handles_n);
    for _ in 0..handles_n {
        handles.push(synth_handle(rng));
    }

    let mut polys = Vec::with_capacity(polys_n);
    for _ in 0..polys_n {
        polys.push(synth_polygon(rng));
    }

    let mut elements = Vec::new();
    elements.extend(texts.into_iter().map(UiElementYaml::Text));
    elements.extend(circles.into_iter().map(UiElementYaml::Circle));
    elements.extend(outlines.into_iter().map(UiElementYaml::Outline));
    elements.extend(handles.into_iter().map(UiElementYaml::Handle));
    elements.extend(polys.into_iter().map(UiElementYaml::Polygon));

    UiLayerYaml {
        name,
        order: order as u32,
        elements: Some(elements),
        active: true,
        opaque: rng.next_bool(),
    }
}

fn synth_text(rng: &mut SimpleRng) -> UiButtonTextYaml {
    UiButtonTextYaml {
        id: format!("t_{}", rng.next_u64()),
        action: "None".to_string(),
        style: "None".to_string(),
        x: rng.next_f32_range(0.0, 1.0),
        y: rng.next_f32_range(0.0, 1.0),
        top_left_offset: [0.0, 0.0],
        bottom_left_offset: [0.0, 0.0],
        top_right_offset: [0.0, 0.0],
        bottom_right_offset: [0.0, 0.0],
        px: rng.next_f32_range(0.0, 0.1),
        color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 0.9),
        ],
        text: synth_text_string(rng),
        misc: MiscButtonSettingsYaml {
            active: true,
            pressable: rng.next_bool(),
            editable: rng.next_bool(),
        },
        input_box: rng.next_bool(),
        anchor: None,
    }
}

fn synth_circle(rng: &mut SimpleRng) -> UiButtonCircleYaml {
    UiButtonCircleYaml {
        id: format!("c_{}", rng.next_u64()),
        action: "None".to_string(),
        style: "Bent".to_string(),
        x: rng.next_f32_range(0.0, 1.0),
        y: rng.next_f32_range(0.0, 1.0),
        radius: rng.next_f32_range(0.0, 0.3),
        inside_border_thickness_percentage: rng.next_f32_range(0.0, 0.3),
        border_thickness_percentage: rng.next_f32_range(0.0, 0.3),
        fade: rng.next_f32_range(0.0, 1.0),
        fill_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 0.9),
        ],
        inside_border_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 0.9),
        ],
        border_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 0.9),
        ],
        glow_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 0.9),
        ],
        glow_misc: GlowMisc {
            glow_size: rng.next_f32_range(0.0, 0.3),
            glow_speed: rng.next_f32_range(0.0, 10.0),
            glow_intensity: rng.next_f32_range(0.0, 10.0),
        },
        misc: MiscButtonSettingsYaml {
            active: true,
            pressable: rng.next_bool(),
            editable: rng.next_bool(),
        },
    }
}

fn synth_handle(rng: &mut SimpleRng) -> UiButtonHandleYaml {
    UiButtonHandleYaml {
        id: format!("h_{}", rng.next_u64()),
        x: rng.next_f32_range(0.0, 1.0),
        y: rng.next_f32_range(0.0, 1.0),
        radius: rng.next_f32_range(0.0, 0.3),
        handle_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 0.9),
        ],
        handle_misc: HandleMisc {
            handle_len: rng.next_f32_range(0.0, 0.1),
            handle_width: rng.next_f32_range(0.0, 0.1),
            handle_roundness: rng.next_f32_range(0.0, 2.0),
            handle_speed: rng.next_f32_range(0.0, 2.0),
        },
        sub_handle_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 1.0),
        ],
        sub_handle_misc: HandleMisc {
            handle_len: rng.next_f32_range(0.0, 0.1),
            handle_width: rng.next_f32_range(0.0, 0.1),
            handle_roundness: rng.next_f32_range(0.0, 2.0),
            handle_speed: rng.next_f32_range(0.0, 2.0),
        },
        misc: MiscButtonSettingsYaml {
            active: true,
            pressable: rng.next_bool(),
            editable: rng.next_bool(),
        },
        parent: None,
    }
}

fn synth_outline(rng: &mut SimpleRng) -> UiButtonOutlineYaml {
    let x = rng.next_f32_range(0.0, 1.0);
    let y = rng.next_f32_range(0.0, 1.0);
    let r = rng.next_f32_range(0.0, 0.3);
    UiButtonOutlineYaml {
        id: format!("o_{}", rng.next_u64()),
        parent: None,
        mode: if rng.next_bool() { 0.0 } else { 1.0 },
        shape_data: ShapeData {
            x,
            radius: r,
            y,
            border_thickness: rng.next_f32_range(0.0, 1.0),
        },
        dash_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 1.0),
        ],
        dash_misc: DashMisc {
            dash_len: rng.next_f32_range(0.0, 10.0),
            dash_spacing: rng.next_f32_range(0.0, 10.0),
            dash_roundness: rng.next_f32_range(0.0, 3.0),
            dash_speed: rng.next_f32_range(0.0, 10.0),
        },
        sub_dash_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 1.0),
        ],
        sub_dash_misc: DashMisc {
            dash_len: rng.next_f32_range(0.0, 10.0),
            dash_spacing: rng.next_f32_range(0.0, 10.0),
            dash_roundness: rng.next_f32_range(0.0, 3.0),
            dash_speed: rng.next_f32_range(0.0, 10.0),
        },
        misc: MiscButtonSettingsYaml {
            active: true,
            pressable: rng.next_bool(),
            editable: rng.next_bool(),
        },
    }
}

fn synth_polygon(rng: &mut SimpleRng) -> UiButtonPolygonYaml {
    let verts_n = 3 + rng.next_usize(6); // 3..8 vertices
    let mut verts = Vec::with_capacity(verts_n);
    for _ in 0..verts_n {
        verts.push(UiVertexYaml {
            pos: [rng.next_f32_range(0.0, 1.0), rng.next_f32_range(0.0, 1.0)],
            color: [
                rng.next_f32_range(0.0, 2.0),
                rng.next_f32_range(0.0, 2.0),
                rng.next_f32_range(0.0, 2.0),
                rng.next_f32_range(0.0, 0.9),
            ],
            roundness: rng.next_f32_range(0.0, 1.0),
        });
    }

    UiButtonPolygonYaml {
        id: format!("p_{}", rng.next_u64()),
        action: "None".to_string(),
        style: "BentPoly".to_string(),
        vertices: verts,
        misc: MiscButtonSettingsYaml {
            active: true,
            pressable: rng.next_bool(),
            editable: rng.next_bool(),
        },
    }
}

fn synth_text_string(rng: &mut SimpleRng) -> String {
    let words = [
        "glitch", "void", "pulse", "warp", "error", "flux", "zz", "α", "β", "µ",
    ];
    let a = words[rng.next_usize(words.len())];
    let b = words[rng.next_usize(words.len())];
    format!("{}-{}-{:x}", a, b, rng.next_u64())
}
