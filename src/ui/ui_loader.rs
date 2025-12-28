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
        (lo as i128 + (r % (span as i128 + 1))) as i32
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

/// Top-level loader function.
pub fn load_gui_from_file(path: PathBuf, mode: BendMode) -> Result<GuiLayout, Box<dyn Error>> {
    let bytes = fs::read(&path)?;
    match mode {
        BendMode::Strict => {
            // Normal behavior
            let parsed: GuiLayout = serde_json::from_slice(&bytes)?;
            Ok(parsed)
        }
        BendMode::Bent => {
            // Always synthesize from raw bytes deterministically
            let seed = fnv1a_64(&bytes);
            let mut rng = SimpleRng::new(seed);
            let layout = synthesize_layout_from_bytes(&bytes, &mut rng);
            Ok(layout)
        }
    }
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
        menus.push(MenuJson {
            name: format!("menu_{}_{}", mi, rng.next_u64()),
            layers,
        });
    }

    GuiLayout { menus }
}

fn synth_layer(rng: &mut SimpleRng, menu_idx: usize, layer_idx: usize) -> UiLayerJson {
    // ensure active true always, visible
    let name = format!("layer_{}_{}_{}", menu_idx, layer_idx, rng.next_u64());
    let order = (rng.next_u64() % 100) as i32;

    // counts
    let texts_n = rng.next_usize(3); // 0..2 texts
    let circles_n = rng.next_usize(3); // 0..2 circles
    let outlines_n = rng.next_usize(2); // 0..1 outlines
    let handles_n = rng.next_usize(2); // 0..1 handles
    let polys_n = rng.next_usize(3); // 0..2 polygons

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

    UiLayerJson {
        name,
        order: order as u32,
        texts: if texts.is_empty() { None } else { Some(texts) },
        circles: if circles.is_empty() {
            None
        } else {
            Some(circles)
        },
        outlines: if outlines.is_empty() {
            None
        } else {
            Some(outlines)
        },
        handles: if handles.is_empty() {
            None
        } else {
            Some(handles)
        },
        polygons: if polys.is_empty() { None } else { Some(polys) },
        active: Some(true),
        opaque: Some(rng.next_bool()),
    }
}

fn synth_text(rng: &mut SimpleRng) -> UiButtonTextJson {
    UiButtonTextJson {
        id: Some(format!("t_{}", rng.next_u64())),
        action: "None".to_string(),
        style: "None".to_string(),
        z_index: rng.next_i32_range(0, 200),

        // user requested 0..1e3 and active true always
        x: rng.next_f32_range(0.0, 1_000.0),
        y: rng.next_f32_range(0.0, 1_000.0),
        top_left_offset: [0.0, 0.0],
        bottom_left_offset: [0.0, 0.0],
        top_right_offset: [0.0, 0.0],
        bottom_right_offset: [0.0, 0.0],
        px: rng.next_f32_range(0.0, 0.3),
        color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 1.0),
        ],
        text: synth_text_string(rng),
        misc: MiscButtonSettingsJson {
            active: true,
            pressable: rng.next_bool(),
            editable: rng.next_bool(),
        },
        input_box: rng.next_bool(),
        anchor: None,
    }
}

fn synth_circle(rng: &mut SimpleRng) -> UiButtonCircleJson {
    UiButtonCircleJson {
        id: Some(format!("c_{}", rng.next_u64())),
        action: "None".to_string(),
        style: "Bent".to_string(),
        z_index: rng.next_i32_range(0, 200),
        x: rng.next_f32_range(0.0, 1_000.0),
        y: rng.next_f32_range(0.0, 1_000.0),
        radius: rng.next_f32_range(1.0, 500.0),
        inside_border_thickness: rng.next_f32_range(0.0, 20.0),
        border_thickness: rng.next_f32_range(0.0, 20.0),
        fade: rng.next_f32_range(0.0, 1.0),
        fill_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 1.0),
        ],
        inside_border_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 1.0),
        ],
        border_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 1.0),
        ],
        glow_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 1.0),
        ],
        glow_misc: GlowMisc {
            glow_size: rng.next_f32_range(0.0, 200.0),
            glow_speed: rng.next_f32_range(0.0, 10.0),
            glow_intensity: rng.next_f32_range(0.0, 10.0),
        },
        misc: MiscButtonSettingsJson {
            active: true,
            pressable: rng.next_bool(),
            editable: rng.next_bool(),
        },
    }
}

fn synth_handle(rng: &mut SimpleRng) -> UiButtonHandleJson {
    UiButtonHandleJson {
        id: Some(format!("h_{}", rng.next_u64())),
        z_index: rng.next_i32_range(0, 200),
        x: rng.next_f32_range(0.0, 1_000.0),
        y: rng.next_f32_range(0.0, 1_000.0),
        radius: rng.next_f32_range(1.0, 100.0),
        handle_thickness: rng.next_f32_range(0.5, 20.0),
        handle_color: [
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 2.0),
            rng.next_f32_range(0.0, 1.0),
        ],
        handle_misc: HandleMisc {
            handle_len: rng.next_f32_range(0.0, 20.0),
            handle_width: rng.next_f32_range(0.0, 20.0),
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
            handle_len: rng.next_f32_range(0.0, 20.0),
            handle_width: rng.next_f32_range(0.0, 20.0),
            handle_roundness: rng.next_f32_range(0.0, 2.0),
            handle_speed: rng.next_f32_range(0.0, 2.0),
        },
        misc: MiscButtonSettingsJson {
            active: true,
            pressable: rng.next_bool(),
            editable: rng.next_bool(),
        },
        parent_id: None,
    }
}

fn synth_outline(rng: &mut SimpleRng) -> UiButtonOutlineJson {
    let x = rng.next_f32_range(0.0, 1_000.0);
    let y = rng.next_f32_range(0.0, 1_000.0);
    let r = rng.next_f32_range(1.0, 400.0);
    UiButtonOutlineJson {
        id: Some(format!("o_{}", rng.next_u64())),
        z_index: rng.next_i32_range(0, 200),
        parent_id: None,
        mode: if rng.next_bool() { 0.0 } else { 1.0 },
        shape_data: ShapeData {
            x,
            radius: r,
            y,
            border_thickness: rng.next_f32_range(0.1, 50.0),
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
        misc: MiscButtonSettingsJson {
            active: true,
            pressable: rng.next_bool(),
            editable: rng.next_bool(),
        },
    }
}

fn synth_polygon(rng: &mut SimpleRng) -> UiButtonPolygonJson {
    let verts_n = 3 + rng.next_usize(6); // 3..8 vertices
    let mut verts = Vec::with_capacity(verts_n);
    for _ in 0..verts_n {
        verts.push(UiVertexJson {
            pos: [
                rng.next_f32_range(0.0, 1_000.0),
                rng.next_f32_range(0.0, 1_000.0),
            ],
            color: [
                rng.next_f32_range(0.0, 2.0),
                rng.next_f32_range(0.0, 2.0),
                rng.next_f32_range(0.0, 2.0),
                rng.next_f32_range(0.0, 1.0),
            ],
            roundness: rng.next_f32_range(0.0, 1.0),
        });
    }

    UiButtonPolygonJson {
        id: Some(format!("p_{}", rng.next_u64())),
        action: "None".to_string(),
        style: "BentPoly".to_string(),
        z_index: rng.next_i32_range(0, 200),
        vertices: verts,
        misc: MiscButtonSettingsJson {
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
