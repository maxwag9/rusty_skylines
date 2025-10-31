use crate::renderer::ui::CircleParams;
use crate::resources::MouseState;
use crate::vertex::{
    GuiLayout, UiButtonCircle, UiButtonPolygon, UiButtonRectangle, UiButtonText, UiButtonTriangle,
    UiVertex, UiVertexPoly,
};
use std::collections::HashMap;
use std::fs;
use std::io::Result;
use std::path::PathBuf;

pub struct UiButtonLoader {
    pub texts: Vec<UiButtonText>,
    pub circles: Vec<UiButtonCircle>,
    pub rectangles: Vec<UiButtonRectangle>,
    pub triangles: Vec<UiButtonTriangle>,
    pub polygons: Vec<UiButtonPolygon>,
    pub ui_runtime: UiRuntime,
    pub id_lookup: HashMap<u32, String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ButtonRuntime {
    pub touched_time: f32,
    pub is_down: bool,
    pub just_pressed: bool,
    pub just_released: bool,
}

pub struct UiRuntime {
    pub elements: HashMap<String, ButtonRuntime>,
}

impl UiRuntime {
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
        }
    }

    pub fn update_touch(&mut self, id: &str, touched_now: bool, dt: f32) {
        let entry = self
            .elements
            .entry(id.to_string())
            .or_insert_with(ButtonRuntime::default);

        entry.just_pressed = false;
        entry.just_released = false;

        match (entry.is_down, touched_now) {
            // pressed this frame
            (false, true) => {
                entry.is_down = true;
                entry.just_pressed = true;
                entry.touched_time = 0.0;
                println!("{dt}Pressed {}", id);
            }

            // still holding
            (true, true) => {
                entry.touched_time += dt;
            }

            // released this frame
            (true, false) => {
                entry.is_down = false;
                entry.just_released = true;
                println!("Released {}", id);
            }

            // idle
            (false, false) => {
                entry.touched_time = 0.0;
            }
        }
    }

    pub fn get(&self, id: &str) -> ButtonRuntime {
        *self.elements.get(id).unwrap_or(&ButtonRuntime::default())
    }
}

impl UiButtonLoader {
    pub fn new() -> Self {
        match Self::load_gui_from_file("ui_data/gui_layout.json") {
            Ok(layout) => {
                println!(
                    "✅ Loaded GUI layout with {} texts, {} circles, {} rectangles {} triangles, {} polygons",
                    layout.texts.len(),
                    layout.circles.len(),
                    layout.rectangles.len(),
                    layout.triangles.len(),
                    layout.polygons.len(),
                );
                Self {
                    texts: layout.texts,
                    circles: layout.circles,
                    rectangles: layout.rectangles,
                    triangles: layout.triangles,
                    polygons: layout.polygons,
                    ui_runtime: UiRuntime::new(),
                    id_lookup: Default::default(),
                }
            }
            Err(e) => {
                eprintln!("❌ Failed to load GUI layout: {e}");
                Self {
                    texts: Vec::new(),
                    circles: Vec::new(),
                    rectangles: Vec::new(),
                    triangles: Vec::new(),
                    polygons: Vec::new(),
                    ui_runtime: UiRuntime::new(),
                    id_lookup: Default::default(),
                }
            }
        }
    }

    pub fn load_gui_from_file(path: &str) -> Result<GuiLayout> {
        let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        full_path.push("src/renderer");
        full_path.push(path);

        let data = fs::read_to_string(&full_path)?;
        let parsed: GuiLayout = serde_json::from_str(&data)?;
        Ok(parsed)
    }

    pub fn collect_texts(&self) -> Vec<UiButtonText> {
        self.texts
            .iter()
            .map(|t| UiButtonText {
                id: t.id.clone(),
                x: t.x,
                y: t.y,
                stretch_x: 0.0,
                stretch_y: 0.0,
                top_left_vertex: UiVertex {
                    pos: [0.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                    roundness: 0.0,
                },
                bottom_left_vertex: UiVertex {
                    pos: [0.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                    roundness: 0.0,
                },
                top_right_vertex: UiVertex {
                    pos: [0.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                    roundness: 0.0,
                },
                bottom_right_vertex: UiVertex {
                    pos: [0.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                    roundness: 0.0,
                },
                px: t.px,
                color: t.color,
                text: t.text.clone(),
                active: false,
            })
            .collect()
    }

    pub fn collect_rectangles(&self) -> Vec<UiVertexPoly> {
        let mut verts = Vec::new();
        for r in &self.rectangles {
            verts.extend([
                UiVertexPoly {
                    pos: r.top_left_vertex.pos,
                    color: r.top_left_vertex.color,
                },
                UiVertexPoly {
                    pos: r.top_right_vertex.pos,
                    color: r.top_right_vertex.color,
                },
                UiVertexPoly {
                    pos: r.bottom_left_vertex.pos,
                    color: r.bottom_left_vertex.color,
                },
                UiVertexPoly {
                    pos: r.bottom_right_vertex.pos,
                    color: r.bottom_right_vertex.color,
                },
            ]);
        }
        verts
    }

    pub fn collect_circles(&mut self) -> Vec<CircleParams> {
        self.circles
            .iter()
            .map(|c| {
                // `c.id` is Option<String>, so get &str
                let id_str = c.id.as_deref().unwrap_or("");
                let id_hash = if !id_str.is_empty() {
                    UiButtonLoader::hash_id(id_str)
                } else {
                    f32::MAX
                };

                // keep an owned clone for lookup table (safe clone, small strings)
                if let Some(id_owned) = &c.id {
                    self.id_lookup.insert(id_hash as u32, id_owned.clone());
                }

                // runtime lookup
                let r = self
                    .ui_runtime
                    .elements
                    .get(id_str)
                    .copied()
                    .unwrap_or_default();

                CircleParams {
                    center_radius_border: [c.x, c.y, c.radius, 6.0],
                    fill_color: c.fill_color,
                    border_color: c.border_color,
                    glow_color: c.glow_color,
                    glow_misc: [
                        c.glow_misc.glow_size,
                        c.glow_misc.glow_speed,
                        c.glow_misc.glow_intensity,
                        1.0,
                    ],
                    misc: [
                        f32::from(c.misc.active),
                        r.touched_time,
                        f32::from(r.is_down),
                        id_hash,
                    ],
                }
            })
            .collect()
    }

    pub fn handle_touches(&mut self, mouse: &MouseState, dt: f32) {
        // === Circles ===

        for c in &self.circles {
            if !c.misc.active {
                continue;
            }
            let dx = mouse.pos.x - c.x;
            let dy = mouse.pos.y - c.y;
            let dist = (dx * dx + dy * dy).sqrt();
            let touched = dist <= c.radius && mouse.left_pressed;
            if let Some(id) = &c.id {
                self.ui_runtime.update_touch(id, touched, dt);
            }
        }

        // === Rectangles ===
        for r in &self.rectangles {
            if !r.active {
                continue;
            }
            let left = r.x;
            let right = r.x + (r.stretch_x * 100.0);
            let top = r.y;
            let bottom = r.y + (r.stretch_y * 100.0);

            let inside = mouse.pos.x >= left
                && mouse.pos.x <= right
                && mouse.pos.y >= top
                && mouse.pos.y <= bottom;
            if let Some(id) = &r.id {
                self.ui_runtime
                    .update_touch(id, inside && mouse.left_pressed, dt);
            }
        }

        // Add more types as needed (triangles, polygons)
    }
    fn hash_id(id: &str) -> f32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        let hash_u64 = hasher.finish(); // 64-bit
        // map to [0, 1]
        (hash_u64 as f64 / u64::MAX as f64) as f32
    }
}
