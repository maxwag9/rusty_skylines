use crate::renderer::ui::CircleParams;
use crate::vertex::{
    GuiLayout, UiButtonCircle, UiButtonPolygon, UiButtonRectangle, UiButtonText, UiButtonTriangle,
    UiVertex, UiVertexPoly,
};
use std::fs;
use std::io::Result;
use std::path::PathBuf;

pub struct UiButtonLoader {
    pub texts: Vec<UiButtonText>,
    pub circles: Vec<UiButtonCircle>,
    pub rectangles: Vec<UiButtonRectangle>,
    pub triangles: Vec<UiButtonTriangle>,
    pub polygons: Vec<UiButtonPolygon>,
}

impl UiButtonLoader {
    pub fn new() -> Self {
        match Self::load_gui_from_file("ui_data/gui_layout.json") {
            Ok(layout) => {
                println!(
                    "✅ Loaded GUI layout with {} texts, {} circles, {} rectangles",
                    layout.texts.len(),
                    layout.circles.len(),
                    layout.rectangles.len()
                );
                Self {
                    texts: layout.texts,
                    circles: layout.circles,
                    rectangles: layout.rectangles,
                    triangles: layout.triangles,
                    polygons: layout.polygons,
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
            .filter(|t| t.active)
            .map(|t| UiButtonText {
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
            if !r.active {
                continue;
            }
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

    pub fn collect_circles(&self) -> Vec<CircleParams> {
        self.circles
            .iter()
            .map(|c| CircleParams {
                center_radius_border: [c.x, c.y, c.radius, 4.0],
                fill_color: c.fill_color,
                border_color: c.border_color,
                glow_color: c.glow_settings.glow_color,
                glow_misc: [
                    c.glow_settings.glow_size,
                    c.glow_settings.glow_speed,
                    c.glow_settings.glow_intensity,
                    if c.active { 1.0 } else { 0.0 },
                ],
            })
            .collect()
    }

    pub fn collect_text(&self) -> String {
        let text = String::new();
        text
    }
}
