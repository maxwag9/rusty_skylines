use crate::renderer::ui::CircleParams;
use crate::vertex::UiVertexPoly;
use serde::Deserialize;
use std::fs;
use std::io::Result;
use std::path::PathBuf;

// --- your basic per-vertex data ---
#[derive(Deserialize, Debug, Clone, Copy)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub roundness: f32,
}

#[derive(Deserialize, Debug)]
pub struct GlowSettings {
    pub glow_color: [f32; 4],
    pub glow_size: f32,
    pub glow_speed: f32,
    pub glow_intensity: f32,
}

// --- all possible button shapes ---
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum UiButtonDef {
    #[serde(rename = "circle")]
    Circle {
        x: f32,
        y: f32,
        stretch_x: f32,
        stretch_y: f32,
        radius: f32,
        fill_color: [f32; 4],
        border_color: [f32; 4],
        glow_settings: GlowSettings,
        active: bool,
    },

    #[serde(rename = "rectangle")]
    Rectangle {
        x: f32,
        y: f32,
        stretch_x: f32,
        stretch_y: f32,
        top_left_vertex: UiVertex,
        bottom_left_vertex: UiVertex,
        top_right_vertex: UiVertex,
        bottom_right_vertex: UiVertex,
        active: bool,
    },

    #[serde(rename = "polygon")]
    Polygon {
        vertices: Vec<UiVertex>,
        x: f32,
        y: f32,
        stretch_x: f32,
        stretch_y: f32,
        active: bool,
    },

    #[serde(rename = "triangle")]
    Triangle {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        top_vertex: UiVertex,
        left_vertex: UiVertex,
        right_vertex: UiVertex,
        active: bool,
    },
}

// --- the loader ---
pub struct UiButtonLoader {
    pub buttons: Vec<UiButtonDef>,
}

impl UiButtonLoader {
    pub fn new() -> Self {
        let buttons = Self::load_gui_from_file("ui_data/gui_layout.json").unwrap_or_else(|e| {
            eprintln!("Failed to load GUI file: {e}");
            Vec::new()
        });
        println!("Loaded buttons: {:?}", buttons);
        Self { buttons }
    }

    pub fn load_gui_from_file(path: &str) -> Result<Vec<UiButtonDef>> {
        let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        full_path.push("src/renderer");
        full_path.push(path);
        // println!("readin' GUI layout from: {:?}", full_path);

        let data = fs::read_to_string(&full_path)?;
        let parsed: Vec<UiButtonDef> =
            serde_json::from_str(&data).expect("Invalid JSON in GUI layout");
        Ok(parsed)
    }

    /// Collects all vertices from active buttons for rendering
    pub fn collect_vertices(&self) -> Vec<UiVertexPoly> {
        let mut all_vertices = Vec::new();

        for def in &self.buttons {
            match def {
                UiButtonDef::Polygon {
                    vertices, active, ..
                } if *active => {
                    for v in vertices {
                        all_vertices.push(UiVertexPoly {
                            pos: v.pos,
                            color: v.color,
                        });
                    }
                }

                UiButtonDef::Rectangle {
                    top_left_vertex,
                    top_right_vertex,
                    bottom_left_vertex,
                    bottom_right_vertex,
                    active,
                    ..
                } if *active => {
                    all_vertices.extend([
                        UiVertexPoly {
                            pos: top_left_vertex.pos,
                            color: top_left_vertex.color,
                        },
                        UiVertexPoly {
                            pos: top_right_vertex.pos,
                            color: top_right_vertex.color,
                        },
                        UiVertexPoly {
                            pos: bottom_left_vertex.pos,
                            color: bottom_left_vertex.color,
                        },
                        UiVertexPoly {
                            pos: bottom_right_vertex.pos,
                            color: bottom_right_vertex.color,
                        },
                    ]);
                }

                UiButtonDef::Triangle {
                    top_vertex,
                    left_vertex,
                    right_vertex,
                    active,
                    ..
                } if *active => {
                    all_vertices.extend([
                        UiVertexPoly {
                            pos: top_vertex.pos,
                            color: top_vertex.color,
                        },
                        UiVertexPoly {
                            pos: left_vertex.pos,
                            color: left_vertex.color,
                        },
                        UiVertexPoly {
                            pos: right_vertex.pos,
                            color: right_vertex.color,
                        },
                    ]);
                }

                _ => {}
            }
        }

        all_vertices
    }

    pub fn collect_circles(&self) -> Vec<CircleParams> {
        let mut circles = Vec::new();

        for def in &self.buttons {
            if let UiButtonDef::Circle {
                x,
                y,
                stretch_x: _,
                stretch_y: _,
                radius,
                fill_color,
                border_color,
                glow_settings,
                active,
            } = def
            {
                // Always push a circle, regardless of "active"
                circles.push(CircleParams {
                    center_radius_border: [*x, *y, *radius, 4.0],
                    fill_color: *fill_color,
                    border_color: *border_color,
                    glow_color: glow_settings.glow_color,
                    // encode the "active" state, e.g. in glow_misc.w
                    glow_misc: [
                        glow_settings.glow_size,
                        glow_settings.glow_speed,
                        glow_settings.glow_intensity,
                        if *active { 1.0 } else { 0.0 },
                    ],
                });
            }
        }

        circles
    }

    pub fn collect_text(&self) -> String {
        let text = String::new();
        text
    }
}
