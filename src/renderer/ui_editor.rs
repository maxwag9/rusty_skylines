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
        color: [f32; 4],
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
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
    pub fn collect_vertices(&self) -> Vec<Vertex> {
        let mut all_vertices = Vec::new();

        for def in &self.buttons {
            match def {
                UiButtonDef::Polygon {
                    vertices, active, ..
                } if *active => {
                    for v in vertices {
                        all_vertices.push(Vertex {
                            position: v.pos,
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
                        Vertex {
                            position: top_left_vertex.pos,
                            color: top_left_vertex.color,
                        },
                        Vertex {
                            position: top_right_vertex.pos,
                            color: top_right_vertex.color,
                        },
                        Vertex {
                            position: bottom_left_vertex.pos,
                            color: bottom_left_vertex.color,
                        },
                        Vertex {
                            position: bottom_right_vertex.pos,
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
                        Vertex {
                            position: top_vertex.pos,
                            color: top_vertex.color,
                        },
                        Vertex {
                            position: left_vertex.pos,
                            color: left_vertex.color,
                        },
                        Vertex {
                            position: right_vertex.pos,
                            color: right_vertex.color,
                        },
                    ]);
                }

                UiButtonDef::Circle { .. } => {
                    // you'd generate the circle mesh procedurally here
                }

                _ => {}
            }
        }

        all_vertices
    }
}
