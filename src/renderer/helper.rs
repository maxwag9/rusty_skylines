use crate::renderer::ui::{CircleParams, TextParams};
use crate::renderer::ui_editor::{LayerCache, RuntimeLayer, UiButtonLoader, UiRuntime};
use crate::vertex::{UiVertex, UiVertexPoly};

pub(crate) fn is_point_in_quad(point: [f32; 2], verts: &[[f32; 2]; 4]) -> bool {
    // Assumes verts in order: TL, BL, BR, TR (or any cyclic order).
    // Uses cross-product to check if point is on the left of each edge (for convex).
    let mut inside = true;
    for i in 0..4 {
        let j = (i + 1) % 4;
        let edge_start = verts[i];
        let edge_end = verts[j];
        let to_point = [point[0] - edge_start[0], point[1] - edge_start[1]];
        let edge_dir = [edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]];
        let cross = edge_dir[0] * to_point[1] - edge_dir[1] * to_point[0];
        if cross > 0.0 {
            // Adjust sign if winding is clockwise vs counterclockwise.
            inside = false;
            break;
        }
    }
    inside
}

pub(crate) fn is_axis_aligned_rect(verts: &[[f32; 2]; 4]) -> bool {
    // Check if edges are horizontal/vertical.
    verts[0][0] == verts[1][0] &&  // Left edge vertical
        verts[1][1] == verts[2][1] &&  // Bottom horizontal
        verts[2][0] == verts[3][0] &&  // Right vertical
        verts[3][1] == verts[0][1] // Top horizontal
}

pub(crate) fn get_aabb(verts: &[[f32; 2]; 4]) -> (f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for v in verts {
        min_x = min_x.min(v[0]);
        min_y = min_y.min(v[1]);
        max_x = max_x.max(v[0]);
        max_y = max_y.max(v[1]);
    }
    (min_x, min_y, max_x, max_y)
}

pub(crate) fn has_roundness(vertices: &[UiVertex; 4]) -> bool {
    vertices.iter().any(|v| v.roundness > 0.0)
}

pub(crate) fn is_point_in_rounded_rect(
    point: [f32; 2],
    verts: &[[f32; 2]; 4],
    radius: f32,
) -> bool {
    // Assumes uniform radius; for per-vertex, you'd need to check each corner separately.
    // Get inner rect (excluding radii).
    let (min_x, min_y, max_x, max_y) = get_aabb(verts);
    let inner_min_x = min_x + radius;
    let inner_min_y = min_y + radius;
    let inner_max_x = max_x - radius;
    let inner_max_y = max_y - radius;

    if point[0] >= inner_min_x
        && point[0] <= inner_max_x
        && point[1] >= inner_min_y
        && point[1] <= inner_max_y
    {
        return true; // Inside flat area.
    }

    // Check corner circles (quarter-circles).
    let corners = [
        [min_x + radius, min_y + radius], // BL? Wait, assuming TL min if y-down.
        [max_x - radius, min_y + radius],
        [max_x - radius, max_y - radius],
        [min_x + radius, max_y - radius],
    ];
    for center in corners {
        let dx = point[0] - center[0];
        let dy = point[1] - center[1];
        if dx * dx + dy * dy <= radius * radius {
            return true;
        }
    }

    false
}

pub fn rebuild_layer_cache(layer: &mut RuntimeLayer, runtime: &UiRuntime) {
    let l = layer;
    l.cache = LayerCache::default();

    // ------- TEXTS -------
    for t in &l.texts {
        // id is Option<String>
        let id_str = t.id.as_deref().unwrap_or("");
        let rt = runtime.get(id_str); // &str OK
        let hash = if id_str.is_empty() {
            f32::MAX
        } else {
            UiButtonLoader::hash_id(id_str)
        };

        l.cache.texts.push(TextParams {
            pos: [t.x, t.y],
            px: t.px,
            color: t.color,
            id_hash: hash,
            misc: [
                f32::from(t.misc.active),
                rt.touched_time,
                f32::from(rt.is_down),
                hash,
            ],
            text: t.text.clone(),
        });
    }

    // ------- CIRCLES -------
    for c in &l.circles {
        let id_str = c.id.as_deref().unwrap_or("");
        let rt = runtime.get(id_str);
        let hash = if id_str.is_empty() {
            f32::MAX
        } else {
            UiButtonLoader::hash_id(id_str)
        };

        l.cache.circle_params.push(CircleParams {
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
                rt.touched_time,
                f32::from(rt.is_down),
                hash,
            ],
        });
    }

    // Common builder for vertex-emitting shapes
    let push_with_misc = |v: &UiVertex, misc: [f32; 4], out: &mut Vec<UiVertexPoly>| {
        out.push(UiVertexPoly {
            pos: v.pos,
            _pad: [1.0; 2],
            color: v.color,
            misc,
        });
    };

    // ------- RECTANGLES (4 verts) -------
    for r in &l.rectangles {
        let id_str = r.id.as_deref().unwrap_or("");
        let rt = runtime.get(id_str);
        let hash = if id_str.is_empty() {
            f32::MAX
        } else {
            UiButtonLoader::hash_id(id_str)
        };

        let misc = [
            f32::from(r.misc.active),
            rt.touched_time,
            f32::from(rt.is_down),
            hash,
        ];

        push_with_misc(&r.top_left_vertex, misc, &mut l.cache.rect_vertices);
        push_with_misc(&r.top_right_vertex, misc, &mut l.cache.rect_vertices);
        push_with_misc(&r.bottom_left_vertex, misc, &mut l.cache.rect_vertices);
        push_with_misc(&r.bottom_right_vertex, misc, &mut l.cache.rect_vertices);
    }

    // ------- TRIANGLES (3 verts) -------
    for tri in &l.triangles {
        let id_str = tri.id.as_deref().unwrap_or("");
        let rt = runtime.get(id_str);
        let hash = if id_str.is_empty() {
            f32::MAX
        } else {
            UiButtonLoader::hash_id(id_str)
        };

        let misc = [
            f32::from(tri.misc.active),
            rt.touched_time,
            f32::from(rt.is_down),
            hash,
        ];

        push_with_misc(&tri.top_vertex, misc, &mut l.cache.triangle_vertices);
        push_with_misc(&tri.left_vertex, misc, &mut l.cache.triangle_vertices);
        push_with_misc(&tri.right_vertex, misc, &mut l.cache.triangle_vertices);
    }

    // ------- POLYGONS (N verts) -------
    for poly in &l.polygons {
        let id_str = poly.id.as_deref().unwrap_or("");
        let rt = runtime.get(id_str);
        let hash = if id_str.is_empty() {
            f32::MAX
        } else {
            UiButtonLoader::hash_id(id_str)
        };

        let misc = [
            f32::from(poly.misc.active),
            rt.touched_time,
            f32::from(rt.is_down),
            hash,
        ];

        for v in &poly.vertices {
            push_with_misc(v, misc, &mut l.cache.polygon_vertices);
        }
    }

    l.dirty = false;
}
