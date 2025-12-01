struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

// CPU: UiVertexPoly { pos, data=[roundness_norm, polygon_index], color, misc }
struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) data: vec2<f32>,   // x = roundness_norm (0..1), y = polygon_index
    @location(2) color: vec4<f32>,
    @location(3) misc: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) roundness_norm: f32,  // 0..1, interpolated per-fragment
    @location(2) vertex_pos: vec2<f32>,// screen-space position in pixels
    @location(3) poly_index: f32,      // polygon index as float
};

struct PolygonInfo {
    edge_offset: u32,
    edge_count: u32,
    _pad0: vec2<u32>,
};

struct PolygonEdge {
    p0: vec2<f32>,
    p1: vec2<f32>,
};

@group(1) @binding(0)
var<storage, read> polygon_infos: array<PolygonInfo>;

@group(1) @binding(1)
var<storage, read> polygon_edges: array<PolygonEdge>;

// distance from p to line segment [a,b]
fn segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ab_len2 = max(dot(ab, ab), 1e-6);
    let t = clamp(dot(p - a, ab) / ab_len2, 0.0, 1.0);
    let closest = a + t * ab;
    return length(p - closest);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // pos is in screen space pixels
    let x = (in.pos.x / screen.size.x) * 2.0 - 1.0;
    let y = 1.0 - (in.pos.y / screen.size.y) * 2.0;

    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.color = in.color;
    out.roundness_norm = in.data.x;  // 0..1
    out.vertex_pos = in.pos;
    out.poly_index = in.data.y;

    return out;
}
fn safe_normalize(v: vec2<f32>) -> vec2<f32> {
    let len2 = dot(v, v);
    if len2 > 1e-12 {
        return v * inverseSqrt(len2);
    }
    return vec2<f32>(0.0, 0.0);
}

fn inward_normal_ccw(edge: vec2<f32>) -> vec2<f32> {
    let n_out = vec2<f32>(edge.y, -edge.x);
    let n = safe_normalize(n_out);
    return -n;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = in.color;

    let r_norm = clamp(in.roundness_norm, 0.0, 1.0);
    if r_norm <= 0.0 {
        // no rounding, but still do AA via edge distance
        // (optional: early return with base_color if you are fine with HW AA)
        let poly_i = u32(in.poly_index + 0.5);
        let info = polygon_infos[poly_i];
        let offset = info.edge_offset;
        let count  = info.edge_count;

        if count < 3u {
            return base_color;
        }

        let p = in.vertex_pos;
        var min_d_edge = 1e9;

        for (var i: u32 = 0u; i < count; i = i + 1u) {
            let edge = polygon_edges[offset + i];
            let d = segment_distance(p, edge.p0, edge.p1);
            if d < min_d_edge {
                min_d_edge = d;
            }
        }

        let aa = 1.0;
        let d = min_d_edge;
        let coverage = smoothstep(0.0, aa, d);
        if coverage <= 0.0 {
            discard;
        }
        return vec4<f32>(base_color.rgb, base_color.a * coverage);
    }

    let max_radius_px = 24.0;
    var r = r_norm * max_radius_px;

    let poly_i = u32(in.poly_index + 0.5);
    let info = polygon_infos[poly_i];
    let offset = info.edge_offset;
    let count  = info.edge_count;

    if count < 3u {
        return base_color;
    }

    let p = in.vertex_pos;

    // 1. base distance: distance to *polygon edges*
    var min_d_edge = 1e9;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let edge = polygon_edges[offset + i];
        let d = segment_distance(p, edge.p0, edge.p1);
        if d < min_d_edge {
            min_d_edge = d;
        }
    }

    // this will be our "distance to rounded boundary" from the inside
    var d = min_d_edge;

    // 2. inject rounded corners into the distance field
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let prev_index = (i + count - 1u) % count;
        let curr_index = i;
        let next_index = (i + 1u) % count;

        let edge_prev = polygon_edges[offset + prev_index];
        let edge_curr = polygon_edges[offset + curr_index];

        let v_prev = edge_prev.p0;
        let v_curr = edge_curr.p0;
        let v_next = edge_curr.p1;

        // tangent directions along the two incident edges (from vertex outwards)
        let f1 = safe_normalize(v_prev - v_curr);
        let f2 = safe_normalize(v_next - v_curr);

        if dot(f1, f1) == 0.0 || dot(f2, f2) == 0.0 {
            continue;
        }

        let cos_theta = clamp(dot(f1, f2), -0.9999, 0.9999);
        let theta = acos(cos_theta);
        if theta <= 0.01 || theta >= 3.13 {
            continue;
        }

        // edge length constraint for local radius
        let edge_prev_len = length(v_curr - v_prev);
        let edge_next_len = length(v_next - v_curr);
        let max_r_by_len = 0.5 * min(edge_prev_len, edge_next_len);
        let r_local = min(r, max_r_by_len);

        let half_theta = 0.5 * theta;
        let tan_half = tan(half_theta);
        if abs(tan_half) < 1e-4 {
            continue;
        }
        var t_max = r_local / tan_half;
        t_max = min(t_max, max_r_by_len);

        // project into edge directions to detect the corner window
        let rel = p - v_curr;
        let u1 = dot(rel, f1);
        let u2 = dot(rel, f2);

        if u1 < 0.0 || u2 < 0.0 || u1 > t_max || u2 > t_max {
            continue;
        }

        // inward normals
        let e_prev = safe_normalize(v_curr - v_prev);
        let e_next = safe_normalize(v_next - v_curr);
        let n_prev = inward_normal_ccw(e_prev);
        let n_next = inward_normal_ccw(e_next);

        let cos_phi = clamp(dot(n_prev, n_next), -0.9999, 0.9999);
        let denom = 1.0 + cos_phi;
        if denom < 1e-4 {
            continue;
        }

        let k = r_local / denom;
        let center = v_curr + (n_prev + n_next) * k;

        let dist_to_center = length(p - center);

        // distance from inside to the circular arc boundary:
        // positive inside, zero on the arc, negative outside the arc
        let d_arc = r_local - dist_to_center;

        // union of edge-boundary and arc-boundary:
        // distance to *closest* part of rounded boundary
        d = min(d, d_arc);
    }

    // 3. final coverage from *our* distance field only
    let aa = 1.0; // AA width in pixels
    let coverage = smoothstep(0.0, aa, d);

    if coverage <= 0.0 {
        discard;
    }

    return vec4<f32>(base_color.rgb, base_color.a * coverage);
}
