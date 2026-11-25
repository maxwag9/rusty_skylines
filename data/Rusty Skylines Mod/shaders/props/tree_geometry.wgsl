// Procedural broad-crowned oak (bracketed, stochastic L-system + turtle).
//
// Symbols understood by the turtle:
//   F / G  draw a segment          f  move without drawing
//   + -    yaw  (about up)         & ^  pitch (about right)
//   \ /    roll (about heading)    |    turn around
//   [ ]    push / pop (child branch, one depth level deeper)
//   L      leaf tuft at a branch tip (only if thin or deep enough)
//   A      growth bud. Rewritten by the rules; any A that survives the last
//          iteration is a terminal bud and gets a leaf tuft.
//   X, ... any other symbol is ignored by the turtle.

struct TreeParams {
    iterations: u32,
    axiom_length: u32,
    rule_count: u32,
    command_capacity: u32,
    base_length: f32,
    base_thickness: f32,
    base_angle: f32,
    length_decay: f32,
    thickness_decay: f32,
    seed: u32,
    radial_segments: u32,
    min_branch_thickness: f32,
    leaf_scale: f32,
    leaf_density: f32,
    leaf_thickness_threshold: f32,
    leaf_depth_threshold: u32,
    taper: f32, // radius multiplier per F segment along one branch
    tropism: f32, // bend toward +Y per segment (radians, scaled by |H x Y|)
    jitter: f32, // random bend per segment (radians) - gives gnarled limbs
    roll_angle: f32, // step used by \ and /
    billboard: vec4<u32>,
    axiom: array<vec4<u32>, 4>,
    rules: array<vec4<u32>, 4>,
    productions: array<vec4<u32>, 32>,
}

struct TurtleState {
    position: vec3<f32>,
    direction: vec3<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    thickness: f32,
    segment_length: f32,
    depth: u32,
    // Last vertex ring of the branch currently being drawn. The next segment
    // re-uses it as its start ring, so joints are always watertight.
    ring_start: u32,
    ring_up: vec3<f32>, // ring orientation, carried along without roll twist
    ring_dir: vec3<f32>, // axis of that ring
    v_coord: f32,
    u_repeat: f32,
}

@group(2) @binding(0)
var<uniform> params: TreeParams;
@group(2) @binding(1)
var<storage, read_write> vertex_words: array<u32>;
@group(2) @binding(2)
var<storage, read_write> index_words: array<u32>;
@group(2) @binding(3)
var<storage, read_write> counters: array<u32>;

const MAX_COMMANDS: u32 = 4096u;
const MAX_STACK: u32 = 64u;
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const NO_RING: u32 = 0xFFFFFFFFu;
const VERTEX_STRIDE: u32 = 13u; // PropVertex = 13 u32 words
const ROOT_FLARE: f32 = 1.35; // trunk is this much wider at the ground
const GROUND_FADE_HEIGHT: f32 = 5.0; // below this height limbs are pulled upward harder

// ---------------------------------------------------------------------------
// Random helpers
// ---------------------------------------------------------------------------

fn hash_u32(value: u32) -> u32 {
    var h = value * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn random01(rng: ptr<function, u32>) -> f32 {
    *rng = (*rng) * 1103515245u + 12345u;
    return f32((*rng >> 16u) & 32767u) / 32767.0;
}

fn random_range(rng: ptr<function, u32>, low: f32, high: f32) -> f32 {
    return low + random01(rng) * (high - low);
}

fn random_vec3(rng: ptr<function, u32>, radius: f32) -> vec3<f32> {
    return vec3<f32>(
            random_range(rng, -radius, radius),
            random_range(rng, -radius, radius),
            random_range(rng, -radius, radius),
        );
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    return v / max(length(v), 0.00001);
}

fn rotate_axis(value: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let unit_axis = normalize(axis);
    let sine = sin(angle);
    let cosine = cos(angle);
    return value * cosine + cross(unit_axis, value) * sine
        + unit_axis * dot(unit_axis, value) * (1.0 - cosine);
}

// ---------------------------------------------------------------------------
// Turtle rotations. Each one rotates TWO frame vectors by the SAME angle, so
// the (right, up, direction) frame always stays orthonormal.
// ---------------------------------------------------------------------------

fn turn_yaw(s: ptr<function, TurtleState>, angle: f32) {
    let axis = (*s).up;
    (*s).direction = normalize(rotate_axis((*s).direction, axis, angle));
    (*s).right = normalize(rotate_axis((*s).right, axis, angle));
}

fn turn_pitch(s: ptr<function, TurtleState>, angle: f32) {
    let axis = (*s).right;
    (*s).direction = normalize(rotate_axis((*s).direction, axis, angle));
    (*s).up = normalize(rotate_axis((*s).up, axis, angle));
}

fn turn_roll(s: ptr<function, TurtleState>, angle: f32) {
    let axis = (*s).direction;
    (*s).right = normalize(rotate_axis((*s).right, axis, angle));
    (*s).up = normalize(rotate_axis((*s).up, axis, angle));
}

fn orthonormalize(s: ptr<function, TurtleState>) {
    let h = normalize((*s).direction);
    let u = normalize((*s).up - h * dot((*s).up, h));
    (*s).direction = h;
    (*s).up = u;
    (*s).right = cross(u, h);
}

// Bend the heading toward +Y. Strongest on horizontal limbs, zero on a
// vertical trunk, and much stronger near the ground so nothing droops into it.
fn apply_tropism(s: ptr<function, TurtleState>) {
    let torque = cross((*s).direction, vec3<f32>(0.0, 1.0, 0.0));
    let magnitude = length(torque);
    if magnitude < 0.0001 {
        return;
    }
    // Limbs that already point downward get extra lift (|H x Y| alone is weak for them).
    let droop = max(0.0, -(*s).direction.y);
    let ground_pull = clamp((GROUND_FADE_HEIGHT - (*s).position.y) / GROUND_FADE_HEIGHT, 0.0, 1.0);
    let angle = params.tropism * (magnitude + 2.0 * droop) * (1.0 + 3.0 * ground_pull);
    let axis = torque / magnitude;
    (*s).direction = normalize(rotate_axis((*s).direction, axis, angle));
    (*s).right = normalize(rotate_axis((*s).right, axis, angle));
    (*s).up = normalize(rotate_axis((*s).up, axis, angle));
}

// ---------------------------------------------------------------------------
// Mesh output
// ---------------------------------------------------------------------------

fn vertex_capacity() -> u32 {
    return arrayLength(&vertex_words) / VERTEX_STRIDE;
}

fn write_vertex(
    vertex_index: u32,
    position: vec3<f32>,
    normal: vec3<f32>,
    color: vec4<f32>,
    uv: vec2<f32>,
    texture_id: u32,
) {
    let base = vertex_index * VERTEX_STRIDE;
    vertex_words[base + 0u] = bitcast<u32>(position.x);
    vertex_words[base + 1u] = bitcast<u32>(position.y);
    vertex_words[base + 2u] = bitcast<u32>(position.z);
    vertex_words[base + 3u] = bitcast<u32>(normal.x);
    vertex_words[base + 4u] = bitcast<u32>(normal.y);
    vertex_words[base + 5u] = bitcast<u32>(normal.z);
    vertex_words[base + 6u] = bitcast<u32>(color.x);
    vertex_words[base + 7u] = bitcast<u32>(color.y);
    vertex_words[base + 8u] = bitcast<u32>(color.z);
    vertex_words[base + 9u] = bitcast<u32>(color.w);
    vertex_words[base + 10u] = bitcast<u32>(uv.x);
    vertex_words[base + 11u] = bitcast<u32>(uv.y);
    vertex_words[base + 12u] = texture_id;
}
const BILLBOARD_OUTLINE: array<vec2<f32>, 24> = array<vec2<f32>, 24>(
        vec2<f32>(0.9, 3.3),
        vec2<f32>(2.0, 3.9),
        vec2<f32>(3.1, 4.7),
        vec2<f32>(4.1, 5.8),
        vec2<f32>(4.7, 7.0),
        vec2<f32>(4.1, 8.2),
        vec2<f32>(4.6, 9.3),
        vec2<f32>(4.2, 10.5),
        vec2<f32>(3.3, 11.5),
        vec2<f32>(2.3, 12.2),
        vec2<f32>(1.2, 12.9),
        vec2<f32>(0.2, 13.3),
        vec2<f32>(-0.9, 12.7),
        vec2<f32>(-2.1, 11.8),
        vec2<f32>(-2.5, 10.7),
        vec2<f32>(-3.5, 10.2),
        vec2<f32>(-4.2, 9.2),
        vec2<f32>(-4.8, 7.9),
        vec2<f32>(-4.1, 6.8),
        vec2<f32>(-4.4, 5.8),
        vec2<f32>(-3.8, 4.5),
        vec2<f32>(-2.6, 3.8),
        vec2<f32>(-1.6, 3.2),
        vec2<f32>(-0.4, 3.4),
    );

fn emit_billboard_canopy(z: f32, color: vec4<f32>) {
    let point_count = 24u;
    let vertex_count = point_count + 1u;
    let index_count = point_count * 3u;

    let vertex_base = counters[0];
    let index_base = counters[1];

    if vertex_base + vertex_count > vertex_capacity()
        || index_base + index_count > arrayLength(&index_words) {
        counters[2] = 1u;
        return;
    }

    let center = vec2<f32>(0.0, 8.3);

    write_vertex(
        vertex_base,
        vec3<f32>(center, z),
        vec3<f32>(0.0, 0.0, 1.0),
        color,
        vec2<f32>(0.5, 0.5),
        0u,
    );

    for(var i = 0u; i < point_count; i += 1u) {
        let p = BILLBOARD_OUTLINE[i];

        write_vertex(
            vertex_base + i + 1u,
            vec3<f32>(p, z),
            vec3<f32>(0.0, 0.0, 1.0),
            color,
            vec2<f32>((p.x + 5.0) / 10.0, (p.y - 3.0) / 11.0),
            0u,
        );

        let next = (i + 1u) % point_count;

        index_words[index_base + i * 3u] = vertex_base;
        index_words[index_base + i * 3u + 1u] = vertex_base + i + 1u;
        index_words[index_base + i * 3u + 2u] = vertex_base + next + 1u;
    }

    counters[0] += vertex_count;
    counters[1] += index_count;
}

fn emit_billboard_blob(center: vec2<f32>, radius: vec2<f32>, z: f32, color: vec4<f32>, seed: u32) {
    let segments = 10u;
    let vertex_count = segments + 1u;
    let index_count = segments * 3u;

    let vertex_base = counters[0];
    let index_base = counters[1];

    if vertex_base + vertex_count > vertex_capacity()
        || index_base + index_count > arrayLength(&index_words) {
        counters[2] = 1u;
        return;
    }

    write_vertex(
        vertex_base,
        vec3<f32>(center, z),
        vec3<f32>(0.0, 0.0, 1.0),
        color,
        vec2<f32>(0.5, 0.5),
        0u,
    );

    let phase = f32(seed) * 0.17;

    for(var i = 0u; i < segments; i += 1u) {
        let angle = TWO_PI * f32(i) / f32(segments);
        let variation = 0.90 + 0.10 * sin(f32(i) * 2.4 + phase);

        let p = center + vec2<f32>(cos(angle) * radius.x, sin(angle) * radius.y) * variation;

        let shade = 0.94 + 0.06 * sin(f32(i) * 1.7 + phase);

        write_vertex(
            vertex_base + i + 1u,
            vec3<f32>(p, z),
            vec3<f32>(0.0, 0.0, 1.0),
            vec4<f32>(color.rgb * shade, color.a),
            vec2<f32>(0.5, 0.5),
            0u,
        );

        let next = (i + 1u) % segments;

        index_words[index_base + i * 3u] = vertex_base;
        index_words[index_base + i * 3u + 1u] = vertex_base + i + 1u;
        index_words[index_base + i * 3u + 2u] = vertex_base + next + 1u;
    }

    counters[0] += vertex_count;
    counters[1] += index_count;
}

fn emit_billboard_trunk() {
    let vertex_base = counters[0];
    let index_base = counters[1];

    if vertex_base + 4u > vertex_capacity() || index_base + 6u > arrayLength(&index_words) {
        counters[2] = 1u;
        return;
    }

    var positions: array<vec3<f32>, 4>;
    positions[0] = vec3<f32>(-0.55, -0.3, 0.08);
    positions[1] = vec3<f32>(0.55, -0.3, 0.08);
    positions[2] = vec3<f32>(0.27, 6.4, 0.08);
    positions[3] = vec3<f32>(-0.27, 6.4, 0.08);

    var uvs: array<vec2<f32>, 4>;
    uvs[0] = vec2<f32>(0.0, 0.0);
    uvs[1] = vec2<f32>(1.0, 0.0);
    uvs[2] = vec2<f32>(1.0, 1.0);
    uvs[3] = vec2<f32>(0.0, 1.0);

    for(var i = 0u; i < 4u; i += 1u) {
        write_vertex(
            vertex_base + i,
            positions[i],
            vec3<f32>(0.0, 0.0, 1.0),
            vec4<f32>(1.0),
            uvs[i],
            2u,
        );
    }

    index_words[index_base + 0u] = vertex_base;
    index_words[index_base + 1u] = vertex_base + 1u;
    index_words[index_base + 2u] = vertex_base + 2u;

    index_words[index_base + 3u] = vertex_base;
    index_words[index_base + 4u] = vertex_base + 2u;
    index_words[index_base + 5u] = vertex_base + 3u;

    counters[0] += 4u;
    counters[1] += 6u;
}

fn emit_billboard_tree() {
    emit_billboard_canopy(0.0, vec4<f32>(0.18, 0.32, 0.10, 1.0));

    emit_billboard_blob(
        vec2<f32>(-2.2, 6.7),
        vec2<f32>(1.9, 1.9),
        0.015,
        vec4<f32>(0.23, 0.38, 0.12, 1.0),
        1u,
    );

    emit_billboard_blob(
        vec2<f32>(2.1, 7.0),
        vec2<f32>(2.0, 2.0),
        0.025,
        vec4<f32>(0.22, 0.37, 0.11, 1.0),
        2u,
    );

    emit_billboard_blob(
        vec2<f32>(-1.7, 9.8),
        vec2<f32>(1.9, 1.9),
        0.035,
        vec4<f32>(0.25, 0.41, 0.13, 1.0),
        3u,
    );

    emit_billboard_blob(
        vec2<f32>(1.7, 10.0),
        vec2<f32>(1.9, 1.9),
        0.045,
        vec4<f32>(0.21, 0.36, 0.10, 1.0),
        4u,
    );

    emit_billboard_blob(
        vec2<f32>(0.0, 11.5),
        vec2<f32>(1.6, 1.7),
        0.055,
        vec4<f32>(0.27, 0.43, 0.14, 1.0),
        5u,
    );

    emit_billboard_trunk();
}
fn write_ring(
    first_vertex: u32,
    segments: u32,
    center: vec3<f32>,
    radius: f32,
    basis_up: vec3<f32>,
    basis_right: vec3<f32>,
    u_repeat: f32,
    v: f32,
) {
    let bark_color = vec4<f32>(0.30, 0.20, 0.12, 1.0);
    for(var i = 0u; i <= segments; i += 1u) {
        let fraction = f32(i) / f32(segments);
        let angle = fraction * TWO_PI;
        let radial_normal = basis_up * cos(angle) + basis_right * sin(angle);
        write_vertex(
            first_vertex + i,
            center + radial_normal * radius,
            radial_normal,
            bark_color,
            vec2<f32>(fraction * u_repeat, v),
            2u,
        );
    }
}

// Emit one branch segment. If the branch already has a ring at its current
// position (the end of the previous segment) that ring is shared instead of
// writing a new one, so consecutive segments are stitched together even when
// the branch bends. Only the END ring is new; no gaps can open at a joint.
fn emit_segment(
    state: ptr<function, TurtleState>,
    start_position: vec3<f32>,
    end_position: vec3<f32>,
    start_radius: f32,
    end_radius: f32,
    segments: u32,
) {
    let segment_length = length(end_position - start_position);
    if segment_length <= 0.000001 {
        return;
    }

    let ring_vertices = segments + 1u;
    let needs_start_ring = (*state).ring_start == NO_RING;
    var vertex_count = ring_vertices;
    if needs_start_ring {
        vertex_count += ring_vertices;
    }
    let index_count = segments * 6u;
    let vertex_base = counters[0];
    let index_base = counters[1];

    // vertex_base counts VERTICES, so compare against the vertex capacity
    // (buffer words / 13), not the word count.
    if vertex_base + vertex_count > vertex_capacity()
        || index_base + index_count > arrayLength(&index_words) {
        counters[2] = 1u;
        (*state).ring_start = NO_RING;
        return;
    }

    // Ring orientation: carry the previous ring's "up" over to the new axis
    // (projection = minimal twist). Rolls of the turtle frame must not twist
    // the tube, which is why the turtle's own up vector is not used here.
    let heading = (*state).direction;
    var projected_up = (*state).up;
    if !needs_start_ring {
        projected_up = (*state).ring_up;
    }
    projected_up = projected_up - heading * dot(projected_up, heading);
    if length(projected_up) < 0.01 {
        projected_up = (*state).up;
    }
    let basis_up = normalize(projected_up);
    let basis_right = cross(basis_up, heading);

    var next_vertex = vertex_base;
    var start_ring = (*state).ring_start;
    if needs_start_ring {
        (*state).u_repeat = max(1.0, floor(start_radius * 2.0 + 0.5));
        start_ring = next_vertex;
        write_ring(
            start_ring,
            segments,
            start_position,
            start_radius,
            basis_up,
            basis_right,
            (*state).u_repeat,
            (*state).v_coord,
        );
        next_vertex += ring_vertices;
    }

    let mean_radius = max(0.5 * (start_radius + end_radius), 0.05);
    let end_v = (*state).v_coord + segment_length * (*state).u_repeat / (TWO_PI * mean_radius);
    let end_ring = next_vertex;
    write_ring(
        end_ring,
        segments,
        end_position,
        end_radius,
        basis_up,
        basis_right,
        (*state).u_repeat,
        end_v,
    );

    for(var i = 0u; i < segments; i += 1u) {
        let bl = start_ring + i;
        let br = start_ring + i + 1u;
        let tl = end_ring + i;
        let tr = end_ring + i + 1u;
        let output_index = index_base + i * 6u;
        index_words[output_index + 0u] = bl;
        index_words[output_index + 1u] = tl;
        index_words[output_index + 2u] = br;
        index_words[output_index + 3u] = br;
        index_words[output_index + 4u] = tl;
        index_words[output_index + 5u] = tr;
    }

    counters[0] = vertex_base + vertex_count;
    counters[1] = index_base + index_count;
    (*state).ring_start = end_ring;
    (*state).ring_up = basis_up;
    (*state).ring_dir = heading;
    (*state).v_coord = end_v;
}

// Close the open end of a branch with a low rounded dome (one extra vertex).
fn emit_cap(state: ptr<function, TurtleState>, segments: u32) {
    let ring = (*state).ring_start;
    if ring == NO_RING {
        return;
    }
    let vertex_base = counters[0];
    let index_base = counters[1];
    let index_count = segments * 3u;
    if vertex_base + 1u > vertex_capacity()
        || index_base + index_count > arrayLength(&index_words) {
        counters[2] = 1u;
        return;
    }

    let bark_color = vec4<f32>(0.30, 0.20, 0.12, 1.0);
    let apex = (*state).position + (*state).ring_dir * ((*state).thickness * 0.7);
    write_vertex(
        vertex_base,
        apex,
        (*state).ring_dir,
        bark_color,
        vec2<f32>(0.5 * (*state).u_repeat, (*state).v_coord),
        2u,
    );
    for(var i = 0u; i < segments; i += 1u) {
        let output_index = index_base + i * 3u;
        index_words[output_index + 0u] = ring + i + 1u;
        index_words[output_index + 1u] = ring + i;
        index_words[output_index + 2u] = vertex_base;
    }
    counters[0] = vertex_base + 1u;
    counters[1] = index_base + index_count;
    (*state).ring_start = NO_RING;
}

// A rounded tuft of 4-5 crossed, double-sided leaf cards. `size` is the
// approximate tuft diameter in world units. Normals are blended toward the
// tuft's outward direction so the canopy shades like soft clumps instead of
// flat cards.
fn emit_leaf_cluster(position: vec3<f32>, direction: vec3<f32>, size: f32, seed: u32) {
    var rng = hash_u32(seed);
    let leaf_color = vec4<f32>(
            0.30 + random_range(&rng, -0.05, 0.05),
            0.50 + random_range(&rng, -0.08, 0.08),
            0.22 + random_range(&rng, -0.04, 0.04),
            1.0,
        );
    let cluster_center = position + direction * (size * 0.35);
    let plane_count = 4u + min(u32(random01(&rng) * 2.0), 1u);

    for(var plane_index = 0u; plane_index < plane_count; plane_index += 1u) {
        let plane_normal =
            safe_normalize(random_vec3(&rng, 1.0) + vec3<f32>(0.0001, 0.0002, 0.0003));
        var reference_up = vec3<f32>(0.0, 1.0, 0.0);
        if abs(plane_normal.y) > 0.95 {
            reference_up = vec3<f32>(1.0, 0.0, 0.0);
        }
        let base_up = safe_normalize(reference_up - plane_normal * dot(reference_up, plane_normal));
        let base_right = cross(base_up, plane_normal);
        let spin = random_range(&rng, -0.9, 0.9);
        let plane_right = base_right * cos(spin) + base_up * sin(spin);
        let plane_up = base_up * cos(spin) - base_right * sin(spin);

        let half_width = size * random_range(&rng, 0.40, 0.58);
        let half_height = size * random_range(&rng, 0.40, 0.58);
        let plane_center = cluster_center + random_vec3(&rng, size * 0.30);

        var corners: array<vec3<f32>, 4>;
        corners[0] = plane_center - plane_right * half_width - plane_up * half_height;
        corners[1] = plane_center + plane_right * half_width - plane_up * half_height;
        corners[2] = plane_center + plane_right * half_width + plane_up * half_height;
        corners[3] = plane_center - plane_right * half_width + plane_up * half_height;

        var uvs: array<vec2<f32>, 4>;
        uvs[0] = vec2<f32>(0.0, 1.0);
        uvs[1] = vec2<f32>(1.0, 1.0);
        uvs[2] = vec2<f32>(1.0, 0.0);
        uvs[3] = vec2<f32>(0.0, 0.0);

        var face_normal = plane_normal;
        if dot(face_normal, plane_center - cluster_center) < 0.0 {
            face_normal = -face_normal;
        }

        let vertex_base = counters[0];
        let index_base = counters[1];
        if vertex_base + 4u > vertex_capacity() || index_base + 12u > arrayLength(&index_words) {
            counters[2] = 1u;
            return;
        }

        for(var i = 0u; i < 4u; i += 1u) {
            let outward = safe_normalize(corners[i] - cluster_center);
            let leaf_normal =
                safe_normalize(mix(face_normal, outward, 0.65) + vec3<f32>(0.0, 0.15, 0.0));
            write_vertex(vertex_base + i, corners[i], leaf_normal, leaf_color, uvs[i], 1u);
        }

        // Front and back faces, so the card is visible from both sides.
        index_words[index_base + 0u] = vertex_base;
        index_words[index_base + 1u] = vertex_base + 1u;
        index_words[index_base + 2u] = vertex_base + 2u;
        index_words[index_base + 3u] = vertex_base;
        index_words[index_base + 4u] = vertex_base + 2u;
        index_words[index_base + 5u] = vertex_base + 3u;
        index_words[index_base + 6u] = vertex_base + 2u;
        index_words[index_base + 7u] = vertex_base + 1u;
        index_words[index_base + 8u] = vertex_base;
        index_words[index_base + 9u] = vertex_base + 3u;
        index_words[index_base + 10u] = vertex_base + 2u;
        index_words[index_base + 11u] = vertex_base;

        counters[0] = vertex_base + 4u;
        counters[1] = index_base + 12u;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id != vec3<u32>(0u)) {
        return;
    }

    if arrayLength(&counters) < 4u {
        return;
    }

    counters[0] = 0u;
    counters[1] = 0u;
    counters[2] = 0u;
    counters[3] = 0u;
    if params.billboard.x != 0u {
        emit_billboard_tree();
        counters[3] = 1u;
        return;
    }
    if params.axiom_length == 0u || params.axiom_length > 16u || params.rule_count > 4u
        || params.command_capacity == 0u || params.base_length <= 0.0
        || params.base_thickness <= 0.0 {
        counters[2] = 1u;
        return;
    }

    // ----- 1. Expand the L-system -------------------------------------------
    var commands: array<u32, MAX_COMMANDS>;
    var command_count = params.axiom_length;

    for(var i = 0u; i < params.axiom_length; i += 1u) {
        commands[i] = params.axiom[i / 4u][i % 4u];
    }

    let effective_capacity = min(params.command_capacity, MAX_COMMANDS);
    if command_count > effective_capacity {
        counters[2] = 1u;
        return;
    }

    var overflow = false;
    for(var iteration = 0u; iteration < params.iterations; iteration += 1u) {
        var scan = command_count;
        loop {
            if scan == 0u {
                break;
            }
            scan -= 1u;

            let symbol = commands[scan];
            var found_rule = false;
            var production_offset = 0u;
            var production_length = 0u;

            for(var rule_index = 0u; rule_index < params.rule_count; rule_index += 1u) {
                let rule = params.rules[rule_index];
                if rule.x == symbol {
                    production_offset = rule.y;
                    production_length = rule.z;
                    found_rule = true;
                    break;
                }
            }

            if found_rule && production_length > 0u {
                if production_offset + production_length > 128u
                    || command_count + production_length - 1u > effective_capacity {
                    // Out of command space: flag it, but still draw what we have
                    // instead of returning an empty mesh.
                    counters[2] = 1u;
                    overflow = true;
                    break;
                }

                let shift_amount = production_length - 1u;
                var move_index = command_count;
                loop {
                    if move_index <= scan + 1u {
                        break;
                    }
                    move_index -= 1u;
                    commands[move_index + shift_amount] = commands[move_index];
                }

                for(
                    var production_index = 0u;
                    production_index < production_length;
                    production_index += 1u
                ) {
                    let source_index = production_offset + production_index;
                    commands[scan + production_index] = params.productions[source_index / 4u][
                            source_index % 4u
                        ];
                }

                command_count += shift_amount;
            }
        }
        if overflow {
            break;
        }
    }

    counters[3] = command_count;

    // ----- 2. Interpret with the turtle -------------------------------------
    let segments = clamp(params.radial_segments, 3u, 12u);

    var state = TurtleState(
            vec3<f32>(0.0, -0.3, 0.0), // start slightly underground
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 0.0, -1.0),
            params.base_thickness * ROOT_FLARE,
            params.base_length,
            0u,
            NO_RING,
            vec3<f32>(0.0, 0.0, -1.0),
            vec3<f32>(0.0, 1.0, 0.0),
            0.0,
            1.0,
        );

    var turtle_stack: array<TurtleState, 64>;
    var stack_size = 0u;
    var rng = hash_u32(params.seed + 1u);
    var leaf_index = 0u;

    // Random azimuth for the whole tree, so different seeds don't all put
    // their first limb on the same side.
    turn_roll(&state, random_range(&rng, 0.0, TWO_PI));

    for(var command_index = 0u; command_index < command_count; command_index += 1u) {
        let command = commands[command_index];
        let angle_variation = random_range(&rng, 0.65, 1.35);
        let length_variation = random_range(&rng, 0.8, 1.2);

        if command == 70u || command == 71u {
            // Gnarl the heading a little (less on the trunk), then bend toward the light.
            let wobble = params.jitter * select(1.0, 0.4, state.depth == 0u);
            turn_pitch(&state, random_range(&rng, -wobble, wobble));
            turn_yaw(&state, random_range(&rng, -wobble, wobble));
            apply_tropism(&state);
            orthonormalize(&state);

            let start_position = state.position;
            let end_position = start_position
                + state.direction * (state.segment_length * length_variation);
            let start_radius = state.thickness;
            var end_radius = start_radius * params.taper * random_range(&rng, 0.97, 1.03);
            if state.depth == 0u && state.ring_start == NO_RING {
                // First trunk segment: flares from the ground up to base_thickness.
                end_radius = params.base_thickness * params.taper;
            }
            end_radius = max(end_radius, 0.004);

            if start_radius >= params.min_branch_thickness {
                emit_segment(
                    &state,
                    start_position,
                    end_position,
                    start_radius,
                    end_radius,
                    segments,
                );
            } else {
                emit_cap(&state, segments);
            }

            state.position = end_position;
            state.thickness = end_radius;
        } else if command == 102u {
            state.position += state.direction * (state.segment_length * length_variation);
            state.ring_start = NO_RING;
        } else if command == 43u {
            turn_yaw(&state, params.base_angle * angle_variation);
        } else if command == 45u {
            turn_yaw(&state, -params.base_angle * angle_variation);
        } else if command == 38u {
            turn_pitch(&state, params.base_angle * angle_variation);
        } else if command == 94u {
            turn_pitch(&state, -params.base_angle * angle_variation);
        } else if command == 92u {
            turn_roll(&state, params.roll_angle * random_range(&rng, 0.8, 1.2));
        } else if command == 47u {
            turn_roll(&state, -params.roll_angle * random_range(&rng, 0.8, 1.2));
        } else if command == 124u {
            turn_yaw(&state, PI);
        } else if command == 91u {
            if stack_size < MAX_STACK {
                turtle_stack[stack_size] = state;
                stack_size += 1u;
                state.depth += 1u;
                state.segment_length *= params.length_decay * random_range(&rng, 0.9, 1.1);
                state.thickness *= params.thickness_decay * random_range(&rng, 0.9, 1.1);
                state.ring_start = NO_RING; // a child branch starts its own tube
                state.v_coord = random01(&rng);
            } else {
                counters[2] = 1u;
            }
        } else if command == 93u {
            if stack_size > 0u {
                emit_cap(&state, segments); // close the tip of the branch that ends here
                stack_size -= 1u;
                state = turtle_stack[stack_size];
            }
        } else if command == 65u || command == 76u {
            // L only fires on thin / deep branches; a leftover A (terminal bud) always fires.
            var wants_leaves = true;
            if command == 76u {
                wants_leaves = state.thickness < params.leaf_thickness_threshold
                    || state.depth >= params.leaf_depth_threshold;
            }
            if wants_leaves {
                if random01(&rng) < params.leaf_density {
                    let tuft_size = params.leaf_scale * random_range(&rng, 0.8, 1.2);
                    emit_leaf_cluster(
                        state.position,
                        state.direction,
                        tuft_size,
                        params.seed + leaf_index * 7919u,
                    );
                }
                leaf_index += 1u;
            }
        }
    }

    // Close the top of the trunk.
    emit_cap(&state, segments);
}
