// leaves.wgsl - Leaf card with multiple branches and depth perspective
struct Params {
    color_primary: vec4<f32>,
    color_secondary: vec4<f32>,
    seed: u32,
    scale: f32,
    roughness: f32,
    moisture: f32,
    shadow_strength: f32,
    sheen_strength: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    return value_noise(p) * 0.5 + value_noise(p * 2.0) * 0.3 + value_noise(p * 4.0) * 0.2;
}

fn rotate2d(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

// Realistic leaf shape with proper silhouette
fn leaf_shape(p: vec2<f32>, seed: f32) -> vec3<f32> {
    let t = (p.y + 1.0) * 0.5;

    // Ovate leaf profile - widest near base-middle, tapers to tip
    let width_curve = sin(pow(clamp(t, 0.0, 1.0), 0.7) * 3.14159);
    let taper = 1.0 - smoothstep(0.6, 1.0, t) * 0.4;
    let base_width = width_curve * taper * 0.42;

    // Serrated edges
    let serration_freq = 16.0 + hash21(vec2(seed, 1.0)) * 8.0;
    let serration = sin(t * serration_freq) * 0.02 * smoothstep(0.05, 0.25, t) * smoothstep(0.98, 0.6, t);

    // Subtle asymmetry
    let asym = (hash21(vec2(seed * 2.3, 0.0)) - 0.5) * 0.025 * t;
    let width = base_width + serration;
    let px = p.x - asym;

    let alpha = smoothstep(width + 0.018, width - 0.012, abs(px));
    let stem_mask = smoothstep(-0.98, -0.85, p.y);
    let tip_mask = smoothstep(1.0, 0.88, p.y);

    // Midrib vein
    let midrib = exp(-abs(px) * 50.0) * 0.9;

    // Secondary veins - alternating, curved
    var secondary = 0.0;
    for (var i = 0; i < 7; i++) {
        let fi = f32(i);
        let vein_y = -0.65 + fi * 0.22;
        let side = select(-1.0, 1.0, i % 2 == 0);
        let dy = p.y - vein_y;

        if (dy > 0.0 && dy < 0.25) {
            let curve = dy * dy * 0.35;
            let expected_x = side * (dy * 0.5 + curve);
            let dist = abs(px - expected_x);
            let vein_strength = exp(-dist * 65.0) * smoothstep(0.25, 0.03, dy);
            secondary = max(secondary, vein_strength * 0.55);
        }
    }

    // Tertiary vein hints
    let tertiary = value_noise(vec2(px * 30.0, p.y * 15.0) + seed) * 0.15;

    return vec3(alpha * stem_mask * tip_mask, midrib + secondary + tertiary * alpha, t);
}

fn sample_leaf(uv: vec2<f32>, leaf_pos: vec2<f32>, leaf_rot: f32, leaf_scale: f32, leaf_seed: f32, depth: f32) -> vec4<f32> {
    var local = uv - leaf_pos;
    local = rotate2d(local, leaf_rot);
    local = local / leaf_scale;

    if (abs(local.x) > 0.6 || abs(local.y) > 1.15) {
        return vec4(0.0);
    }

    let shape = leaf_shape(local, leaf_seed);
    let alpha = shape.x;

    if (alpha < 0.01) {
        return vec4(0.0);
    }

    let vein = shape.y;
    let t = shape.z;

    // Color variation per leaf (original coloring)
    let color_var = hash21(vec2(leaf_seed, leaf_seed * 1.7));
    var color = mix(params.color_primary.rgb, params.color_secondary.rgb, color_var * 0.5 + fbm(local * 4.0) * 0.15);

    // Vein darkening
    color = mix(color, color * 0.45, vein * 0.3);

    // Center translucency
    color += (1.0 - abs(local.x) * 2.5) * params.moisture * 0.06;

    // Tip/edge aging
    let age = hash21(vec2(leaf_seed * 2.0, 0.0));
    color = mix(color, vec3(0.5, 0.45, 0.25), smoothstep(0.65, 0.95, t) * age * 0.2);

    // Depth-based darkening (further = darker, less saturated)
    color *= 1.0 - depth * 0.25;
    color = mix(color, vec3(dot(color, vec3(0.3, 0.5, 0.2))), depth * 0.12);

    // Highlight
    let highlight = exp(-length(local - vec2(-0.1, 0.15)) * 3.0) * params.sheen_strength * 0.1;
    color += highlight * (1.0 - params.roughness) * (1.0 - depth * 0.5);

    // Fine texture
    color += (hash21(uv * 250.0 + leaf_seed) - 0.5) * 0.025 * params.roughness;

    return vec4(clamp(color, vec3(0.0), vec3(1.0)), alpha);
}

fn sample_branch(uv: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, thickness: f32, depth: f32, seed: f32) -> vec4<f32> {
    let pa = uv - p0;
    let ba = p1 - p0;
    let len_sq = dot(ba, ba);
    if (len_sq < 0.0001) { return vec4(0.0); }

    let h = clamp(dot(pa, ba) / len_sq, 0.0, 1.0);
    let dist = length(pa - ba * h);

    // Taper along segment
    let tapered_thick = thickness * (1.0 - h * 0.35);

    // Slight organic wobble
    let wobble = sin(h * 15.0 + seed * 3.0) * thickness * 0.15;
    let effective_dist = dist + wobble;

    let alpha = smoothstep(tapered_thick + 0.003, tapered_thick * 0.3, effective_dist);

    if (alpha < 0.01) {
        return vec4(0.0);
    }

    // Greenish-brown bark color derived from params
    let bark_base = mix(params.color_primary.rgb, params.color_secondary.rgb, 0.5) * 0.5;
    let bark_brown = vec3(0.25, 0.2, 0.1);
    var color = mix(bark_brown, bark_base, 0.35);

    // Bark texture variation
    let bark_noise = value_noise(uv * 150.0 + seed);
    color = mix(color, color * 0.7, bark_noise * 0.3);

    // Cylindrical shading
    let norm_dist = dist / tapered_thick;
    let shade = 1.0 - norm_dist * 0.4;
    color *= shade;

    // Depth darkening
    color *= 1.0 - depth * 0.3;

    // Fine texture
    color += (hash21(uv * 400.0 + seed) - 0.5) * 0.03;

    return vec4(clamp(color, vec3(0.0), vec3(1.0)), alpha);
}

// Branch data structure
struct BranchInfo {
    start: vec2<f32>,
    dir: vec2<f32>,
    length: f32,
    depth: f32,
    thickness: f32,
}

fn get_branch_info(idx: i32, seed: f32) -> BranchInfo {
    var b: BranchInfo;

    let h1 = hash22(vec2(f32(idx) * 17.3 + seed, seed * 0.7));
    let h2 = hash22(vec2(f32(idx) * 29.1 + seed, seed * 1.3 + 50.0));
    let h3 = hash21(vec2(f32(idx) * 41.7, seed + 100.0));

    // Different entry points/directions based on index
    let entry_type = idx % 5;

    if (entry_type == 0) {
        // From bottom-left area
        b.start = vec2(h1.x * 0.25, -0.02);
        b.dir = normalize(vec2(0.3 + h2.x * 0.4, 0.7 + h2.y * 0.25));
    } else if (entry_type == 1) {
        // From bottom-right area
        b.start = vec2(0.75 + h1.x * 0.25, -0.02);
        b.dir = normalize(vec2(-0.3 - h2.x * 0.4, 0.7 + h2.y * 0.25));
    } else if (entry_type == 2) {
        // From left side
        b.start = vec2(-0.02, 0.2 + h1.y * 0.5);
        b.dir = normalize(vec2(0.75 + h2.x * 0.2, h2.y - 0.4));
    } else if (entry_type == 3) {
        // From right side
        b.start = vec2(1.02, 0.2 + h1.y * 0.5);
        b.dir = normalize(vec2(-0.75 - h2.x * 0.2, h2.y - 0.4));
    } else {
        // From bottom-center, branching up
        b.start = vec2(0.35 + h1.x * 0.3, -0.02);
        b.dir = normalize(vec2(h2.x - 0.5, 0.8 + h2.y * 0.15));
    }

    b.length = 0.5 + h3 * 0.45;
    b.depth = h1.y * 0.85;  // Z-depth: 0 = front, ~0.85 = back
    b.thickness = (0.006 + h2.y * 0.005) * (1.0 - b.depth * 0.4);

    return b;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / vec2<f32>(f32(size.x), f32(size.y));
    let seed_f = f32(params.seed);

    var final_color = vec3(0.0);
    var final_alpha = 0.0;

    let num_branches = 4 + i32(params.scale * 1.5);
    let num_depth_passes = 4;

    // Render back-to-front for proper depth ordering
    for (var depth_pass = num_depth_passes - 1; depth_pass >= 0; depth_pass--) {
        let depth_min = f32(depth_pass) / f32(num_depth_passes);
        let depth_max = f32(depth_pass + 1) / f32(num_depth_passes);

        for (var bi = 0; bi < num_branches; bi++) {
            let branch = get_branch_info(bi, seed_f);

            // Check if branch belongs to this depth pass
            if (branch.depth < depth_min || branch.depth >= depth_max) {
                continue;
            }

            // Perspective scale factor (further = smaller)
            let perspective_scale = 1.0 - branch.depth * 0.45;

            // Draw branch as multiple curved segments
            let num_segments = 5;
            var prev_pos = branch.start;
            var cur_dir = branch.dir;
            let segment_len = branch.length / f32(num_segments);

            for (var si = 0; si < num_segments; si++) {
                let seg_t = f32(si) / f32(num_segments);
                let seg_seed = seed_f + f32(bi) * 100.0 + f32(si) * 10.0;

                // Gradual curve
                let curve_noise = (hash21(vec2(seg_seed, 0.0)) - 0.5) * 0.25;
                let perp = vec2(-cur_dir.y, cur_dir.x);
                cur_dir = normalize(cur_dir + perp * curve_noise * 0.15);

                let next_pos = prev_pos + cur_dir * segment_len;

                // Segment thickness tapers
                let seg_thick = branch.thickness * (1.0 - seg_t * 0.5) * perspective_scale;

                // Draw branch segment
                let branch_result = sample_branch(uv, prev_pos, next_pos, seg_thick, branch.depth, seg_seed);
                if (branch_result.a > 0.01) {
                    let shadow = final_alpha * params.shadow_strength * 0.1;
                    let shaded = branch_result.rgb * (1.0 - shadow);
                    final_color = mix(final_color, shaded, branch_result.a);
                    final_alpha = final_alpha + branch_result.a * (1.0 - final_alpha);
                }

                // Add leaves along this segment
                let leaves_per_seg = 2 + i32(hash21(vec2(seg_seed, 1.0)) * 2.5);

                for (var li = 0; li < leaves_per_seg; li++) {
                    let leaf_key = vec2(seg_seed + f32(li) * 7.7, f32(li) * 13.3);
                    let lh1 = hash22(leaf_key);
                    let lh2 = hash22(leaf_key + vec2(100.0, 0.0));
                    let lh3 = hash21(leaf_key + vec2(200.0, 0.0));

                    // Position along segment
                    let leaf_t = lh1.x;
                    let pos_on_branch = mix(prev_pos, next_pos, leaf_t);

                    // Offset perpendicular to branch
                    let side = select(-1.0, 1.0, lh1.y > 0.5);
                    let offset_dist = 0.015 + lh2.x * 0.035;
                    let leaf_pos = pos_on_branch + perp * side * offset_dist;

                    // Skip if outside bounds
                    let margin = 0.06;
                    if (leaf_pos.x < margin || leaf_pos.x > 1.0 - margin ||
                        leaf_pos.y < margin || leaf_pos.y > 1.0 - margin) {
                        continue;
                    }

                    // Leaf rotation - generally pointing outward/upward from branch
                    let branch_angle = atan2(cur_dir.y, cur_dir.x);
                    let outward_angle = side * (1.0 + lh2.y * 0.7);
                    let random_twist = (lh3 - 0.5) * 0.5;
                    let leaf_rot = branch_angle + outward_angle + random_twist;

                    // Leaf scale - bigger, with perspective
                    let base_leaf_size = 0.09 + lh2.y * 0.06;
                    let leaf_scale = base_leaf_size * perspective_scale / params.scale;

                    let leaf_seed = lh3 * 100.0;

                    let leaf_result = sample_leaf(uv, leaf_pos, leaf_rot, leaf_scale, leaf_seed, branch.depth);

                    if (leaf_result.a > 0.01) {
                        let shadow = final_alpha * params.shadow_strength * 0.15;
                        let shaded = leaf_result.rgb * (1.0 - shadow);
                        final_color = mix(final_color, shaded, leaf_result.a);
                        final_alpha = final_alpha + leaf_result.a * (1.0 - final_alpha);
                    }
                }

                prev_pos = next_pos;
            }

            // Add some leaves at branch tip
            let tip_pos = prev_pos;
            let tip_leaves = 2 + i32(hash21(vec2(seed_f + f32(bi) * 50.0, 999.0)) * 2.0);

            for (var ti = 0; ti < tip_leaves; ti++) {
                let tip_key = vec2(seed_f + f32(bi) * 77.0 + f32(ti) * 11.0, 888.0);
                let th1 = hash22(tip_key);
                let th2 = hash22(tip_key + vec2(50.0, 0.0));

                let spread = (th1.x - 0.5) * 0.08;
                let fwd = th1.y * 0.05;
                let leaf_pos = tip_pos + cur_dir * fwd + vec2(-cur_dir.y, cur_dir.x) * spread;

                let margin = 0.06;
                if (leaf_pos.x < margin || leaf_pos.x > 1.0 - margin ||
                    leaf_pos.y < margin || leaf_pos.y > 1.0 - margin) {
                    continue;
                }

                let branch_angle = atan2(cur_dir.y, cur_dir.x);
                let leaf_rot = branch_angle + (th2.x - 0.5) * 1.2;

                let base_leaf_size = 0.1 + th2.y * 0.05;
                let leaf_scale = base_leaf_size * perspective_scale / params.scale;

                let leaf_seed = th1.x * 100.0;

                let leaf_result = sample_leaf(uv, leaf_pos, leaf_rot, leaf_scale, leaf_seed, branch.depth);

                if (leaf_result.a > 0.01) {
                    let shadow = final_alpha * params.shadow_strength * 0.15;
                    let shaded = leaf_result.rgb * (1.0 - shadow);
                    final_color = mix(final_color, shaded, leaf_result.a);
                    final_alpha = final_alpha + leaf_result.a * (1.0 - final_alpha);
                }
            }
        }
    }

    textureStore(output, vec2<i32>(gid.xy), vec4(final_color, final_alpha));
}