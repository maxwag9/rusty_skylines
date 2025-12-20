struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,

    sun_direction: vec3<f32>,
    time: f32,

    camera_pos: vec3<f32>,
    orbit_radius: f32,

    moon_direction: vec3<f32>,
    _pad0: f32,
};

struct FogUniforms {
    screen_size: vec2<f32>,
    proj_params: vec2<f32>,

    fog_density: f32,
    fog_height: f32,
    cam_height: f32,
    _pad0: f32,

    fog_color: vec3<f32>,
    _pad1: f32,

    fog_sky_factor: f32,
    fog_height_falloff: f32,
    fog_start: f32,
    fog_end: f32,
};

struct PickUniform {
    pos: vec3<f32>,   // offset 0, size 12, alignment 16
    radius: f32,      // offset 16
    underwater: u32,     // offset 20
    // padding to 32
    color: vec3<f32>, // offset 32
    // padding to 48
}


@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var<uniform> fog: FogUniforms;

@group(2) @binding(0)
var<uniform> pick: PickUniform;


struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let world_pos = in.position;

    out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = normalize(in.normal);
    out.color = in.color;
    out.world_pos = world_pos;

    return out;
}




@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let is_underwater = in.world_pos.y < 0.0;
    let underwater_bool = pick.underwater == 1u;
    if (underwater_bool != is_underwater) {
        discard;
    }

    let n = normalize(in.world_normal);
    let l = normalize(uniforms.sun_direction);

    let up = vec3<f32>(0.0, 1.0, 0.0);
    let hemi = clamp(dot(n, up) * 0.5 + 0.5, 0.0, 1.0);
    let sky_ambient = vec3<f32>(0.25, 0.28, 0.32);
    let ground_ambient = vec3<f32>(0.10, 0.09, 0.08);
    let ambient = mix(ground_ambient, sky_ambient, hemi);

    let ndotl = dot(n, l);
    let wrap = 0.3;
    let diffuse = clamp((ndotl + wrap) / (1.0 + wrap), 0.0, 1.0);


    let base_color = in.color * (ambient + diffuse) + 0.05;

    let dist = distance(in.world_pos, uniforms.camera_pos);

    let f = fog_factor(dist, in.world_pos.y);

    // slight desaturation improves realism
    let gray = dot(base_color, vec3<f32>(0.3, 0.59, 0.11));
    let desat = mix(base_color, vec3<f32>(gray), f * 0.05);

    let final_color = mix(desat, fog.fog_color, f);

    var color = final_color;

    if (pick.radius > 0.0) {
        let d = distance(in.world_pos, pick.pos);

        if (d < pick.radius) {
            // smooth falloff looks better than a hard edge
            let t = 1.0 - smoothstep(0.0, pick.radius, d);
            color = mix(color, pick.color, t);
        }
    }

    return vec4<f32>(color, 1.0);

}

fn height_factor_at(pos_y: f32) -> f32 {
    let dy = pos_y - fog.fog_height;
    return exp(-dy * fog.fog_height_falloff);
}

fn fog_factor(dist: f32, pos_y: f32) -> f32 {
    let h = height_factor_at(pos_y);
    return 1.0 - exp(-dist * fog.fog_density * h);
}