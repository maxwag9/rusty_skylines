// ----------------------------------------------------------------------------
// Downsample Normals to Half Resolution
// ----------------------------------------------------------------------------
// Takes full resolution normals (Rgba8Unorm, encoded as n*0.5+0.5) and
// downsamples by averaging 2x2 blocks with proper renormalization.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Group 0: Input Texture (Full Resolution Normals)
// ----------------------------------------------------------------------------
@group(0) @binding(0) var input_normals: texture_2d<f32>;

// ----------------------------------------------------------------------------
// Group 1: Output Storage (Half Resolution Normals)
// ----------------------------------------------------------------------------
@group(1) @binding(0) var output_normals_half: texture_storage_2d<rgba8unorm, write>;

// ----------------------------------------------------------------------------
// Helper: Decode normal from [0,1] to [-1,1]
// ----------------------------------------------------------------------------
fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return encoded * 2.0 - 1.0;
}

// ----------------------------------------------------------------------------
// Helper: Encode normal from [-1,1] to [0,1]
// ----------------------------------------------------------------------------
fn encode_normal(normal: vec3<f32>) -> vec3<f32> {
    return normal * 0.5 + 0.5;
}

// ----------------------------------------------------------------------------
// Main Compute
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_coords = vec2<i32>(gid.xy);
    let out_dims = vec2<i32>(textureDimensions(output_normals_half));

    // Bounds check against output dimensions
    if (out_coords.x >= out_dims.x || out_coords.y >= out_dims.y) {
        return;
    }

    // Map output coordinates to input coordinates (2x2 block)
    let in_coords = out_coords * 2;
    let in_dims = vec2<i32>(textureDimensions(input_normals));

    // Sample 2x2 block from full resolution texture
    // Clamp to valid range to handle edge cases
    let c00 = min(in_coords + vec2<i32>(0, 0), in_dims - 1);
    let c10 = min(in_coords + vec2<i32>(1, 0), in_dims - 1);
    let c01 = min(in_coords + vec2<i32>(0, 1), in_dims - 1);
    let c11 = min(in_coords + vec2<i32>(1, 1), in_dims - 1);

    // Load and decode normals from [0,1] to [-1,1]
    let n00 = decode_normal(textureLoad(input_normals, c00, 0).rgb);
    let n10 = decode_normal(textureLoad(input_normals, c10, 0).rgb);
    let n01 = decode_normal(textureLoad(input_normals, c01, 0).rgb);
    let n11 = decode_normal(textureLoad(input_normals, c11, 0).rgb);

    // Average the normals
    var avg_normal = (n00 + n10 + n01 + n11) * 0.25;

    // Renormalize (handle degenerate case where average is zero)
    let len = length(avg_normal);
    if (len > 0.0001) {
        avg_normal = avg_normal / len;
    } else {
        // Fallback to up vector if normals cancel out (shouldn't happen normally)
        avg_normal = vec3<f32>(0.0, 1.0, 0.0);
    }

    // Encode back to [0,1] and store
    let encoded = encode_normal(avg_normal);
    textureStore(output_normals_half, out_coords, vec4<f32>(encoded, 1.0));
}