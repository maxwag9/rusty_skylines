// ============================================================================
// Apply GTAO to HDR Image
// ============================================================================
// Applies ambient occlusion to the HDR image with configurable power curve.
// Note: Ideally AO only affects ambient lighting, but without a separate
// lighting pass, we apply it as a general darkening factor.
// ============================================================================

struct ApplyParams {
    power: f32,           // Power curve for AO (1.0 = linear, 2.0 = stronger)
    intensity: f32,       // Final intensity multiplier (0.0 = no AO, 1.0 = full)
    min_ao: f32,          // Minimum AO value to prevent over-darkening (e.g., 0.1)
    debug_mode: u32,      // 0 = normal, 1 = show AO only, 2 = show edges
};

@group(2) @binding(0) var<uniform> params: ApplyParams;

// ----------------------------------------------------------------------------
// Input Textures
// ----------------------------------------------------------------------------
@group(0) @binding(0) var hdr_input: texture_2d<f32>;   // Resolved HDR
@group(0) @binding(1) var ao_texture: texture_2d<f32>;  // Full-res GTAO

// ----------------------------------------------------------------------------
// Output
// ----------------------------------------------------------------------------
@group(1) @binding(0) var hdr_output: texture_storage_2d<rgba16float, write>;

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coords = vec2<i32>(gid.xy);
    let dims = vec2<i32>(textureDimensions(hdr_output));

    // Bounds check
    if (coords.x >= dims.x || coords.y >= dims.y) {
        return;
    }

    // Load HDR color
    let hdr_color = textureLoad(hdr_input, coords, 0);

    // Load raw AO
    let raw_ao = textureLoad(ao_texture, coords, 0).r;

    // Apply power curve for artistic control
    // Higher power = stronger/darker AO in occluded areas
    let powered_ao = pow(raw_ao, params.power);

    // Apply intensity and clamp to minimum
    // ao = 1.0 means no occlusion, ao = 0.0 means fully occluded
    let final_ao = max(mix(1.0, powered_ao, params.intensity), params.min_ao);

    // Debug modes
    var output_color: vec4<f32>;

    switch (params.debug_mode) {
        case 1u: {
            // Show AO only (grayscale)
            output_color = vec4<f32>(final_ao, final_ao, final_ao, 1.0);
        }
        case 2u: {
            // Show AO as color overlay (red = occluded, white = not occluded)
            let ao_viz = vec3<f32>(final_ao, final_ao * final_ao, final_ao * final_ao);
            output_color = vec4<f32>(ao_viz, 1.0);
        }
        case 3u: {
            // Show raw AO before power curve
            output_color = vec4<f32>(raw_ao, raw_ao, raw_ao, 1.0);
        }
        default: {
            // Normal mode: apply AO to color
            // Note: Physically, AO should only affect ambient/indirect lighting.
            // Since we don't have separate lighting, we apply to the whole image
            // but preserve highlights somewhat by using a luminance-aware blend.

            // Simple approach: multiply RGB by AO
            output_color = vec4<f32>(hdr_color.rgb * final_ao, hdr_color.a);

            // Alternative: Preserve some highlights (uncomment if desired)
            // let luminance = dot(hdr_color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            // let highlight_preserve = smoothstep(1.0, 5.0, luminance);
            // let adjusted_ao = mix(final_ao, 1.0, highlight_preserve);
            // output_color = vec4<f32>(hdr_color.rgb * adjusted_ao, hdr_color.a);
        }
    }

    textureStore(hdr_output, coords, output_color);
}
