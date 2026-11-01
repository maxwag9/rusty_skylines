
struct AutoExposureParams {
    delta_time: f32,
    key_value: f32,
    min_exposure: f32,
    max_exposure: f32,
    bright_scene_speed: f32,
    dark_scene_speed: f32,
    padding: vec2<f32>,
};

struct ExposureState {
    exposure: f32,
    average_log_luminance: f32,
    target_exposure: f32,
    _padding: f32,
};

@group(0) @binding(0)
var hdr_texture: texture_2d<f32>;

@group(2) @binding(0)
var<storage, read> current_exposure: ExposureState;

@group(2) @binding(1)
var<storage, read_write> next_exposure: ExposureState;

@group(2) @binding(2)
var<uniform> params: AutoExposureParams;

const WORKGROUP_SIZE: u32 = 256u;
const SAMPLE_GRID_SIZE: u32 = 128u;
const HISTOGRAM_BIN_COUNT: u32 = 256u;
const HISTOGRAM_REPLICAS: u32 = 4u;

const MIN_LUMINANCE: f32 = 0.0001;
const MIN_LOG_LUMINANCE: f32 = -12.0;
const MAX_LOG_LUMINANCE: f32 = 16.0;
const EXPOSURE_PERCENTILE: f32 = 0.60;

const HISTOGRAM_WEIGHT_SCALE: f32 = 256.0;
const CENTER_WEIGHT_STRENGTH: f32 = 50.0;
const CENTER_WEIGHT_FALLOFF: f32 = 12.0;

var<workgroup> partial_log_luminance: array<f32, 256>;
var<workgroup> partial_total_weight: array<f32, 256>;
var<workgroup> luminance_histogram: array<atomic<u32>, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_index) local_index: u32
) {
    let dimensions = textureDimensions(hdr_texture, 0);

    let sample_width = min(SAMPLE_GRID_SIZE, dimensions.x);
    let sample_height = min(SAMPLE_GRID_SIZE, dimensions.y);
    let sample_count = sample_width * sample_height;

    for (
        var i = local_index;
        i < HISTOGRAM_BIN_COUNT * HISTOGRAM_REPLICAS;
        i += WORKGROUP_SIZE
    ) {
        atomicStore(&luminance_histogram[i], 0u);
    }

    workgroupBarrier();

    var local_sum = 0.0;
    var local_total_weight = 0.0;

    let histogram_replica = local_index % HISTOGRAM_REPLICAS;
    let histogram_bin_width =
        (MAX_LOG_LUMINANCE - MIN_LOG_LUMINANCE)
        / f32(HISTOGRAM_BIN_COUNT);

    for (
        var sample_index = local_index;
        sample_index < sample_count;
        sample_index += WORKGROUP_SIZE
    ) {
        let grid_x = sample_index % sample_width;
        let grid_y = sample_index / sample_width;

        let pixel_x =
            ((2u * grid_x + 1u) * dimensions.x)
            / (2u * sample_width);

        let pixel_y =
            ((2u * grid_y + 1u) * dimensions.y)
            / (2u * sample_height);

        let hdr = textureLoad(
            hdr_texture,
            vec2<i32>(i32(pixel_x), i32(pixel_y)),
            0
        ).rgb;

        let luminance = dot(
            max(hdr, vec3<f32>(0.0)),
            vec3<f32>(0.2126, 0.7152, 0.0722)
        );

        if luminance == 0.0 {
            continue;
        }

        let log_luminance = log2(max(luminance, MIN_LUMINANCE));

        let sample_uv =
            (vec2<f32>(
                f32(grid_x) + 0.5,
                f32(grid_y) + 0.5
            ) / vec2<f32>(
                f32(sample_width),
                f32(sample_height)
            )) * 2.0 - vec2<f32>(1.0);

        let radius_squared = dot(sample_uv, sample_uv);

        let raw_weight = 1.0
            + CENTER_WEIGHT_STRENGTH
            * exp(-CENTER_WEIGHT_FALLOFF * radius_squared);

        let quantized_weight = max(
            1u,
            u32(round(raw_weight * HISTOGRAM_WEIGHT_SCALE))
        );

        let sample_weight =
            f32(quantized_weight) / HISTOGRAM_WEIGHT_SCALE;

        local_sum += log_luminance * sample_weight;
        local_total_weight += sample_weight;

        let histogram_log_luminance = clamp(
            log_luminance,
            MIN_LOG_LUMINANCE,
            MAX_LOG_LUMINANCE
        );

        let normalized_luminance =
            (histogram_log_luminance - MIN_LOG_LUMINANCE)
            / (MAX_LOG_LUMINANCE - MIN_LOG_LUMINANCE);

        let bin = min(
            u32(normalized_luminance * f32(HISTOGRAM_BIN_COUNT)),
            HISTOGRAM_BIN_COUNT - 1u
        );

        let histogram_index =
            histogram_replica * HISTOGRAM_BIN_COUNT + bin;

        atomicAdd(
            &luminance_histogram[histogram_index],
            quantized_weight
        );
    }

    partial_log_luminance[local_index] = local_sum;
    partial_total_weight[local_index] = local_total_weight;

    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;

    loop {
        if stride == 0u {
            break;
        }

        if local_index < stride {
            partial_log_luminance[local_index] +=
                partial_log_luminance[local_index + stride];

            partial_total_weight[local_index] +=
                partial_total_weight[local_index + stride];
        }

        workgroupBarrier();
        stride /= 2u;
    }

    if local_index == 0u {
        let total_weight = partial_total_weight[0];

        if total_weight > 0.0 {
            let average_log_luminance =
                partial_log_luminance[0] / total_weight;

            let percentile_sample = u32(ceil(
                total_weight
                * HISTOGRAM_WEIGHT_SCALE
                * EXPOSURE_PERCENTILE
            ));

            var cumulative_samples = 0u;
            var metered_log_luminance = MIN_LOG_LUMINANCE;

            for (
                var bin = 0u;
                bin < HISTOGRAM_BIN_COUNT;
                bin += 1u
            ) {
                var bin_count = 0u;

                for (
                    var replica = 0u;
                    replica < HISTOGRAM_REPLICAS;
                    replica += 1u
                ) {
                    let histogram_index =
                        replica * HISTOGRAM_BIN_COUNT + bin;

                    bin_count += atomicLoad(
                        &luminance_histogram[histogram_index]
                    );
                }

                cumulative_samples += bin_count;

                if cumulative_samples >= percentile_sample {
                    metered_log_luminance =
                        MIN_LOG_LUMINANCE
                        + (f32(bin) + 0.5) * histogram_bin_width;

                    break;
                }
            }

            let min_exposure = max(
                params.min_exposure,
                MIN_LUMINANCE
            );

            let max_exposure = max(
                params.max_exposure,
                min_exposure
            );

            let key_value = max(
                params.key_value,
                MIN_LUMINANCE
            );

            let target_log_exposure = clamp(
                log2(key_value) - metered_log_luminance,
                log2(min_exposure),
                log2(max_exposure)
            );

            let target_exposure = exp2(target_log_exposure);
            let previous_exposure = current_exposure.exposure;

            var adapted_exposure = target_exposure;

            if previous_exposure >= min_exposure
                && previous_exposure <= max_exposure
            {
                let current_log_exposure = log2(previous_exposure);

                let adaptation_speed = select(
                    max(params.dark_scene_speed, 0.0),
                    max(params.bright_scene_speed, 0.0),
                    target_log_exposure < current_log_exposure
                );

                let blend = 1.0 - exp(
                    -max(params.delta_time, 0.0) * adaptation_speed
                );

                adapted_exposure = exp2(
                    mix(
                        current_log_exposure,
                        target_log_exposure,
                        blend
                    )
                );
            }

            next_exposure.exposure = adapted_exposure;
            next_exposure.average_log_luminance = average_log_luminance;
            next_exposure.target_exposure = target_exposure;
            next_exposure._padding = 0.0;
        } else {
            next_exposure.exposure = current_exposure.exposure;
            next_exposure.average_log_luminance =
                current_exposure.average_log_luminance;
            next_exposure.target_exposure =
                current_exposure.target_exposure;
            next_exposure._padding = 0.0;
        }

        let average_log_luminance =
            partial_log_luminance[0] / max(total_weight, 0.000001);

        let percentile_sample = u32(ceil(
            total_weight
            * HISTOGRAM_WEIGHT_SCALE
            * EXPOSURE_PERCENTILE
        ));

        var cumulative_samples = 0u;
        var metered_log_luminance = MIN_LOG_LUMINANCE;

        for (
            var bin = 0u;
            bin < HISTOGRAM_BIN_COUNT;
            bin += 1u
        ) {
            var bin_count = 0u;

            for (
                var replica = 0u;
                replica < HISTOGRAM_REPLICAS;
                replica += 1u
            ) {
                let histogram_index =
                    replica * HISTOGRAM_BIN_COUNT + bin;

                bin_count += atomicLoad(
                    &luminance_histogram[histogram_index]
                );
            }

            cumulative_samples += bin_count;

            if cumulative_samples >= percentile_sample {
                metered_log_luminance =
                    MIN_LOG_LUMINANCE
                    + (f32(bin) + 0.5) * histogram_bin_width;

                break;
            }
        }

        let min_exposure = max(
            params.min_exposure,
            MIN_LUMINANCE
        );

        let max_exposure = max(
            params.max_exposure,
            min_exposure
        );

        let key_value = max(
            params.key_value,
            MIN_LUMINANCE
        );

        let target_log_exposure = clamp(
            log2(key_value) - metered_log_luminance,
            log2(min_exposure),
            log2(max_exposure)
        );

        let target_exposure = exp2(target_log_exposure);
        let previous_exposure = current_exposure.exposure;

        var adapted_exposure = target_exposure;

        if previous_exposure >= min_exposure
            && previous_exposure <= max_exposure
        {
            let current_log_exposure = log2(previous_exposure);

            let adaptation_speed = select(
                max(params.dark_scene_speed, 0.0),
                max(params.bright_scene_speed, 0.0),
                target_log_exposure < current_log_exposure
            );

            let blend = 1.0 - exp(
                -max(params.delta_time, 0.0) * adaptation_speed
            );

            adapted_exposure = exp2(
                mix(
                    current_log_exposure,
                    target_log_exposure,
                    blend
                )
            );
        }

        next_exposure.exposure = adapted_exposure;
        next_exposure.average_log_luminance = average_log_luminance;
        next_exposure.target_exposure = target_exposure;
        next_exposure._padding = 0.0;
    }
}
