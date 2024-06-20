@group(0) @binding(0) var in_texture: texture_2d<f32>;
@group(0) @binding(1) var out_texture: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(0) var<uniform> avg_pool_params: AvgPoolParams;

struct AvgPoolParams {
    scale: i32,
    black_level: f32,
}

@compute
@workgroup_size(1)
fn avg_pool(@builtin(global_invocation_id) gid: vec3u) {
    var out_pixel = 0.0f;
    let p0 = vec2i(gid.xy) * avg_pool_params.scale;

    for (var dx = 0i; dx < avg_pool_params.scale; dx++) {
        for (var dy = 0i; dy < avg_pool_params.scale; dy++) {
            out_pixel += f32(textureLoad(in_texture, p0 + vec2i(dx, dy), 0).r) - avg_pool_params.black_level;
        }
    }

    out_pixel /= f32(avg_pool_params.scale * avg_pool_params.scale);

    textureStore(out_texture, gid.xy, vec4f(out_pixel, out_pixel, out_pixel, 1.0));
}

// @compute
// fn avg_pool_normalization(@builtin(global_invocation_id) gid: vec3u) {
//     var out_pixel = 0.0f;

//     let x0 = gid.x * scale;
//     let y0 = gid.y * scale;

//     let norm_factors = vec4f(factor_red, factor_green, factor_green, factor_blue);
//     let mean_factor = 0.25 * (norm_factors.x + norm_factors.y + norm_factors.z + norm_factors.w);

//     for (var dx = 0; dx < scale; dx++) {
//         for (var dy = 0; dy < scale; dy++) {
//             let  x = x0 + dx;
//             let  y = y0 + dy;

//             out_pixel += (mean_factor / norm_factors[dy * scale + dx] * textureLoad(in_texture, vec2u(x, y), 0).r - black_level);
//         }
//     }

//     textureStore(out_texture, gid.xy, out_pixel / scale * scale);
// }
