fn color_difference() {
    var total_diff = 0;

    let x0 = gid.x * mosaic_pattern_width;
    let y0 = gid.y * mosaic_pattern_width;

    for (var dx = 0; dx < mosaic_pattern_width; dx++) {
        for (var dy = 0; dy < mosaic_pattern_width; dy++) {
            let x = x0 + dx;
            let y = y0 + dy;

            let i1 = textureLoad(texture1, vec2<u64>(x, y)).r;
            let i2 = textureLoad(texture2, vec2<u64>(x, y)).r;

            total_diff += abs(i1 - i2);
        }
    }

    textureStore(out_texture, gid.xy, total_diff);
}

fn compute_merge_weight() {
    // load args
    let noise_sd = noise_sd_buffer[0];
    
    // load texture difference
    let diff = textureLoad(texture_diff, gid.xy).r;
    
    // compute the weight to assign to the comparison frame
    // weight == 0 means that the aligned image is ignored
    // weight == 1 means that the aligned image has full weight
    var weight: f32;

    if robustness == 0 {
        // robustness == 0 means that robust merge is turned off
        weight = 1.0;
    } else {
        // compare the difference to image noise
        // as diff increases, the weight of the aligned image will continuously decrease from 1.0 to 0.0
        // the two extreme cases are:
        // diff == 0                   --> aligned image will have weight 1.0
        // diff >= noise_sd/robustness --> aligned image will have weight 0.0
        let max_diff = noise_sd / robustness;

        weight = 1.0 - f32(diff) / max_diff;
        weight = clamp(weight, 0.0, 1.0);
    }
    
    // write weight
    textureStore(weight_texture, gitd.xy, weight);
}
