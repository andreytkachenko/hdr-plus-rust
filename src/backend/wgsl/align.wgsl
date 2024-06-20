

// Generic function for computation of tile differences that works for any search distance
fn compute_tile_differences() {
    // load args
    let dims = textureDimensions(ref_texture);

    let texture_width = dims.x;
    let texture_height = dims.y;
    let n_pos_1d = 2 * search_dist + 1;

    // compute tile position if previous alignment were 0
    let x0 = gid.x * tile_size / 2;
    let y0 = gid.y * tile_size / 2;
    
    // compute current tile displacement based on thread index
    var dy0 = gid.z / n_pos_1d - search_dist;
    var dx0 = gid.z % n_pos_1d - search_dist;
    
    // factor in previous alignment
    let prev_align = textureLoad(prev_alignment, vec2u(gid.x, gid.y));

    dx0 += downscale_factor * prev_align.x  ;
    dy0 += downscale_factor * prev_align.y  ;
    
    // compute tile difference
    var diff = 0.0;
    for (var dx1 = 0; dx1 < tile_size; dx1++) {
        for (var dy1 = 0; dy1 < tile_size; dy1++) {
            // compute the indices of the pixels to compare
            let ref_tile_x = x0 + dx1;
            let ref_tile_y = y0 + dy1;

            let comp_tile_x = ref_tile_x + dx0;
            let comp_tile_y = ref_tile_y + dy0;

            var diff_abs: f32;
            
            // if the comparison pixels are outside of the frame, attach a high loss to them
            if (comp_tile_x < 0) || (comp_tile_y < 0) || (comp_tile_x >= texture_width) || (comp_tile_y >= texture_height) {
                diff_abs = abs(textureLoad(ref_texture, vec2u(ref_tile_x, ref_tile_y)).r - 2 * FLOAT16_MIN_VAL, 0);
            } else {
                diff_abs = abs(textureLoad(ref_texture, vec2u(ref_tile_x, ref_tile_y), 0).r - textureLoad(comp_texture, vec2u(comp_tile_x, comp_tile_y), 0).r);
            }

            diff += (1 - weight_ssd) * diff_abs + weight_ssd * diff_abs * diff_abs;
        }
    }
    
    // store tile difference
    textureStore(tile_diff, gid, diff);
}


// 
//  Highly-optimized function for computation of tile differences that works only for search_distance == 2 (25 total combinations).
//  The aim of this function is to reduce the number of memory accesses required compared to the more simple function compute_tile_differences() while providing equal results. As the alignment always checks shifts on a 5x5 pixel grid, a simple implementation would read 25 pixels in the comparison texture for each pixel in the reference texture. This optimized function however uses a buffer vector covering 5 complete rows of the texture that slides line by line through the comparison texture and reduces the number of memory reads considerably.
//  
fn compute_tile_differences25() {
    // load args
    let texture_width = ref_texture.get_width();
    let texture_height = ref_texture.get_height();
    // int ref_tile_x, ref_tile_y, comp_tile_x, comp_tile_y, tmp_index, dx_i, dy_i,;
    
    // compute tile position if previous alignment were 0
    let x0 = gid.x * tile_size / 2;
    let y0 = gid.y * tile_size / 2;
    
    // factor in previous alignment
    let prev_align = textureLoad(prev_alignment, vec2u(gid.x, gid.y), 0);
    let dx0 = downscale_factor * prev_align.x;
    let dy0 = downscale_factor * prev_align.y;

    var diff: array<f32, 25>;
    var tmp_comp: array<f16, 340>; // 5 * 68
    
    // loop over first 4 rows of comp_texture
    for (var dy = -2i; dy < 2; dy++) {
        
        // loop over columns of comp_texture to copy first 4 rows of comp_texture into tmp_comp
        for (var dx = -2i; dx < tile_size + 2; dx++) {
            let comp_tile_x = x0 + dx0 + dx;
            let comp_tile_y = y0 + dy0 + dy;
            
            // index of corresponding pixel value in tmp_comp
            let tmp_index = (dy + 2) * (tile_size + 4) + dx + 2;
            
            // if the comparison pixels are outside of the frame, attach a high loss to them
            if (comp_tile_x < 0) || (comp_tile_y < 0) || (comp_tile_x >= texture_width) || (comp_tile_y >= texture_height) {
                tmp_comp[tmp_index] = FLOAT16_MIN_VAL;
            } else {
                tmp_comp[tmp_index] = FLOAT16_05_VAL * textureLoad(comp_texture, vec2u(comp_tile_x, comp_tile_y), 0).r;
            }
        }
    }
    
    // loop over rows of ref_texture
    for (var dy = 0i; dy < tile_size; dy++) {
        
        // loop over columns of comp_texture to copy 1 additional row of comp_texture into tmp_comp
        for (var dx = -2i; dx < tile_size + 2; dx++) {
            let comp_tile_x = x0 + dx0 + dx;
            let comp_tile_y = y0 + dy0 + dy + 2;
            
            // index of corresponding pixel value in tmp_comp
            let tmp_index = ((dy + 4) % 5) * (tile_size + 4) + dx + 2;
            
            // if the comparison pixels are outside of the frame, attach a high loss to them
            if (comp_tile_x < 0) || (comp_tile_y < 0) || (comp_tile_x >= texture_width) || (comp_tile_y >= texture_height) {
                tmp_comp[tmp_index] = FLOAT16_MIN_VAL;
            } else {
                tmp_comp[tmp_index] = FLOAT16_05_VAL * textureLoad(comp_texture, vec2u(comp_tile_x, comp_tile_y), 0).r;
            }
        }
        
        // loop over columns of ref_texture
        for (var dx = 0; dx < tile_size; dx += 2) {
            let ref_tile_x = x0 + dx;
            let ref_tile_y = y0 + dy;

            let tmp_ref0 = textureLoad(ref_texture, vec2u(ref_tile_x + 0, ref_tile_y), 0).r;
            let tmp_ref1 = textureLoad(ref_texture, vec2u(ref_tile_x + 1, ref_tile_y), 0).r;
            
            // loop over 25 test displacements
            for (var i = 0; i < 25; i++) {
                let dx_i = i % 5;
                let dy_i = i / 5;
                
                // index of corresponding pixel value in tmp_comp
                let tmp_index = ((dy + dy_i) % 5) * (tile_size + 4) + dx + dx_i;

                let diff_abs0 = abs(tmp_ref0 - 2.0 * tmp_comp[tmp_index + 0]);
                let diff_abs1 = abs(tmp_ref1 - 2.0 * tmp_comp[tmp_index + 1]);
                
                // add difference to corresponding combination
                diff[i] += ((1 - weight_ssd) * (diff_abs0 + diff_abs1) + weight_ssd * (diff_abs0 * diff_abs0 + diff_abs1 * diff_abs1));
            }
        }
    }
    
    // store tile differences in texture
    for (var i = 0; i < 25; i++) {
        textureStore(tile_diff, vec3u(i, gid.x, gid.y), diff[i]);
    }
}
//
//  Highly - optimized function for computation of tile differences that works only for search_distance == 2 (25 total combinations).
// The aim of this function is to reduce the number of memory accesses required compared to the more simple function compute_tile_differences() while extending it with a scaling of pixel intensities by the ratio of mean values of both tiles.
//

fn compute_tile_differences_exposure25() {
    
    // load args
    let texture_width = ref_texture.get_width();
    let texture_height = ref_texture.get_height();

    // int ref_tile_x, ref_tile_y, comp_tile_x, comp_tile_y, tmp_index, dx_i, dy_i;
    
    // compute tile position if previous alignment were 0
    let x0 = gid.x * tile_size / 2;
    let y0 = gid.y * tile_size / 2;
    
    // factor in previous alignment
    let prev_align = textureLoad(prev_alignment, vec2u(gid.x, gid.y), 0);
    let dx0 = downscale_factor * prev_align.x;
    let dy0 = downscale_factor * prev_align.y;

    var sum_u: array<f32, 25>;
    var sum_v: array<f32, 25>;
    var diff: array<f32, 25>;
    var ratio: array<f32, 25>;
    var tmp_comp: array<f16, 340>; // 5 * 68

    // loop over first 4 rows of comp_texture
    for (var dy = -2i; dy < 2; dy++) {
        
        // loop over columns of comp_texture to copy first 4 rows of comp_texture into tmp_comp
        for (var dx = -2i; dx < tile_size + 2; dx++) {

            let comp_tile_x = x0 + dx0 + dx;
            let comp_tile_y = y0 + dy0 + dy;
            
            // index of corresponding pixel value in tmp_comp
            let tmp_index = (dy + 2) * (tile_size + 4) + dx + 2;
            
            // if the comparison pixels are outside of the frame, attach a high loss to them
            if (comp_tile_x < 0) || (comp_tile_y < 0) || (comp_tile_x >= texture_width) || (comp_tile_y >= texture_height) {
                tmp_comp[tmp_index] = FLOAT16_MAX_VAL;
            } else {
                tmp_comp[tmp_index] = max(FLOAT16_ZERO_VAL, FLOAT16_05_VAL * textureLoad(comp_texture, vec2u(comp_tile_x, comp_tile_y)).r);
            }
        }
    }
    
    // loop over rows of ref_texture
    for (var dy = 0i; dy < tile_size; dy++) {
        
        // loop over columns of comp_texture to copy 1 additional row of comp_texture into tmp_comp
        for (var dx = -2i; dx < tile_size + 2; dx++) {
            let comp_tile_x = x0 + dx0 + dx;
            let comp_tile_y = y0 + dy0 + dy + 2;
            
            // index of corresponding pixel value in tmp_comp
            let tmp_index = ((dy + 4) % 5) * (tile_size + 4) + dx + 2;
            
            // if the comparison pixels are outside of the frame, attach a high loss to them
            if (comp_tile_x < 0) || (comp_tile_y < 0) || (comp_tile_x >= texture_width) || (comp_tile_y >= texture_height) {
                tmp_comp[tmp_index] = FLOAT16_MAX_VAL;
            } else {
                tmp_comp[tmp_index] = max(FLOAT16_ZERO_VAL, FLOAT16_05_VAL * textureLoad(comp_texture, vec2u(comp_tile_x, comp_tile_y), 0).r);
            }
        }
        
        // loop over columns of ref_texture
        for (var dx = 0; dx < tile_size; dx += 2) {
            let ref_tile_x = x0 + dx;
            let ref_tile_y = y0 + dy;

            let tmp_ref0 = max(FLOAT16_ZERO_VAL, textureLoad(ref_texture, vec2u(ref_tile_x + 0, ref_tile_y), 0).r);
            let tmp_ref1 = max(FLOAT16_ZERO_VAL, textureLoad(ref_texture, vec2u(ref_tile_x + 1, ref_tile_y), 0).r);
              
            // loop over 25 test displacements
            for (var i = 0; i < 25; i++) {
                let dx_i = i % 5;
                let dy_i = i / 5;
                
                // index of corresponding pixel value in tmp_comp
                let tmp_index = ((dy + dy_i) % 5) * (tile_size + 4) + dx + dx_i;

                let tmp_comp_val0 = tmp_comp[tmp_index + 0];
                let tmp_comp_val1 = tmp_comp[tmp_index + 1];

                if tmp_comp_val0 > -1 {
                    sum_u[i] += tmp_ref0;
                    sum_v[i] += 2.0 * tmp_comp_val0;
                }

                if tmp_comp_val1 > -1 {
                    sum_u[i] += tmp_ref1;
                    sum_v[i] += 2.0 * tmp_comp_val1;
                }
            }
        }
    }

    for (var i = 0; i < 25; i++) {
        // calculate ratio of mean values of the tiles, which is used for correction of slight differences in exposure
        ratio[i] = clamp(sum_u[i] / (sum_v[i] + 1e-9), 0.9, 1.1);
    }
        
    // loop over first 4 rows of comp_texture
    for (var dy = -2i; dy < 2; dy++) {
        
        // loop over columns of comp_texture to copy first 4 rows of comp_texture into tmp_comp
        for (var dx = -2i; dx < tile_size + 2; dx++) {

            let comp_tile_x = x0 + dx0 + dx;
            let comp_tile_y = y0 + dy0 + dy;
            
            // index of corresponding pixel value in tmp_comp
            let tmp_index = (dy + 2) * (tile_size + 4) + dx + 2;
            
            // if the comparison pixels are outside of the frame, attach a high loss to them
            if (comp_tile_x < 0) || (comp_tile_y < 0) || (comp_tile_x >= texture_width) || (comp_tile_y >= texture_height) {
                tmp_comp[tmp_index] = FLOAT16_MIN_VAL;
            } else {
                tmp_comp[tmp_index] = max(FLOAT16_ZERO_VAL, FLOAT16_05_VAL * textureLoad(comp_texture, vec2u(comp_tile_x, comp_tile_y), 0).r);
            }
        }
    }
    
    // loop over rows of ref_texture
    for (var dy = 0i; dy < tile_size; dy++) {
        
        // loop over columns of comp_texture to copy 1 additional row of comp_texture into tmp_comp
        for (var dx = -2i; dx < tile_size + 2; dx++) {

            let comp_tile_x = x0 + dx0 + dx;
            let comp_tile_y = y0 + dy0 + dy + 2;
            
            // index of corresponding pixel value in tmp_comp
            let tmp_index = ((dy + 4) % 5) * (tile_size + 4) + dx + 2;
            
            // if the comparison pixels are outside of the frame, attach a high loss to them
            if (comp_tile_x < 0) || (comp_tile_y < 0) || (comp_tile_x >= texture_width) || (comp_tile_y >= texture_height) {
                tmp_comp[tmp_index] = FLOAT16_MIN_VAL;
            } else {
                tmp_comp[tmp_index] = max(FLOAT16_ZERO_VAL, FLOAT16_05_VAL * textureLoad(comp_texture, vec2u(comp_tile_x, comp_tile_y)).r);
            }
        }
        
        // loop over columns of ref_texture
        for (var dx = 0; dx < tile_size; dx += 2) {
            let ref_tile_x = x0 + dx;
            let ref_tile_y = y0 + dy;

            let tmp_ref0 = max(FLOAT16_ZERO_VAL, textureLoad(ref_texture, vec2u(ref_tile_x + 0, ref_tile_y)).r);
            let tmp_ref1 = max(FLOAT16_ZERO_VAL, textureLoad(ref_texture, vec2u(ref_tile_x + 1, ref_tile_y)).r);
              
            // loop over 25 test displacements
            for (var i = 0; i < 25; i++) {
                let dx_i = i % 5;
                let dy_i = i / 5;
                
                // index of corresponding pixel value in tmp_comp
                let tmp_index = ((dy + dy_i) % 5) * (tile_size + 4) + dx + dx_i;

                let diff_abs0 = abs(tmp_ref0 - 2.0 * ratio[i] * tmp_comp[tmp_index + 0]);
                let diff_abs1 = abs(tmp_ref1 - 2.0 * ratio[i] * tmp_comp[tmp_index + 1]);
                
                // add difference to corresponding combination
                diff[i] += ((1 - weight_ssd) * (diff_abs0 + diff_abs1) + weight_ssd * (diff_abs0 * diff_abs0 + diff_abs1 * diff_abs1));
            }
        }
    }
    
    // store tile differences in texture
    for (var i = 0; i < 25; i++) {
        textureStore(tile_diff, vec3u(i, gid.x, gid.y), diff[i]);
    }
}


// 
// At transitions between moving objects and non - moving background, the alignment vectors from downsampled images may be inaccurate. 
//   Therefore, after upsampling to the next resolution level, three candidate alignment vectors are evaluated for each tile. 
//   In addition to the vector obtained from upsampling, two vectors from neighboring tiles are checked. 
//   As a consequence, alignment at the transition regions described above is more accurate.
// 
// See section on "Hierarchical alignment" 
//   in https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf 
//   and section "Multi-scale Pyramid Alignment" in https://www.ipol.im/pub/art/2021/336/
       
fn correct_upsampling_error() {
    // load args
    let texture_width = ref_texture.get_width();
    let texture_height = ref_texture.get_height();
    
    // initialize some variables
    var tmp_ref: array<f16, 64>;

    // compute tile position if previous alignment were 0
    let x0 = gid.x * tile_size / 2;
    let y0 = gid.y * tile_size / 2;
    
    // calculate shifts of gid index for 3 candidate alignments to evaluate
    // int3 const x_shift = int3(0, ((gid.x % 2 == 0) ?, -1,: 1), 0);
    // int3 const y_shift = int3(0, 0, ((gid.y % 2 == 0) ?, -1,: 1));

    let x = clamp(vec3i(gid.x + x_shift), 0, n_tiles_x - 1);
    let y = clamp(vec3i(gid.y + y_shift), 0, n_tiles_y - 1);
    
    // factor in previous alignment for 3 candidates
    let prev_align0 = textureLoad(prev_alignment, vec2u(x[0], y[0]), 0);
    let prev_align1 = textureLoad(prev_alignment, vec2u(x[1], y[1]), 0);
    let prev_align2 = textureLoad(prev_alignment, vec2u(x[2], y[2]), 0);

    let dx0 = downscale_factor * vec3i(prev_align0.x, prev_align1.x, prev_align2.x);
    let dy0 = downscale_factor * vec3i(prev_align0.y, prev_align1.y, prev_align2.y);
    
    // compute tile differences for 3 candidates
    var diff = vec3f(0.0, 0.0, 0.0);
    var ratio = vec3f(1.0, 1.0, 1.0);
    
    // calculate exposure correction factors for slight scaling of pixel intensities
    if uniform_exposure != 1 {
        var sum_u = vec3f(0.0, 0.0, 0.0);
        var sum_v = vec3f(0.0, 0.0, 0.0);

        // loop over all rows
        for (var dy = 0; dy < tile_size; dy += 64 / tile_size) {
            // copy 64/tile_size rows into temp vector
            for (var i = 0; i < 64; i++) {
                tmp_ref[i] = max(FLOAT16_ZERO_VAL, textureLoad(ref_texture, vec2u(x0 + (i % tile_size), y0 + dy + i32(i / tile_size)), 0).r);
            }
            
            // loop over three candidates
            for (var c = 0; c < 3; c++) {
                // loop over tmp vector: candidate c of alignment vector
                let tmp_tile_x = x0 + dx0[c];
                let tmp_tile_y = y0 + dy0[c] + dy;

                for (var i = 0; i < 64; i++) {
                    // compute the indices of the pixels to compare
                    let comp_tile_x = tmp_tile_x + (i % tile_size);
                    let comp_tile_y = tmp_tile_y + i32(i / tile_size);

                    if (comp_tile_x >= 0) && (comp_tile_y >= 0) && (comp_tile_x < texture_width) && (comp_tile_y < texture_height) {
                        sum_u[c] += tmp_ref[i];
                        sum_v[c] += max(FLOAT16_ZERO_VAL, textureLoad(comp_texture, vec2u(comp_tile_x, comp_tile_y)).r);
                    }
                }
            }
        }

        for (var c = 0; c < 3; c++) {
            // calculate ratio of mean values of the tiles, which is used for correction of slight differences in exposure
            ratio[c] = clamp(sum_u[c] / (sum_v[c] + 1e-9), 0.9, 1.1);
        }
    }
    
    // loop over all rows
    for (var dy = 0; dy < tile_size; dy += 64 / tile_size) {
        // copy 64/tile_size rows into temp vector
        for (var i = 0; i < 64; i++) {
            tmp_ref[i] = textureLoad(ref_texture, vec2u(x0 + (i % tile_size), y0 + dy + i32(i / tile_size)), 0).r;
        }
        
        // loop over three candidates
        for (var c = 0; c < 3; c++) {
            // loop over tmp vector: candidate c of alignment vector
            let tmp_tile_x = x0 + dx0[c];
            let tmp_tile_y = y0 + dy0[c] + dy;

            for (var i = 0; i < 64; i++) {
                // compute the indices of the pixels to compare
                let comp_tile_x = tmp_tile_x + (i % tile_size);
                let comp_tile_y = tmp_tile_y + i32(i / tile_size);
                
                // if (comp_tile_x < 0 || comp_tile_y < 0 || comp_tile_x >= texture_width || comp_tile_y >= texture_height) => set weight_outside = 1, else set weight_outside = 0
                weight_outside = clamp(texture_width - comp_tile_x - 1, -1, 0) + clamp(texture_height - comp_tile_y - 1, -1, 0) + clamp(comp_tile_x, -1, 0) + clamp(comp_tile_y, -1, 0);
                weight_outside = -max(-1, weight_outside);

                let diff_abs = abs(tmp_ref[i] - (1 - weight_outside) * ratio[c] * (textureLoad(comp_texture, vec2u(comp_tile_x, comp_tile_y), 0).r) - weight_outside * 2 * FLOAT16_MIN_VAL);
                
                // add difference to corresponding combination
                diff[c] += (1.0 - weight_ssd) * diff_abs + weight_ssd * diff_abs * diff_abs;
            }
        }
    }
    
    // store corrected (best) alignment
    if diff[0] < diff[1] & diff[0] < diff[2] {
        textureStore(prev_alignment_corrected, gid.xy, prev_align0);
    } else if diff[1] < diff[2] {
        textureStore(prev_alignment_corrected, gid.xy, prev_align1);
    } else {
        textureStore(prev_alignment_corrected, gid.xy, prev_align2);
    }
}

fn find_best_tile_alignment() {
    // load args
    let n_pos_1d = 2 * search_dist + 1;
    let n_pos_2d = n_pos_1d * n_pos_1d;
    
    // find tile displacement with the lowest pixel difference
    var min_diff_val = 1e20f;
    var min_diff_idx = 0u;

    for (var i = 0u; i < n_pos_2d; i++) {
        let current_diff = textureLoad(tile_diff, vec3u(i, gid.x, gid.y)).r;

        if current_diff < min_diff_val {
            min_diff_val = current_diff;
            min_diff_idx = i;
        }
    }
    
    // compute tile displacement if previous alignment were 0
    let dx = min_diff_idx % n_pos_1d - search_dist;
    let dy = min_diff_idx / n_pos_1d - search_dist;
    
    // factor in previous alignment
    let prev_align = downscale_factor * textureLoad(prev_alignment, gid.xy, 0);
    
    // store alignment
    textureStore(current_alignment, gid.xy, vec4i(prev_align.x + dx, prev_align.y + dy, 0, 0));
}


fn warp_texture_bayer() {
    // load args
    let x = gid.x;
    let y = gid.y;

    let half_tile_size_float = f32(half_tile_size);
    
    // compute the coordinates of output pixel in tile-grid units
    let x_grid = (x + 0.5) / half_tile_size_float - 1.0;
    let y_grid = (y + 0.5) / half_tile_size_float - 1.0;

    let x_grid_floor = i32(max(0.0, floor(x_grid)) + 0.1);
    let y_grid_floor = i32(max(0.0, floor(y_grid)) + 0.1);

    let x_grid_ceil = i32(min(ceil(x_grid), n_tiles_x - 1.0) + 0.1);
    let y_grid_ceil = i32(min(ceil(y_grid), n_tiles_y - 1.0) + 0.1);
    
    // weights calculated for the bilinear interpolation
    let weight_x = (f32(x % half_tile_size) + 0.5) / (2.0 * half_tile_size_float);
    let weight_y = (f32(y % half_tile_size) + 0.5) / (2.0 * half_tile_size_float);
    
    // factor in alignment
    let prev_align0 = downscale_factor * textureLoad(prev_alignment, vec2u(x_grid_floor, y_grid_floor), 0);
    let prev_align1 = downscale_factor * textureLoad(prev_alignment, vec2u(x_grid_ceil, y_grid_floor), 0);
    let prev_align2 = downscale_factor * textureLoad(prev_alignment, vec2u(x_grid_floor, y_grid_ceil), 0);
    let prev_align3 = downscale_factor * textureLoad(prev_alignment, vec2u(x_grid_ceil, y_grid_ceil), 0);
    
    // alignment vector from tile 0
    var pixel_value: f32 = (1.0 - weight_x) * (1.0 - weight_y) * textureLoad(in_texture, vec2u(x + prev_align0.x, y + prev_align0.y), 0).r;
    var total_weight: f32 = (1.0 - weight_x) * (1.0 - weight_y);
    
    // alignment vector from tile 1
    pixel_value += weight_x * (1.0 - weight_y) * textureLoad(in_texture, vec2u(x + prev_align1.x, y + prev_align1.y), 0).r;
    total_weight += weight_x * (1.0 - weight_y);
    
    // alignment vector from tile 2
    pixel_value += (1.0 - weight_x) * weight_y * textureLoad(in_texture, vec2u(x + prev_align2.x, y + prev_align2.y), 0).r;
    total_weight += (1.0 - weight_x) * weight_y;
    
    // alignment vector from tile 3
    pixel_value += weight_x * weight_y * textureLoad(in_texture, vec2u(x + prev_align3.x, y + prev_align3.y), 0).r;
    total_weight += weight_x * weight_y;
    
    // write output pixel
    textureStore(out_texture, gid.xy, pixel_value / total_weight);
}


fn warp_texture_xtrans() {
    
    // load args
    let texture_width = in_texture.get_width();
    let texture_height = in_texture.get_height();

    let tile_half_size = tile_size / 2;
    
    // load coordinates of output pixel
    let x1_pix = gid.x;
    let y1_pix = gid.y;
    
    // compute the coordinates of output pixel in tile-grid units
    let x1_grid = f32(x1_pix - tile_half_size) / f32(texture_width - tile_size - 1) * (n_tiles_x - 1);
    let y1_grid = f32(y1_pix - tile_half_size) / f32(texture_height - tile_size - 1) * (n_tiles_y - 1);
    
    // compute the two possible tile-grid indices that the given output pixel belongs to
    let x_grid_list = vec4i(i32(floor(x1_grid)), i32(floor(x1_grid)), i32(ceil(x1_grid)), i32(ceil(x1_grid)));
    let y_grid_list = vec4i(i32(floor(y1_grid)), i32(ceil(y1_grid)), i32(floor(y1_grid)), i32(ceil(y1_grid)));
    
    // loop over the two possible tile-grid indices that the given output pixel belongs to
    var total_intensity = 0.0f;
    var total_weight = 0.0f;

    for (var i = 0; i < 4; i++) {
        // load the index of the tile
        let x_grid = x_grid_list[i];
        let y_grid = y_grid_list[i];
        
        // compute the pixel coordinates of the center of the reference tile
        let x0_pix = i32(floor(tile_half_size + f32(x_grid) / f32(n_tiles_x - 1) * (texture_width - tile_size - 1)));
        let y0_pix = i32(floor(tile_half_size + f32(y_grid) / f32(n_tiles_y - 1) * (texture_height - tile_size - 1)));
        
        // check that the output pixel falls within the reference tile
        if (abs(x1_pix - x0_pix) <= tile_half_size) && (abs(y1_pix - y0_pix) <= tile_half_size) {
            
            // compute tile displacement
            let prev_align = textureLoad(prev_alignment, vec2u(x_grid, y_grid));

            let dx = downscale_factor * prev_align.x;
            let dy = downscale_factor * prev_align.y;

            // load coordinates of the corresponding pixel from the comparison tile
            let x2_pix = x1_pix + dx;
            let y2_pix = y1_pix + dy;
            
            // compute the weight of the aligned pixel (based on distance from tile center)
            let dist_x = abs(x1_pix - x0_pix);
            let dist_y = abs(y1_pix - y0_pix);

            let weight_x = tile_size - dist_x - dist_y;
            let weight_y = tile_size - dist_x - dist_y;

            let curr_weight = weight_x * weight_y;

            total_weight += curr_weight;
            
            // add pixel value to the output
            total_intensity += curr_weight * textureLoad(in_texture, vec2u(x2_pix, y2_pix), 0).r;
        }
    }
    
    // write output pixel 
    textureStore(out_texture, vec2u(x1_pix, y1_pix), total_intensity / total_weight);
}
