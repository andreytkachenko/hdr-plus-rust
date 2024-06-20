
@group(0)
@binding(0)
var ref_texture_ft: texture_2d<f32>;

@group(0)
@binding(1)
var aligned_texture_ft: texture_2d<f32>;

@group(0)
@binding(2)
var out_texture_ft: texture_2d<f32>;

@group(0)
@binding(3)
var rms_texture: texture_2d<f32>;

@group(0)
@binding(3)
var mismatch_texture: texture_2d<f32>;

@group(0)
@binding(3)
var highlights_norm_texture: texture_2d<f32>;

const PI:f32 = 3.14159265358979323846264338328;
const VEC4_ZERO: vec4<f32> = vec4f(0.0, 0.0, 0.0, 0.0);

var<private> total_diff: array<f32, 49>;

struct MergeParams {
    robustness_norm: f32,
    read_noise: f32,
    max_motion_norm: f32,
    tile_size: u32,
}

@group(0) @binding(0) var<uniform> params: MergeParams;

///  This is the most important function required for the frequency-based merging approach. It is based on ideas from several publications:
///  - [Hasinoff 2016]: https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf
///  - [Monod 2021]: https://www.ipol.im/pub/art/2021/336/
///  - [Liba 2019]: https://graphics.stanford.edu/papers/night-sight-sigasia19/night-sight-sigasia19.pdf
///  - [Delbracio 2015]: https://openaccess.thecvf.com/content_cvpr_2015/papers/Delbracio_Burst_Deblurring_Removing_2015_CVPR_paper.pdf
@compute
fn merge_frequency_domain(@builtin(global_invocation_id) gid: vec3<u32>) {
    // combine estimated shot noise and read noise
    let noise_est = textureLoad(rms_texture, gid.xy) + params.read_noise;

    // normalize with tile size and robustness norm
    let noise_norm = noise_est * f32(tile_size * tile_size) * params.robustness_norm;

    // derive motion norm from mismatch texture to increase the noise reduction 
    // for small values of mismatch using a similar linear relationship as shown in Figure 9f in [Liba 2019]
    let mismatch = textureLoad(mismatch_texture, gid.xy).r;

    // for a smooth transition, the magnitude norm is weighted based on the mismatch
    let mismatch_weight = clamp(1.0 - 10.0 * (mismatch - 0.2), 0.0, 1.0);
    let motion_norm = clamp(params.max_motion_norm - (mismatch - 0.02) * (params.max_motion_norm - 1.0) / 0.15, 1.0, params.max_motion_norm);

    // extract correction factor for clipped highlights
    let highlights_norm = textureLoad(highlights_norm_texture, gid.xy).r;
    
    // compute tile positions from gid
    let m0 = gid.x * params.tile_size;
    let n0 = gid.y * params.tile_size;
    
    // pre-calculate factors for sine and cosine calculation
    let angle = -2.0 * PI / f32(tile_size);
    let shift_step_size = 1.0 / 6.0;

    // fill with zeros
    for (var i = 0; i < 49; i++) {
        total_diff[i] = 0.0;
    }
    
    // subpixel alignment based on the Fourier shift theorem: test shifts between -0.5 and +0.5 pixels specified on the pixel scale of each color channel, which corresponds to -1.0 and +1.0 pixels specified on the original pixel scale
    for (var dn = 0; dn < params.tile_size; dn++) {
        for (var dm = 0; dm < params.tile_size; dm++) {
            let m = 2 * (m0 + dm);
            let n = n0 + dn;
            
            // extract complex frequency data of reference tile and aligned comparison tile
            let refRe = textureLoad(ref_texture_ft, vec2<u32>(m + 0, n));
            let refIm = textureLoad(ref_texture_ft, vec2<u32>(m + 1, n));

            let alignedRe = textureLoad(aligned_texture_ft, vec2<u32>(m + 0, n));
            let alignedIm = textureLoad(aligned_texture_ft, vec2<u32>(m + 1, n));
            
            // test 7x7 discrete steps
            for (var i = 0; i < 49; i++) {
                // potential shift in pixels (specified on the pixel scale of each color channel)
                let shift_x = -0.5f + f32(i % 7) * shift_step_size;
                let shift_y = -0.5f + f32(i / 7) * shift_step_size;
                            
                // calculate coefficients for Fourier shift
                let coefRe = cos(angle * (f32(dm) * shift_x + f32(dn) * shift_y));
                let coefIm = sin(angle * (f32(dm) * shift_x + f32(dn) * shift_y));
                         
                // calculate complex frequency data of shifted tile
                let alignedRe2 = refRe - (coefRe * alignedRe - coefIm * alignedIm);
                let alignedIm2 = refIm - (coefIm * alignedRe + coefRe * alignedIm);

                let weight4 = alignedRe2 * alignedRe2 + alignedIm2 * alignedIm2;
                
                // add magnitudes of differences
                total_diff[i] += (weight4.x + weight4.y + weight4.z + weight4.w);
            }
        }
    }

    // find best shift (which has the lowest total difference)
    var best_diff: f32 = 1e20;
    var best_i: i32 = 0;

    for (var i = 0; i < 49; i++) {
        if total_diff[i] < best_diff {
            best_diff = total_diff[i];
            best_i = i;
        }
    }

    // extract best shifts
    let best_shift_x = -0.5 + f32(best_i % 7) * shift_step_size;
    let best_shift_y = -0.5 + f32(best_i / 7) * shift_step_size;

    // perform the merging of the reference tile and the aligned comparison tile
    for (var dn = 0; dn < params.tile_size; dn++) {
        for (var dm = 0; dm < params.tile_size; dm++) {

            let m = 2 * (m0 + dm);
            let n = n0 + dn;

            let re_coord = vec2<u32>(m + 0, n);
            let im_coord = vec2<u32>(m + 1, n);
            
            // extract complex frequency data of reference tile and aligned comparison tile
            let refRe = textureLoad(ref_texture_ft, re_coord);
            let refIm = textureLoad(ref_texture_ft, im_coord);

            let alignedRe = textureLoad(aligned_texture_ft, re_coord);
            let alignedIm = textureLoad(aligned_texture_ft, im_coord);
            
            // calculate coefficients for best Fourier shift
            let coefRe = cos(angle * (f32(dm) * best_shift_x + f32(dn) * best_shift_y));
            let coefIm = sin(angle * (f32(dm) * best_shift_x + f32(dn) * best_shift_y));
                
            // calculate complex frequency data of shifted tile
            let alignedRe2 = (coefRe * alignedRe - coefIm * alignedIm);
            let alignedIm2 = (coefIm * alignedRe + coefRe * alignedIm);
                       
            // increase merging weights for images with larger frequency magnitudes and decrease weights for lower magnitudes with the idea that larger magnitudes indicate images with higher sharpness
            // this approach is inspired by equation (3) in [Delbracio 2015]
            var magnitude_norm = 1.0f;
            
            // if we are not at the central frequency bin (zero frequency), if the mismatch is low and if the burst has a uniform exposure
            if dm + dn > 0 & mismatch < 0.3 & uniform_exposure == 1 {
                // calculate magnitudes of complex frequency data
                let refMag = sqrt(refRe * refRe + refIm * refIm);
                let alignedMag2 = sqrt(alignedRe2 * alignedRe2 + alignedIm2 * alignedIm2);
                
                // calculate ratio of magnitudes
                let ratio_mag = (alignedMag2.x + alignedMag2.y + alignedMag2.z + alignedMag2.w) / (refMag.x + refMag.y + refMay.z + refMag.w);
                     
                // calculate additional normalization factor that increases the merging weight for larger magnitudes and decreases weight for lower magnitudes
                magnitude_norm = mismatch_weight * clamp(ratio_mag * ratio_mag * ratio_mag * ratio_mag, 0.5, 3.0);
            }
            
            // calculation of merging weight by Wiener shrinkage as described in the section "Robust pairwise temporal merge" and equation (7) in [Hasinoff 2016] or in the section "Spatially varying temporal merging" and equation (7) and (9) in [Liba 2019] or in section "Pairwise Wiener Temporal Denoising" and equation (11) in [Monod 2021]
            // noise_norm corresponds to the original approach described in [Hasinoff 2016] and [Monod 2021]
            // motion_norm corresponds to the additional factor proposed in [Liba 2019]
            // magnitude_norm is based on ideas from [Delbracio 2015]
            // highlights_norm helps prevent clipped highlights from introducing color casts
            let weight4 = (refRe - alignedRe2) * (refRe - alignedRe2) + (refIm - alignedIm2) * (refIm - alignedIm2);
            let weight4 = weight4 / (weight4 + magnitude_norm * motion_norm * noise_norm * highlights_norm);
            
            // use the same weight for all color channels to reduce color artifacts as described in [Liba 2019]
            //weight = clamp(max(weight4[0], max(weight4[1], max(weight4[2], weight4[3]))), 0.0f, 1.0f);
            let min_weight = min(weight4.x, min(weight4.y, min(weight4.z, weight4.w)));
            let max_weight = max(weight4.x, max(weight4.y, max(weight4.z, weight4.w)));

            // instead of the maximum weight as described in the publication, use the mean value of the two central weight values, which removes the two extremes and thus should slightly increase robustness of the approach
            let weight = clamp(0.5 * (weight4.x + weight4.y + weight4.z + weight4.w - min_weight - max_weight), 0.0, 1.0);
            
            // apply pairwise merging of two tiles as described in equation (6) in [Hasinoff 2016] or equation (10) in [Monod 2021]
            let mergedRe = textureLoad(out_texture_ft, re_coord) + (1.0 - weight) * alignedRe2 + weight * refRe;
            let mergedIm = textureLoad(out_texture_ft, im_coord) + (1.0 - weight) * alignedIm2 + weight * refIm;

            textureStore(out_texture_ft, re_coord, mergedRe);
            textureStore(out_texture_ft, im_coord, mergedIm);
        }
    }
}

@compute
fn calculate_abs_diff_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {
    let abs_diff = abs(textureLoad(ref_texture, gid.xy) - textureLoad(aligned_texture, gid.xy));

    textureStore(abs_diff_texture, gid.xy, abs_diff);
}

@compute
fn calculate_highlights_norm_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {
    // set to 1.0, which does not apply any correction
    var clipped_highlights_norm: f32 = 1.0;
    
    // if the frame has no uniform exposure
    if exposure_factor > 1.001 {
        // compute tile positions from gid
        let x0 = gid.x * tile_size;
        let y0 = gid.y * tile_size;

        var pixel_value_max: f32;
        clipped_highlights_norm = 0.0;
        
        // calculate fraction of highlight pixels brighter than 0.5 of white level
        for (var dy = 0; dy < tile_size; dy++) {
            for (var dx = 0; dx < tile_size; dx++) {
                let pixel_value4 = textureLoad(aligned_texture, vec2<u32>(x0 + dx, y0 + dy));
                let pixel_value_max = max(pixel_value4.x, max(pixel_value4.y, max(pixel_value4.z, pixel_value4.w)));
                let pixel_value_max = (pixel_value_max - black_level_mean) * exposure_factor + black_level_mean;
          
                // ensure smooth transition of contribution of pixel values between 0.50 and 0.99 of the white level
                clipped_highlights_norm += clamp((pixel_value_max / white_level - 0.50) / 0.49, 0.0, 1.0);
            }
        }

        clipped_highlights_norm = clipped_highlights_norm / float(tile_size * tile_size);

        // transform into a correction for the merging formula
        clipped_highlights_norm = clamp((1.0 - clipped_highlights_norm) * (1.0 - clipped_highlights_norm), 0.04 / min(exposure_factor, 4.0), 1.0);
    }

    textureStore(highlights_norm_texture, gid.xy, clipped_highlights_norm);
}

@compute
fn calculate_mismatch_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {
        
    // compute tile positions from gid
    let x0 = gid.x * tile_size;
    let y0 = gid.y * tile_size;
    
    // use only estimated shot noise here
    let  noise_est = textureLoad(rms_texture, gid.xy);
    
    // estimate motion mismatch as the absolute difference of reference tile and comparison tile
    // see section "Spatially varying temporal merging" in https://graphics.stanford.edu/papers/night-sight-sigasia19/night-sight-sigasia19.pdf for more details
    // use a spatial support twice of the tile size used for merging
    
    // clamp at top/left border of image frame
    let x_start = max(0, x0 - tile_size / 2);
    let y_start = max(0, y0 - tile_size / 2);

    let dims = textureDimensions(abs_diff_texture);
        
    // clamp at bottom/right border of image frame
    let x_end = min(dims.x - 1, x0 + tile_size * 3 / 2);
    let y_end = min(dims.y - 1, y0 + tile_size * 3 / 2);
    
    // calculate shift for cosine window to shift to range 0 - (tile_size-1)
    let x_shift = -(x0 - tile_size / 2);
    let y_shift = -(y0 - tile_size / 2);
    
    // pre-calculate factors for sine and cosine calculation
    let angle = -2.0 * PI / f32(tile_size);

    var tile_diff = 0.0;
    var n_total = 0.0;

    for (var dy = y_start; dy < y_end; dy++) {
        for (var dx = x_start; dx < x_end; dx++) {
            // use modified raised cosine window to apply lower weights at outer regions of the patch
            let norm_cosine = (0.5 - 0.17 * cos(-angle * (f32(dx + x_shift) + 0.5))) * (0.5 - 0.17 * cos(-angle * (f32(dy + y_shift) + 0.5)));

            tile_diff += norm_cosine * textureLoad(abs_diff_texture, vec2<u32>(dx, dy));
            n_total += norm_cosine;
        }
    }

    tile_diff /= n_total;

    // calculation of mismatch ratio, which is different from the Wiener shrinkage proposed in the publication above (equation (8)). The quadratic terms of the Wiener shrinkage led to a strong separation of bright and dark pixels in the mismatch texture while mismatch should be (almost) independent of pixel brightness
    let mismatch4 = tile_diff / sqrt(0.5 * noise_est + 0.5 * noise_est / exposure_factor + 1.0);
    let mismatch = 0.25 * (mismatch4.x + mismatch4.y + mismatch4.z + mismatch4.w);

    textureStore(mismatch_texture, gid.xy, mismatch);
}


/// See section "Noise model and tiled approximation" 
///   in https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf 
///   or section "Noise Level Estimation" in https://www.ipol.im/pub/art/2021/336/
@compute
fn calculate_rms_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {
    // compute tile positions from gid
    let x0 = gid.x * params.tile_size;
    let y0 = gid.y * params.tile_size;
    
    // fill with zeros
    var noise_est = VEC4_ZERO;

    // use tile size merge here
    for (var dy = 0; dy < params.tile_size; dy++) {
        for (var dx = 0; dx < params.tile_size; dx++) {
            let data_noise = textureLoad(ref_texture, vec2<u32>(x0 + dx, y0 + dy));

            noise_est += data_noise * data_noise;
        }
    }

    noise_est = 0.25 * sqrt(noise_est) / f32(tile_size);
    textureStore(rms_texture, gid.xy, noise_est);
}


@compute
fn deconvolute_frequency_domain(@builtin(global_invocation_id) gid: vec3<u32>) {
    // compute tile positions from gid
    let m0 = gid.x * params.tile_size;
    let n0 = gid.y * params.tile_size;

    var cw: array<f32, 16>;
    
    // tile size-dependent gains used for the different frequencies
    if params.tile_size == 8 {
        cw[0] = 0.00; cw[1] = 0.02; cw[2] = 0.04; cw[3] = 0.08;
        cw[4] = 0.04; cw[5] = 0.08; cw[6] = 0.04; cw[7] = 0.02;
    } else if params.tile_size == 16 {
        cw[ 0] = 0.00; cw[ 1] = 0.01; cw[ 2] = 0.02; cw[ 3] = 0.03;
        cw[ 4] = 0.04; cw[ 5] = 0.06; cw[ 6] = 0.08; cw[ 7] = 0.06;
        cw[ 8] = 0.04; cw[ 9] = 0.06; cw[10] = 0.08; cw[11] = 0.06;
        cw[12] = 0.04; cw[13] = 0.03; cw[14] = 0.02; cw[15] = 0.01;
    }

    let mismatch = textureLoad(total_mismatch_texture, gid.xy).r;

    // for a smooth transition, the deconvolution is weighted based on the mismatch
    let mismatch_weight = clamp(1.0 - 10.0 * (mismatch - 0.2), 0.0, 1.0);

    let convRe = textureLoad(final_texture_ft, vec2<u32>(2 * m0 + 0, n0));
    let convIm = textureLoad(final_texture_ft, vec2<u32>(2 * m0 + 1, n0));
    let convMag = sqrt(convRe * convRe + convIm * convIm);
    let magnitude_zero = (convMag.x + convMag.y + convMag.z + convMag.w);

    for (var dn = 0; dn < params.tile_size; dn++) {
        for (var dm = 0; dm < params.tile_size; dm++) {
            if dm + dn > 0 & mismatch < 0.3f {
                let m = 2 * (m0 + dm);
                let n = n0 + dn;

                let convRe = textureLoad(final_texture_ft, vec2<u32>(m + 0, n));
                let convIm = textureLoad(final_texture_ft, vec2<u32>(m + 1, n));
                let convMag = sqrt(convRe * convRe + convIm * convIm);
                let magnitude = convMag.x + convMag.y + convMag.z + convMag.w;
                  
                // reduce the increase for frequencies with high magnitude
                // weight becomes 0 for ratio >= 0.05
                // weight becomes 1 for ratio <= 0.01
                let weight = mismatch_weight * clamp(1.25 - 25.0 * magnitude / magnitude_zero, 0.0, 1.0);

                let convRe = (1.0 + weight * cw[dm]) * (1.0 + weight * cw[dn]) * convRe;
                let convIm = (1.0 + weight * cw[dm]) * (1.0 + weight * cw[dn]) * convIm;

                textureStore(final_texture_ft, vec2u(m + 0, n), convRe);
                textureStore(final_texture_ft, vec2u(m + 1, n), convIm);
            }
        }
    }
}

@compute
fn normalize_mismatch(@builtin(global_invocation_id) gid: vec3<u32>) {
    // load args
    let mean_mismatch = mean_mismatch_buffer[0];
    var mismatch_norm = textureLoad(mismatch_texture, gid.xy).r;
    
    // normalize that mean value of mismatch texture is set to 0.12, which is close to the threshold value of 0.17. For values larger than the threshold, the strength of temporal denoising is not increased anymore
    let mismatch_norm = mismatch_norm * (0.12 / (mean_mismatch + 1e-12));
    
    // clamp to range of 0 to 1 to remove very large values
    let mismatch_norm = clamp(mismatch_norm, 0.0, 1.0);

    textureStore(mismatch_texture, gid.xy, mismatch_norm);
}

@compute
fn reduce_artifacts_tile_border(@builtin(global_invocation_id) gid: vec3<u32>) {
    
    // compute tile positions from gid
    let x0 = gid.x * params.tile_size;
    let y0 = gid.y * params.tile_size;
 
    // set min values and max values
    let min_values = vec4<f32>(black_level0 - 1.0, black_level1 - 1.0, black_level2 - 1.0, black_level3 - 1.0);
    let max_values = vec4<f32>(f32(UINT16_MAX_VAL), f32(UINT16_MAX_VAL), f32(UINT16_MAX_VAL), f32(UINT16_MAX_VAL));
    
    // pre-calculate factors for sine and cosine calculation
    let angle = -2.0 * PI / f32(params.tile_size);

    for (var dy = 0; dy < params.tile_size; dy++) {
        for (var dx = 0; dx < params.tile_size; dx++) {
            let x = x0 + dx;
            let y = y0 + dy;
            
            // see section "Overlapped tiles" 
            //    in https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf 
            //    or section "Overlapped Tiles and Raised Cosine Window" in https://www.ipol.im/pub/art/2021/336/
            // calculate modified raised cosine window weight for blending tiles to suppress artifacts
            let norm_cosine = (0.5 - 0.5 * cos(-angle * (f32(dx) + 0.5))) * (0.5 - 0.5 * cos(-angle * (f32(dy) + 0.5)));
            
            // extract RGBA pixel values
            var pixel_value = textureLoad(out_texture, vec2<u32>(x, y));

            // clamp values, which reduces potential artifacts (black lines) at tile borders by removing pixels with negative entries (negative when black level is subtracted)
            pixel_value = clamp(pixel_value, norm_cosine * min_values, max_values);
            
            // blend pixel values at tile borders with reference texture
            if dx == 0 | dx == params.tile_size - 1 | dy == 0 | dy == params.tile_size - 1 {
                pixel_value = 0.5 * (norm_cosine * textureLoad(ref_texture, vec2u(x, y)) + pixel_value);
            }

            textureStore(out_texture, vec2y(x, y), pixel_value)
        }
    }
}

@compute
//  Simple and slow discrete Fourier transform applied to each color channel independently
fn backward_dft(@builtin(global_invocation_id) gid: vec3u) {
    
    // compute tile positions from gid
    let m0 = gid.x * params.tile_size;
    let n0 = gid.y * params.tile_size;
    
    // pre-calculate factors for sine and cosine calculation
    let angle = 2.0 * PI / f32(params.tile_size);
    
    // pre-initalize some vectors
    let n_elemns = f32(params.n_textures * params.tile_size * params.tile_size)
    let norm_factor = vec4<f32>(n_elemns, n_elemns, n_elemns, n_elemns);

    
    // row-wise one-dimensional discrete Fourier transform along x-direction
    for (var dn = 0; dn < params.tile_size; dn++) {
        for (var dm = 0; dm < params.tile_size; dm++) {

            let m = 2 * (m0 + dm);
            let n = n0 + dn;
            
            // fill with zeros
            var Re = zeros;
            var Im = zeros;

            for (var dx = 0; dx < tile_size; dx++) {
                let x = 2 * (m0 + dx);
              
                // calculate coefficients
                let coefRe = cos(angle * f32(dm * dx));
                let coefIm = sin(angle * f32(dm * dx));

                let dataRe = textureLoad(in_texture_ft, vec2u(x + 0, n));
                let dataIm = textureLoad(in_texture_ft, vec2u(x + 1, n));

                Re += (coefRe * dataRe - coefIm * dataIm);
                Im += (coefIm * dataRe + coefRe * dataIm);
            }
            
            // write into temporary textures
            textureStore(tmp_texture_ft, vec2u(m + 0, n), Re);
            textureStore(tmp_texture_ft, vec2u(m + 1, n), Im);
        }
    }
    
    // column-wise one-dimensional discrete Fourier transform along y-direction
    for (var dm = 0; dm < tile_size; dm++) {
        for (var dn = 0; dn < tile_size; dn++) {
            let  m = m0 + dm;
            let  n = n0 + dn;
             
            // fill with zeros
            var Re = VEC4_ZERO;

            for (var dy = 0; dy < tile_size; dy++) {
                let y = n0 + dy;
                
                // calculate coefficients
                let coefRe = cos(angle * f32(dn * dy));
                let coefIm = sin(angle * f32(dn * dy));

                let dataRe = textureLoad(tmp_texture_ft, vec2u(2 * m + 0, y));
                let dataIm = textureLoad(tmp_texture_ft, vec2u(2 * m + 1, y));

                Re += (coefRe * dataRe - coefIm * dataIm);
            }
            
            // normalize result
            Re = Re / norm_factor;

            textureStore(out_texture, vec2u(m, n), Re);
        }
    }
}

/// Highly-optimized fast Fourier transform applied to each color channel independently
/// The aim of this function is to provide improved performance compared to the more simple function backward_dft() while providing equal results. It uses the following features for reduced calculation times:
/// - the four color channels are stored as a float4 and all calculations employ SIMD instructions.
/// - the one-dimensional transformation along y-direction employs the fast Fourier transform algorithm: At first, 4 small DFTs are calculated and then final results are obtained by two steps of cross-combination of values (based on a so-called butterfly diagram). This approach reduces the total number of memory reads and computational steps considerably.
/// - the one-dimensional transformation along x-direction employs the fast Fourier transform algorithm: At first, 4 small DFTs are calculated and then final results are obtained by two steps of cross-combination of values (based on a so-called butterfly diagram). This approach reduces the total number of memory reads and computational steps considerably.
fn backward_fft(@builtin(global_invocation_id) gid: vec3u) {
    
    // compute tile positions from gid
    let  m0 = gid.x * params.tile_size;
    let  n0 = gid.y * params.tile_size;

    let  tile_size_14 = params.tile_size / 4;
    let  tile_size_24 = params.tile_size / 2;
    let  tile_size_34 = params.tile_size / 4 * 3;
    
    // pre-calculate factors for sine and cosine calculation
    let angle = -2.0 * PI / f32(params.tile_size);
    
    // pre-initalize some vectors
    let n_elemns = f32(params.n_textures * params.tile_size * params.tile_size)
    let norm_factor = vec4f(n_elemns, n_elemns, n_elemns, n_elemns);

    var tmp_data: array<f32, 16>;
    var tmp_tile: array<f32, 128>;
    
    // row-wise one-dimensional fast Fourier transform along x-direction
    for (var dn = 0; dn < params.tile_size; dn++) {
        let n_tmp = dn * 2 * params.tile_size;
        
        // copy data to temp vector
        for (var dm = 0; dm < tile_size; dm++) {
            tmp_data[2 * dm + 0] = textureLoad(in_texture_ft, vec2u(2 * (m0 + dm) + 0, n0 + dn));
            tmp_data[2 * dm + 1] = textureLoad(in_texture_ft, vec2u(2 * (m0 + dm) + 1, n0 + dn));
        }
        
        // calculate 4 small discrete Fourier transforms
        for (var dm = 0; dm < tile_size / 4; dm++) {
            // fill with zeros
            var Re0 = VEC4_ZERO;
            var Im0 = VEC4_ZERO;
            var Re1 = VEC4_ZERO;
            var Im1 = VEC4_ZERO;
            var Re2 = VEC4_ZERO;
            var Im2 = VEC4_ZERO;
            var Re3 = VEC4_ZERO;
            var Im3 = VEC4_ZERO;

            for (var dx = 0; dx < tile_size; dx += 4) {
                
                // calculate coefficients
                let coefRe = cos(angle * f32(dm * dx));
                let coefIm = sin(angle * f32(dm * dx));
                
                // DFT0
                let dataRe = tmp_data[2 * dx + 0];
                let dataIm = tmp_data[2 * dx + 1];

                Re0 += (coefRe * dataRe + coefIm * dataIm);
                Im0 += (coefIm * dataRe - coefRe * dataIm);

                // DFT1
                let dataRe = tmp_data[2 * dx + 2];
                let dataIm = tmp_data[2 * dx + 3];

                Re2 += (coefRe * dataRe + coefIm * dataIm);
                Im2 += (coefIm * dataRe - coefRe * dataIm);

                // DFT2
                let dataRe = tmp_data[2 * dx + 4];
                let dataIm = tmp_data[2 * dx + 5];

                Re1 += (coefRe * dataRe + coefIm * dataIm);
                Im1 += (coefIm * dataRe - coefRe * dataIm);
                
                //DFT3
                let dataRe = tmp_data[2 * dx + 6];
                let dataIm = tmp_data[2 * dx + 7];
                Re3 += (coefRe * dataRe + coefIm * dataIm);
                Im3 += (coefIm * dataRe - coefRe * dataIm);
            }
            
            // first butterfly to combine results
            let coefRe = cos(angle * 2.0 * f32(dm));
            let coefIm = sin(angle * 2.0 * f32(dm));

            let Re00 = Re0 + coefRe * Re1 - coefIm * Im1;
            let Im00 = Im0 + coefIm * Re1 + coefRe * Im1;
            let Re22 = Re2 + coefRe * Re3 - coefIm * Im3;
            let Im22 = Im2 + coefIm * Re3 + coefRe * Im3;

            let coefRe = cos(angle * 2.0 * (f32(dm) + tile_size_14));
            let coefIm = sin(angle * 2.0 * (f32(dm) + tile_size_14));

            let Re11 = Re0 + coefRe * Re1 - coefIm * Im1;
            let Im11 = Im0 + coefIm * Re1 + coefRe * Im1;
            let Re33 = Re2 + coefRe * Re3 - coefIm * Im3;
            let Im33 = Im2 + coefIm * Re3 + coefRe * Im3;
            
            // second butterfly to combine results
            Re0 = Re00 + cos(angle * f32(dm)) * Re22 - sin(angle * f32(dm)) * Im22;
            Im0 = Im00 + sin(angle * f32(dm)) * Re22 + cos(angle * f32(dm)) * Im22;

            Re2 = Re00 + cos(angle * f32(dm + tile_size_24)) * Re22 - sin(angle * f32(dm + tile_size_24)) * Im22;
            Im2 = Im00 + sin(angle * f32(dm + tile_size_24)) * Re22 + cos(angle * f32(dm + tile_size_24)) * Im22;
            Re1 = Re11 + cos(angle * f32(dm + tile_size_14)) * Re33 - sin(angle * f32(dm + tile_size_14)) * Im33;
            Im1 = Im11 + sin(angle * f32(dm + tile_size_14)) * Re33 + cos(angle * f32(dm + tile_size_14)) * Im33;
            Re3 = Re11 + cos(angle * f32(dm + tile_size_34)) * Re33 - sin(angle * f32(dm + tile_size_34)) * Im33;
            Im3 = Im11 + sin(angle * f32(dm + tile_size_34)) * Re33 + cos(angle * f32(dm + tile_size_34)) * Im33;
            
            // write into temporary tile storage
            tmp_tile[n_tmp + 2 * dm + 0] = Re0;
            tmp_tile[n_tmp + 2 * dm + 1] = -Im0;
            tmp_tile[n_tmp + 2 * dm + tile_size_24 + 0] = Re1;
            tmp_tile[n_tmp + 2 * dm + tile_size_24 + 1] = -Im1;
            tmp_tile[n_tmp + 2 * dm + tile_size + 0] = Re2;
            tmp_tile[n_tmp + 2 * dm + tile_size + 1] = -Im2;
            tmp_tile[n_tmp + 2 * dm + tile_size_24 * 3 + 0] = Re3;
            tmp_tile[n_tmp + 2 * dm + tile_size_24 * 3 + 1] = -Im3;
        }
    }
  
    // column-wise one-dimensional fast Fourier transform along y-direction
    for (var dm = 0; dm < params.tile_size; dm++) {
        let m = m0 + dm;
        
        // copy data to temp vector
        for (var dn = 0; dn < params.tile_size; dn++) {
            tmp_data[2 * dn + 0] = tmp_tile[dn * 2 * params.tile_size + 2 * dm + 0];
            tmp_data[2 * dn + 1] = tmp_tile[dn * 2 * params.tile_size + 2 * dm + 1];
        }
        
        // calculate 4 small discrete Fourier transforms
        for (var dn = 0; dn < params.tile_size / 4; dn++) {
            let n = n0 + dn;
            
            // fill with zeros
            var Re0 = VEC4_ZERO;
            var Im0 = VEC4_ZERO;
            var Re1 = VEC4_ZERO;
            var Im1 = VEC4_ZERO;
            var Re2 = VEC4_ZERO;
            var Im2 = VEC4_ZERO;
            var Re3 = VEC4_ZERO;
            var Im3 = VEC4_ZERO;

            for (var dy = 0; dy < params.tile_size; dy += 4) {
              
                // calculate coefficients
                let coefRe = cos(angle * f32(dn * dy));
                let coefIm = sin(angle * f32(dn * dy));
                
                // DFT0
                let dataRe = tmp_data[2 * dy + 0];
                let dataIm = tmp_data[2 * dy + 1];
                Re0 += (coefRe * dataRe + coefIm * dataIm);
                Im0 += (coefIm * dataRe - coefRe * dataIm);
                
                // DFT1
                let dataRe = tmp_data[2 * dy + 2];
                let dataIm = tmp_data[2 * dy + 3];
                Re2 += (coefRe * dataRe + coefIm * dataIm);
                Im2 += (coefIm * dataRe - coefRe * dataIm);
                
                // DFT2
                let dataRe = tmp_data[2 * dy + 4];
                let dataIm = tmp_data[2 * dy + 5];
                Re1 += (coefRe * dataRe + coefIm * dataIm);
                Im1 += (coefIm * dataRe - coefRe * dataIm);
                
                // DFT3
                let dataRe = tmp_data[2 * dy + 6];
                let dataIm = tmp_data[2 * dy + 7];
                Re3 += (coefRe * dataRe + coefIm * dataIm);
                Im3 += (coefIm * dataRe - coefRe * dataIm);
            }
            
            // first butterfly to combine results
            let coefRe = cos(angle * 2.0 * f32(dn));
            let coefIm = sin(angle * 2.0 * f32(dn));

            let Re00 = Re0 + coefRe * Re1 - coefIm * Im1;
            let Im00 = Im0 + coefIm * Re1 + coefRe * Im1;
            let Re22 = Re2 + coefRe * Re3 - coefIm * Im3;
            let Im22 = Im2 + coefIm * Re3 + coefRe * Im3;

            let coefRe = cos(angle * 2.0 * (f32(dn) + tile_size_14));
            let coefIm = sin(angle * 2.0 * (f32(dn) + tile_size_14));

            let Re11 = Re0 + coefRe * Re1 - coefIm * Im1;
            let Im11 = Im0 + coefIm * Re1 + coefRe * Im1;
            let Re33 = Re2 + coefRe * Re3 - coefIm * Im3;
            let Im33 = Im2 + coefIm * Re3 + coefRe * Im3;
            
            // second butterfly to combine results
            Re0 = Re00 + cos(angle * f32(dn)) * Re22 - sin(angle * f32(dn)) * Im22;
            Re2 = Re00 + cos(angle * f32(dn + tile_size_24)) * Re22 - sin(angle * f32(dn + tile_size_24)) * Im22;
            Re1 = Re11 + cos(angle * f32(dn + tile_size_14)) * Re33 - sin(angle * f32(dn + tile_size_14)) * Im33;
            Re3 = Re11 + cos(angle * f32(dn + tile_size_34)) * Re33 - sin(angle * f32(dn + tile_size_34)) * Im33;
                      
            // write into output textures
            textureStore(out_texture, vec2u(m, n), Re0 / norm_factor);
            textureStore(out_texture, vec2u(m, n + tile_size_14), Re1 / norm_factor);
            textureStore(out_texture, vec2u(m, n + tile_size_24), Re2 / norm_factor);
            textureStore(out_texture, vec2u(m, n + tile_size_34), Re3 / norm_factor);
        }
    }
}

// Simple and slow discrete Fourier transform applied to each color channel independently
fn forward_dft(@builtin(global_invocation_id) gid: vec3<u32>) {
    
    // compute tile positions from gid
    let m0 = gid.x * tile_size;
    let n0 = gid.y * tile_size;
        
    // pre-calculate factors for sine and cosine calculation
    let angle = -2.0 * PI / f32(tile_size);
    
    // column-wise one-dimensional discrete Fourier transform along y-direction
    for (var dm = 0; dm < tile_size; dm++) {
        for (var dn = 0; dn < tile_size; dn++) {

            let m = m0 + dm;
            let n = n0 + dn;
            
            // fill with zeros
            var Re = VEC4_ZERO;
            var Im = VEC4_ZERO;

            for (var dy = 0; dy < tile_size; dy++) {

                let y = n0 + dy;
                
                // see section "Overlapped tiles" 
                //    in https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf 
                //    or section "Overlapped Tiles and Raised Cosine Window" in https://www.ipol.im/pub/art/2021/336/
                // calculate modified raised cosine window weight for blending tiles to suppress artifacts
                let norm_cosine = (0.5 - 0.5 * cos(-angle * (dm + 0.5))) * (0.5 - 0.5 * cos(-angle * (dy + 0.5)));
                                
                // calculate coefficients
                let coefRe = cos(angle * f32(dn * dy));
                let coefIm = sin(angle * f32(dn * dy));

                let dataRe = norm_cosine * textureLoad(textin_texture, vec2u(m, y));

                Re += coefRe * dataRe;
                Im += coefIm * dataRe;
            }
            
            // write into temporary textures
            textureStore(tmp_texture_ft, vec2u(2 * m + 0, n), Re);
            textureStore(tmp_texture_ft, vec2u(2 * m + 1, n), Im);
        }
    }
    
    // row-wise one-dimensional discrete Fourier transform along x-direction
    for (var dn = 0; dn < tile_size; dn++) {
        for (var dm = 0; dm < tile_size; dm++) {
            let m = 2 * (m0 + dm);
            let n = n0 + dn;
             
            // fill with zeros
            var Re = zeros;
            var Im = zeros;

            for (var dx = 0; dx < tile_size; dx++) {
                let x = 2 * (m0 + dx);
                
                // calculate coefficients
                let coefRe = cos(angle * dm * dx);
                let coefIm = sin(angle * dm * dx);

                let dataRe = textureLoad(tmp_texture_ft, vec2u(x + 0, n));
                let dataIm = textureLoad(tmp_texture_ft, vec2u(x + 1, n));

                Re += coefRe * dataRe - coefIm * dataIm;
                Im += coefIm * dataRe + coefRe * dataIm;
            }

            textureStore(out_texture_ft, vec2u(m + 0, n), Re);
            textureStore(out_texture_ft, vec2u(m + 1, n), Im);
        }
    }
}

/// Highly-optimized fast Fourier transform applied to each color channel independently
/// The aim of this function is to provide improved performance compared to the more simple function forward_dft() while providing equal results. It uses the following features for reduced calculation times:
/// - the four color channels are stored as a float4 and all calculations employ SIMD instructions.
/// - the one-dimensional transformation along y-direction is a discrete Fourier transform. As the input image is real-valued, the frequency domain representation is symmetric and only values for N/2+1 rows have to be calculated.
/// - the one-dimensional transformation along x-direction employs the fast Fourier transform algorithm: At first, 4 small DFTs are calculated and then final results are obtained by two steps of cross-combination of values (based on a so-called butterfly diagram). This approach reduces the total number of memory reads and computational steps considerably.
/// - due to the symmetry mentioned earlier, only N/2+1 rows have to be transformed and the remaining N/2-1 rows can be directly inferred.
fn forward_fft(@builtin(global_invocation_id) gid: vec3u) {
    
    // compute tile positions from gid
    let  m0 = gid.x * tile_size;
    let  n0 = gid.y * tile_size;

    let  tile_size_14 = tile_size / 4;
    let  tile_size_24 = tile_size / 2;
    let  tile_size_34 = tile_size / 4 * 3;
    
    // pre-calculate factors for sine and cosine calculation
    let angle = -2 * PI / f32(params.tile_size);

    var tmp_data: array<f32, 16>;
    var tmp_tile: array<f32, 80>;
    
    // column-wise one-dimensional discrete Fourier transform along y-direction
    for (var dm = 0; dm < params.tile_size; dm += 2) {

        let m = m0 + dm;
        
        // copy data to temp vector
        for (var dn = 0; dn < params.tile_size; dn++) {
            tmp_data[2 * dn + 0] = textureLoad(in_texture, vec2u(m + 0, n0 + dn));
            tmp_data[2 * dn + 1] = textureLoad(in_texture, vec2u(m + 1, n0 + dn));
        }
        
        // exploit symmetry of real dft and calculate reduced number of rows
        for (var dn = 0; dn <= params.tile_size / 2; dn++) {

            let n_tmp = dn * 2 * params.tile_size;
            
            // fill with zeros
            var Re0 = zeros;
            var Im0 = zeros;
            var Re1 = zeros;
            var Im1 = zeros;

            for (var dy = 0; dy < params.tile_size; dy++) {
      
                // see section "Overlapped tiles" 
                //    in https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf 
                // or section "Overlapped Tiles and Raised Cosine Window" 
                //    in https://www.ipol.im/pub/art/2021/336/
                // 
                // calculate modified raised cosine window weight for blending tiles to suppress artifacts
                let norm_cosine0 = (0.5 - 0.5 * cos(-angle * (f32(dm) + 0.5))) * (0.5 - 0.5 * cos(-angle * (f32(dy) + 0.5)));
                let norm_cosine1 = (0.5 - 0.5 * cos(-angle * (f32(dm) + 1.5))) * (0.5 - 0.5 * cos(-angle * (f32(dy) + 0.5)));
                         
                // calculate coefficients
                let coefRe = cos(angle * f32(dn * dy));
                let coefIm = sin(angle * f32(dn * dy));

                let dataRe = norm_cosine0 * tmp_data[2 * dy + 0];
                Re0 += coefRe * dataRe;
                Im0 += coefIm * dataRe;

                let dataRe = norm_cosine1 * tmp_data[2 * dy + 1];
                Re1 += coefRe * dataRe;
                Im1 += coefIm * dataRe;
            }
            
            // write into temporary tile storage
            tmp_tile[n_tmp + 2 * dm + 0] = Re0;
            tmp_tile[n_tmp + 2 * dm + 1] = Im0;
            tmp_tile[n_tmp + 2 * dm + 2] = Re1;
            tmp_tile[n_tmp + 2 * dm + 3] = Im1;
        }
    }
        
    // row-wise one-dimensional fast Fourier transform along x-direction
    // exploit symmetry of real dft and calculate reduced number of rows
    for (var dn = 0; dn <= tile_size / 2; dn++) {
        let n = n0 + dn;
        
        // copy data to temp vector
        for (var dm = 0; dm < tile_size; dm++) {
            tmp_data[2 * dm + 0] = tmp_tile[dn * 2 * tile_size + 2 * dm + 0];
            tmp_data[2 * dm + 1] = tmp_tile[dn * 2 * tile_size + 2 * dm + 1];
        }
        
        // calculate 4 small discrete Fourier transforms
        for (var dm = 0; dm < tile_size / 4; dm++) {

            let m = 2 * (m0 + dm);
            
            // fill with zeros

            var Re0 = VEC4_ZERO;
            var Im0 = VEC4_ZERO;
            var Re1 = VEC4_ZERO;
            var Im1 = VEC4_ZERO;
            var Re2 = VEC4_ZERO;
            var Im2 = VEC4_ZERO;
            var Re3 = VEC4_ZERO;
            var Im3 = VEC4_ZERO;

            for (var dx = 0; dx < tile_size; dx += 4) {
              
                // calculate coefficients
                let coefRe = cos(angle * f32(dm * dx));
                let coefIm = sin(angle * f32(dm * dx));
                                
                // DFT0
                let dataRe = tmp_data[2 * dx + 0];
                let dataIm = tmp_data[2 * dx + 1];
                Re0 += (coefRe * dataRe - coefIm * dataIm);
                Im0 += (coefIm * dataRe + coefRe * dataIm);
                
                // DFT1
                let dataRe = tmp_data[2 * dx + 2];
                let dataIm = tmp_data[2 * dx + 3];
                Re2 += (coefRe * dataRe - coefIm * dataIm);
                Im2 += (coefIm * dataRe + coefRe * dataIm);
                
                // DFT2
                let dataRe = tmp_data[2 * dx + 4];
                let dataIm = tmp_data[2 * dx + 5];
                Re1 += (coefRe * dataRe - coefIm * dataIm);
                Im1 += (coefIm * dataRe + coefRe * dataIm);
                
                // DFT3
                let dataRe = tmp_data[2 * dx + 6];
                let dataIm = tmp_data[2 * dx + 7];
                Re3 += (coefRe * dataRe - coefIm * dataIm);
                Im3 += (coefIm * dataRe + coefRe * dataIm);
            }
            
            // first butterfly to combine results
            let coefRe = cos(angle * 2.0 * f32(dm));
            let coefIm = sin(angle * 2.0 * f32(dm));

            let Re00 = Re0 + coefRe * Re1 - coefIm * Im1;
            let Im00 = Im0 + coefIm * Re1 + coefRe * Im1;
            let Re22 = Re2 + coefRe * Re3 - coefIm * Im3;
            let Im22 = Im2 + coefIm * Re3 + coefRe * Im3;

            let coefRe = cos(angle * 2.0 * f32(dm + tile_size_14));
            let coefIm = sin(angle * 2.0 * f32(dm + tile_size_14));

            let Re11 = Re0 + coefRe * Re1 - coefIm * Im1;
            let Im11 = Im0 + coefIm * Re1 + coefRe * Im1;
            let Re33 = Re2 + coefRe * Re3 - coefIm * Im3;
            let Im33 = Im2 + coefIm * Re3 + coefRe * Im3;
                    
            // second butterfly to combine results
            Re0 = Re00 + cos(angle * f32(dm)) * Re22 - sin(angle * f32(dm)) * Im22;
            Im0 = Im00 + sin(angle * f32(dm)) * Re22 + cos(angle * f32(dm)) * Im22;
            Re2 = Re00 + cos(angle * f32(dm + tile_size_24)) * Re22 - sin(angle * f32(dm + tile_size_24)) * Im22;
            Im2 = Im00 + sin(angle * f32(dm + tile_size_24)) * Re22 + cos(angle * f32(dm + tile_size_24)) * Im22;
            Re1 = Re11 + cos(angle * f32(dm + tile_size_14)) * Re33 - sin(angle * f32(dm + tile_size_14)) * Im33;
            Im1 = Im11 + sin(angle * f32(dm + tile_size_14)) * Re33 + cos(angle * f32(dm + tile_size_14)) * Im33;
            Re3 = Re11 + cos(angle * f32(dm + tile_size_34)) * Re33 - sin(angle * f32(dm + tile_size_34)) * Im33;
            Im3 = Im11 + sin(angle * f32(dm + tile_size_34)) * Re33 + cos(angle * f32(dm + tile_size_34)) * Im33;
                           
            // write into output texture
            textureStore(out_texture_ft, vec2u(m + 0, n), Re0);
            textureStore(out_texture_ft, vec2u(m + 1, n), Im0);
            textureStore(out_texture_ft, vec2u(m + tile_size_24 + 0, n), Re1);
            textureStore(out_texture_ft, vec2u(m + tile_size_24 + 1, n), Im1);
            textureStore(out_texture_ft, vec2u(m + tile_size + 0, n), Re2);
            textureStore(out_texture_ft, vec2u(m + tile_size + 1, n), Im2);
            textureStore(out_texture_ft, vec2u(m + tile_size_24 * 3 + 0, n), Re3);
            textureStore(out_texture_ft, vec2u(m + tile_size_24 * 3 + 1, n), Im3);
              
            // exploit symmetry of real dft and set values for remaining rows
            if dn > 0 & dn != tile_size / 2 {
                let n2 = n0 + tile_size - dn;

                //int const m20 = 2*(m0 + (dm==0 ? 0 : tile_size-dm));
                let m20 = 2 * (m0 + min(dm, 1) * (tile_size - dm));
                let m21 = 2 * (m0 + tile_size - dm - tile_size_14);
                let m22 = 2 * (m0 + tile_size - dm - tile_size_24);
                let m23 = 2 * (m0 + tile_size - dm - tile_size_14 * 3);
                
                // write into output texture
                textureStore(out_texture_ft, vec2u(m20 + 0, n2), Re0);
                textureStore(out_texture_ft, vec2u(m20 + 1, n2), -Im0);
                textureStore(out_texture_ft, vec2u(m21 + 0, n2), Re1);
                textureStore(out_texture_ft, vec2u(m21 + 1, n2), -Im1);
                textureStore(out_texture_ft, vec2u(m22 + 0, n2), Re2);
                textureStore(out_texture_ft, vec2u(m22 + 1, n2), -Im2);
                textureStore(out_texture_ft, vec2u(m23 + 0, n2), Re3);
                textureStore(out_texture_ft, vec2u(m23 + 1, n2), -Im3);
            }
        }
    }
}
