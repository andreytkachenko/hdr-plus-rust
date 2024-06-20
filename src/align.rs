pub struct Align<B: AlignBackend> {
    backend: B,
}

impl<B: AlignBackend> Align<B> {
    pub fn align_image(&mut self) {
        // initialize tile alignments
        // let alignment_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Sint, width: 1, height: 1, mipmapped: false);
        // alignment_descriptor.usage = [.shaderRead, .shaderWrite];
        // alignment_descriptor.storageMode = .private;
        // var prev_alignment = device.makeTexture(descriptor: alignment_descriptor)!;
        // device.makeTexture(descriptor: alignment_descriptor)!;

        let mut current_alignment = self.backend.create_image(1, 1);

        // current_alignment.label = "\(comp_texture.label!.components(separatedBy: ":")[0]): Current alignment Start"
        let mut tile_info = TileInfo {
            tile_size: 0,
            tile_size_merge: 0,
            search_dist: 0,
            n_tiles_x: 0,
            n_tiles_y: 0,
            n_pos_1d: 0,
            n_pos_2d: 0,
        };

        // build comparison pyramid
        let comp_pyramid = self.build_pyramid(
            comp_texture,
            downscale_factor_array,
            black_level_mean,
            color_factors3,
        );

        // align tiles
        for i in (0..downscale_factor_array.len()).rev() {
            // load layer params
            let tile_size = tile_size_array[i];
            let search_dist = search_dist_array[i];
            let ref_layer = ref_pyramid[i];
            let comp_layer = comp_pyramid[i];

            // calculate the number of tiles
            let n_tiles_x = ref_layer.width / (tile_size / 2) - 1;
            let n_tiles_y = ref_layer.height / (tile_size / 2) - 1;
            let n_pos_1d = 2 * search_dist + 1;
            let n_pos_2d = n_pos_1d * n_pos_1d;

            // store the values together in a struct to make it easier and more readable when passing between functions
            let tile_info = TileInfo {
                tile_size,
                tile_size_merge: 0,
                search_dist,
                n_tiles_x,
                n_tiles_y,
                n_pos_1d,
                n_pos_2d,
            };

            // resize previous alignment
            // - 'downscale_factor' has to be loaded from the *previous* layer since that is the layer that generated the current layer
            let mut downscale_facto = if i < downscale_factor_array.len() - 1 {
                downscale_factor_array[i + 1]
            } else {
                0
            };

            // upsample alignment vectors by a factor of 2
            let prev_alignment =
                upsample(current_alignment, n_tiles_x, n_tiles_y, NearestNeighbour);
            // prev_alignment.label = "\(comp_texture.label!.components(separatedBy: ":")[0]): Prev alignment \(i)";

            // compare three alignment vector candidates, which improves alignment at borders of moving object
            // see https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf for more details
            let prev_alignment = self.backend.correct_upsampling_error(
                ref_layer,
                comp_layer,
                prev_alignment,
                downscale_factor,
                uniform_exposure,
                (i != 0),
                tile_info,
            );

            // The parameter 'use_ssd' employed in correct_upsamling_error() and comute_tile_diff() specifies if the
            // calculated cost term shall be based on the absolute difference (L1 norm -> use_ssd = false) or on the
            // sum of squared difference (L2 norm -> use_ssd = true). The alignment is done differently depending on
            // the pyramid scale: for levels with reduced resolution, the L2 norm is calculated while for the highest
            // resolution level, the L1 norm is calculated. This choice is identical to the original publication.
            //
            // see https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf for more details

            // compute tile differences
            let tile_diff = self.backend.compute_tile_diff(
                ref_layer,
                comp_layer,
                prev_alignment,
                downscale_factor,
                uniform_exposure,
                (i != 0),
                tile_info,
            );

            let current_alignment = self.texture_like(prev_alignment);
            // current_alignment.label = "\(comp_texture.label!.components(separatedBy: ":")[0]): Current alignment \(i)";

            // find best tile alignment based on tile differences
            self.backend.find_best_tile_alignment(
                tile_diff,
                prev_alignment,
                current_alignment,
                downscale_factor,
                tile_info,
            );
        }

        // warp the aligned layer
        let aligned_texture = self.backend.warp_image(
            comp_texture,
            current_alignment,
            tile_info,
            downscale_factor_array[0],
        );

        return aligned_texture;
    }

    fn build_pyramid(
        input_image: &Image,
        downscale_factor_list: &[i32],
        black_level_mean: f64,
        color_factors3: &[f64],
    ) -> Vec<Image> {
        // iteratively resize the current layer in the pyramid
        let mut pyramid = vec![];

        for (i, &downscale_factor) in downscale_factor_list.into_iter().enumerate() {
            if i == 0 {
                // If color_factor is NOT available, a negative value will be set and normalization is deactivated.
                pyramid.push(self.backend.avg_pool(
                    input_image,
                    downscale_factor,
                    f64::max(0.0, black_level_mean),
                    color_factors3[0] > 0,
                    color_factors3,
                ));
            } else {
                pyramid.push(self.backend.avg_pool(
                    self.backend.blur(pyramid.last().unwrap(), 1, 2),
                    downscale_factor,
                    0.0,
                    false,
                    color_factors3,
                ));
            }
        }

        pyramid
    }
}
