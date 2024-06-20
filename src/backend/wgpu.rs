use std::borrow::Cow;

use wgpu::{
    CommandEncoderDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    Extent3d, Features, Instance, Limits, PowerPreference, Queue, RequestAdapterOptions,
    ShaderModule, ShaderModuleDescriptor, Texture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages,
};

use super::{Backend, ImageBackend};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no adapters")]
    NoAdapterFound,

    #[error("device request error: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
}

pub struct WgpuBackend {
    device: Device,
    queue: Queue,
    pipeline_table: WgpuShaderTable,
}

pub struct WgpuShaderTable {
    avg_pool: ComputePipeline,
}

impl WgpuShaderTable {
    pub fn new(device: &Device) -> Self {
        Self {
            avg_pool: create_compute_pipeline(
                device,
                "avg_pool",
                "Avg Pool",
                include_str!("./wgsl/avg-pool.wgsl"),
            ),
        }
    }
}

#[inline]
fn create_compute_pipeline(
    device: &Device,
    entry_point: &'static str,
    label: &'static str,
    data: &'static str,
) -> ShaderModule {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(data)),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point,
        compilation_options: Default::default(),
    })
}

impl WgpuBackend {
    pub async fn new() -> Result<Self, Error> {
        // Instantiates instance of WebGPU
        let instance = Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: true,
                ..Default::default()
            })
            .await
            .ok_or_else(|| Error::NoAdapterFound)?;

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_defaults(),
                },
                None,
            )
            .await?;

        Ok(Self {
            pipeline_table: WgpuShaderTable::new(&device),
            device,
            queue,
        })
    }
}

impl Backend for WgpuBackend {
    type ImageBuffer = Texture;
    type Error = wgpu::Error;
}

impl ImageBackend for WgpuBackend {
    fn add_texture(&self) {
        todo!()
    }

    fn add_texture_exposure(&self) {
        todo!()
    }

    fn add_texture_highlights(&self) {
        todo!()
    }

    fn add_texture_uint16(&self) {
        todo!()
    }

    fn add_texture_weighted(&self) {
        todo!()
    }

    fn blur_mosaic_texture(&self) {
        todo!()
    }

    fn calculate_weight_highlights(&self) {
        todo!()
    }

    fn convert_float_to_uint16(&self) {
        todo!()
    }

    fn convert_to_bayer(&self) {
        todo!()
    }

    fn convert_to_rgba(&self) {
        todo!()
    }

    fn copy_texture(&self) {
        todo!()
    }

    fn crop_texture(&self) {
        todo!()
    }

    fn divide_buffer(&self) {
        todo!()
    }

    fn sum_divide_buffer(&self) {
        todo!()
    }

    fn fill_with_zeros(&self) {
        todo!()
    }

    fn find_hotpixels_bayer(&self) {
        todo!()
    }

    fn find_hotpixels_xtrans(&self) {
        todo!()
    }

    fn normalize_texture(&self) {
        todo!()
    }

    fn prepare_texture_bayer(&self) {
        todo!()
    }

    fn sum_rect_columns_float(&self) {
        todo!()
    }

    fn sum_rect_columns_uint(&self) {
        todo!()
    }

    fn sum_row(&self) {
        todo!()
    }

    fn upsample_bilinear_float(
        &self,
        into: &mut Self::ImageBuffer,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn upsample_nearest_int(
        &self,
        into: &mut Self::ImageBuffer,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<(), Self::Error> {
        let size = into.size();

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline_table.avg_pool);
        cpass.set_bind_group(0, &bind_group_0, &[]);
        cpass.set_bind_group(1, &bind_group_1, &[]);

        cpass.insert_debug_marker("compute image::upsample_nearest_int");
        cpass.dispatch_workgroups(size.width, size.height, 1); // Number of cells to run, the (x,y,z) size of item being processed

        // Submits command encoder for processing
        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    fn avg_pool(
        &self,
        scale: u32,
        black_level_mean: f64,
        normalization: bool,
        color_factors3: Vec<f64>,
    ) -> Result<Self::ImageBuffer, Self::Error> {
        let size = Extent3d {
            width: self.width() / scale,
            height: self.height() / scale,
            depth_or_array_layers: 1,
        };

        let output_texture = self.device.create_texture(&TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // let texture_in_view = texture_in.create_view(&wgpu::TextureViewDescriptor::default());
        // queue.write_texture(
        //     texture_in.as_image_copy(),
        //     &texels,
        //     wgpu::ImageDataLayout {
        //         offset: 0,
        //         bytes_per_row: Some(128 * 4 * 4),
        //         rows_per_image: Some(128),
        //     },
        //     texture_extent,
        // );

        Ok(output_texture)
    }

    fn avg_pool_normalization(&self) {
        todo!()
    }
}
