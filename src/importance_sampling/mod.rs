use crate::low_level::out_bgl;
#[cfg(feature = "wip-features")]
use crate::Shader;
use std::mem;
#[cfg(feature = "wip-features")]
use std::ops::Add;
#[cfg(feature = "wip-features")]
use wgpu::Features;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferDescriptor,
    BufferUsages, CommandEncoder, Device,
};

#[cfg(feature = "wip-features")]
/// A shader for importance sampling. Based off reSTIR GI.
pub struct SpatialResampling;

#[cfg(feature = "wip-features")]
impl SpatialResampling {
    pub fn features() -> Features {
        // features required to interact
        Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
    }
    pub fn create_shader() -> Shader {
        Shader {
            base: include_str!("importance_sampler.wgsl")
                .to_string()
                .add(include_str!("shared.wgsl"))
                .add(include_str!("../bindings.wgsl")),
            #[cfg(debug_assertions)]
            label: "Importance Sampler",
        }
    }
}

struct Reservoir {
    _visible_point: [f32; 3],
    _visible_normal: u32,
    _sample_point: [f32; 3],
    _sample_normal: u32,
    _out_radiance: [f32; 3],
    _ty: u32,
    _roughness: f32,
    _pdf: f32,
    _w: f32,
    _m_valid: u32,
    _full_w: f32,
    _pad: [f32; 3],
}

struct Info {
    _emission: [f32; 4],
    _albedo: [f32; 4],
    _cam_loc: [f32; 4],
}

// Right now there are lots of different possible importance samplers so the parts of this will likely change
// Therefore access is restricted to this so that changes to this are not breaking changes
/// Buffers for storing information between executions, if `wip-features` is enabled then this
/// contains importance sampling data. It is recommended to create this instead of making it yourself
/// as it may change in the future but will not be regarded as sem-var breaking.
pub struct DataBuffers {
    current: Buffer,
    old: Buffer,
    info: Buffer,
}

impl DataBuffers {
    #[cfg(feature = "wip-features")]
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        Self::internal_new(device, width, height)
    }

    #[cfg(not(feature = "wip-features"))]
    pub fn new(device: &Device) -> Self {
        Self::internal_new(device, 0, 0)
    }

    fn internal_new(device: &Device, mut width: u32, mut height: u32) -> Self {
        width = width.max(1);
        height = height.max(1);
        let size = width as BufferAddress
            * height as BufferAddress
            * mem::size_of::<Reservoir>() as BufferAddress;
        let info_size = width as BufferAddress
            * height as BufferAddress
            * mem::size_of::<Info>() as BufferAddress;
        let current = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let old = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let info = device.create_buffer(&BufferDescriptor {
            label: None,
            size: info_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { current, old, info }
    }

    pub fn advance_frame(&self, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.current, 0, &self.old, 0, self.current.size());
        encoder.clear_buffer(&self.current, 0, None);
    }

    /// Exactly the same as [Self::advance_frame] but does not add temporal data
    ///
    /// Useful if temporal data breaks things, the camera is moving fast or the scene is being averaged over multiple frames
    pub fn advance_frame_no_temporal(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.current, 0, None);
        encoder.clear_buffer(&self.old, 0, None);
    }

    pub fn create_bind_group(
        &self,
        device: &Device,
        cam_buf: &Buffer,
        tex_view: &wgpu::TextureView,
        tex_normal_view: &wgpu::TextureView,
        tex_albedo_view: &wgpu::TextureView,
        tex_background_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let out_bgl = out_bgl(device);
        device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &out_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(cam_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(tex_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(tex_normal_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(tex_albedo_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(tex_background_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Buffer(self.current.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Buffer(self.old.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::Buffer(self.info.as_entire_buffer_binding()),
                },
            ],
        })
    }
}
