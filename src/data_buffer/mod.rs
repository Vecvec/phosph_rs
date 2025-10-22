use crate::low_level::out_bgl;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferDescriptor,
    BufferUsages, CommandEncoder, Device,
};

bitflags::bitflags! {
    pub struct BufferType: u16 {
        // Need some of the world space stuff for the
        const MARKOV_CHAIN_SCREEN_SPACE = 1 << 0;
        #[cfg(feature = "wip-features")]
        const SPATIAL_RESAMPLING = 1 << 1;
        const MARKOV_CHAIN_WORLD_SPACE = 1 << 2;
    }
}

/// Matches markov chain in bindings.
#[repr(C)]
struct MarkovChain {
    _light_source: [f32; 3],
    _mean_cosine: f32,
    _weight_sum: f32,
    _num_samples: u32,
    _score: f32,
    _align: u32,
}

#[repr(C)]
struct WorldMarkovStorage {
    _secondary_hash: u32,
    _num_samples: u32,
    _radiance: [u32; 3],
    _lock: u32,
    _align: [u32; 2],
    _chain: MarkovChain,
}

struct TemporalBuffers {
    current: Buffer,
    old: Buffer,
}

impl TemporalBuffers {
    fn new(device: &Device, size: BufferAddress) -> Self {
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
        Self { current, old }
    }

    fn advance_frame(&self, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.current, 0, &self.old, 0, self.current.size());
        encoder.clear_buffer(&self.current, 0, None);
    }

    fn advance_frame_no_temporal(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.current, 0, None);
        encoder.clear_buffer(&self.old, 0, None);
    }
}

// Right now there are lots of different possible importance samplers so the parts of this will likely change
// Therefore access is restricted to this so that changes to this are not breaking changes
/// Buffers for storing information between executions, if `wip-features` is enabled then this
/// contains importance sampling data. It is recommended to create this instead of making it yourself
/// as it may change in the future but will not be regarded as sem-var breaking.
pub struct DataBuffers {
    markov_chain_screen_space: TemporalBuffers,
    spatial_resampling: TemporalBuffers,
    markov_chain_world_space: TemporalBuffers,
    info: Buffer,
}

impl DataBuffers {
    pub fn new(device: &Device, mut width: u32, mut height: u32, world_space_buffer_size: u32, buffers: BufferType) -> Self {
        width = width.max(1);
        height = height.max(1);
        #[cfg(feature = "wip-features")]
        let spatial_buffers = buffers.contains(BufferType::SPATIAL_RESAMPLING);
        #[cfg(not(feature = "wip-features"))]
        let spatial_buffers = false;
        let size_spatial = if spatial_buffers {
            width as BufferAddress * height as BufferAddress
        } else {
            1
        } * size_of::<crate::importance_sampling::Reservoir>() as BufferAddress;
        let size_markov = if buffers.contains(BufferType::MARKOV_CHAIN_SCREEN_SPACE) {
            width as BufferAddress * height as BufferAddress
        } else {
            1
        } * size_of::<MarkovChain>() as BufferAddress;
        let size_markov_world = if buffers.contains(BufferType::MARKOV_CHAIN_WORLD_SPACE) {
            world_space_buffer_size.max(1) as BufferAddress
        } else {
            1
        } * size_of::<WorldMarkovStorage>() as BufferAddress;
        let info_size = width as BufferAddress
            * height as BufferAddress
            * size_of::<crate::importance_sampling::Info>() as BufferAddress;
        let info = device.create_buffer(&BufferDescriptor {
            label: None,
            size: info_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_buffers = TemporalBuffers::new(device, size_spatial);
        let markov_buffers = TemporalBuffers::new(device, size_markov);
        let markov_world_buffers = TemporalBuffers::new(device, size_markov_world);
        Self {
            markov_chain_screen_space: markov_buffers,
            spatial_resampling: spatial_buffers,
            markov_chain_world_space: markov_world_buffers,
            info,
        }
    }

    /// Updates states to correct for a new frame, if a buffer is within
    /// `buffers_to_temporally_advance`, data will be moved to last frames buffer, otherwise it will
    /// be removed.
    pub fn advance_frame(&self, encoder: &mut CommandEncoder, buffers_to_temporally_advance: BufferType) {
        #[cfg(feature = "wip-features")]
        if buffers_to_temporally_advance.contains(BufferType::SPATIAL_RESAMPLING) {
            self.spatial_resampling.advance_frame(encoder);
        } else {
            self.spatial_resampling.advance_frame_no_temporal(encoder);
        }

        if buffers_to_temporally_advance.contains(BufferType::MARKOV_CHAIN_SCREEN_SPACE) {
            self.markov_chain_screen_space.advance_frame(encoder);
        } else {
            self.markov_chain_screen_space.advance_frame_no_temporal(encoder);
        }

        if buffers_to_temporally_advance.contains(BufferType::MARKOV_CHAIN_WORLD_SPACE) {
            encoder.copy_buffer_to_buffer(&self.markov_chain_world_space.current, 0, &self.markov_chain_world_space.old, 0, self.markov_chain_world_space.current.size());
        } else {
            self.markov_chain_world_space.advance_frame_no_temporal(encoder);
        }
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
                    resource: BindingResource::Buffer(self.spatial_resampling.current.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Buffer(self.spatial_resampling.old.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::Buffer(self.info.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::Buffer(self.markov_chain_screen_space.current.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: BindingResource::Buffer(self.markov_chain_screen_space.old.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: BindingResource::Buffer(self.markov_chain_world_space.current.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: BindingResource::Buffer(self.markov_chain_world_space.old.as_entire_buffer_binding()),
                },
            ],
        })
    }
}
