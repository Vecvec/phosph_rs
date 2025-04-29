use crate::{DynamicRayTracer, RayTracer, Shader};
use std::num::NonZeroU32;
use std::ops::Add;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, BufferSize, Device, Features, Limits, PipelineLayout,
    PipelineLayoutDescriptor, PushConstantRange, SamplerBindingType, ShaderStages,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

/// Source should return a string that contains functions with names appended with `intersect`.
/// One function should have the definition
/// ````wgsl
/// fn intersect(intersection: RayIntersection) -> AABBIntersection
/// ````
/// It should not cause UB these are not fulfilled, but it is a logic error and may cause panics in code that
/// otherwise should not fail.
///
/// Note: `AABBIntersection` is defined as
/// ````wgsl
/// struct AABBIntersection {
///     hit: bool,
///     normal: vec3<f32>,
///     t: f32,
/// }
/// ````
///
/// It is considered a logic error if the function does not pass validation (but should *not* cause UB)
///
/// # Safety:
///
/// - The function returned by `source`, when executed, *must* return in a finite amount of time.
pub unsafe trait IntersectionHandler: 'static {
    fn source(&self) -> String;
    fn additional_bind_group_layouts(&self, _device: &Device) -> Vec<BindGroupLayout> {
        Vec::new()
    }
}

/// It is considered a logic error if the module does not pass validation (but does *not* cause UB)
///
/// # Safety:
///
/// - The shader returned by `create_shader`, when executed, *must* return in a finite amount of time.
pub unsafe trait RayTracingShader: Sized + 'static {
    fn new() -> Self;
    fn features() -> Features {
        #[cfg(feature = "no-vertex-return")]
        let maybe_vertex_return = Features::empty();
        #[cfg(not(feature = "no-vertex-return"))]
        let maybe_vertex_return = Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN;
        // features required to interact
        Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::EXPERIMENTAL_RAY_QUERY
            | Features::STORAGE_RESOURCE_BINDING_ARRAY
            | Features::BUFFER_BINDING_ARRAY
            | Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | Features::PUSH_CONSTANTS
            | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | Features::TEXTURE_BINDING_ARRAY
            | Features::PARTIALLY_BOUND_BINDING_ARRAY
            | maybe_vertex_return
    }
    fn limits() -> Limits {
        // limits required to interact
        Limits {
            max_push_constant_size: 4,
            max_storage_buffer_binding_size: Limits::default().max_storage_buffer_binding_size,
            // see docs for why 500,000.
            max_binding_array_elements_per_shader_stage: 500_000,
            ..Limits::default()
        }
    }
    fn limits_or(limit: Limits) -> Limits {
        // limits required to interact
        Limits {
            max_push_constant_size: 4.max(limit.max_push_constant_size),
            max_storage_buffer_binding_size: (Limits::default()
                .max_storage_buffer_binding_size)
                .max(limit.max_storage_buffer_binding_size),
            max_binding_array_elements_per_shader_stage: 500_000
                .max(limit.max_binding_array_elements_per_shader_stage),
            ..limit
        }
    }
    fn shader_source_without_intersection_handler() -> String;
    #[cfg(debug_assertions)]
    fn label() -> &'static str;
    fn create_shader() -> Shader {
        Shader {
            base: Self::shader_source_without_intersection_handler()
                .add(include_str!("default_intersection_handler.wgsl")),
            #[cfg(debug_assertions)]
            label: Self::label(),
        }
    }
}

/// Exactly the same as the ray-tracing trait but takes itself so can be made into a dst
///
/// Implementors should implement [RayTracingShader] instead
///
/// # Safety:
///
/// Same as [RayTracingShader]
pub unsafe trait RayTracingShaderDST {
    fn features(&self) -> Features;
    fn limits(&self) -> Limits;
    fn limits_or(&self, limit: Limits) -> Limits;
    fn shader_source_without_intersection_handler(&self) -> String;
    #[cfg(debug_assertions)]
    fn label(&self) -> &'static str;
    fn create_shader(&self) -> Shader;
    fn dyn_ray_tracer(&self, device: &Device) -> DynamicRayTracer;
}

// # Safety:
//
// The safety requirements of `RayTracingShader` and `RayTracingShaderDST` are the same so anyone
// who has implemented `RayTracingShader` must have met the guarantees of `RayTracingShaderDST`.
unsafe impl<T: RayTracingShader> RayTracingShaderDST for T {
    fn features(&self) -> Features {
        T::features()
    }
    fn limits(&self) -> Limits {
        T::limits()
    }
    fn limits_or(&self, limit: Limits) -> Limits {
        T::limits_or(limit)
    }
    fn shader_source_without_intersection_handler(&self) -> String {
        T::shader_source_without_intersection_handler()
    }
    #[cfg(debug_assertions)]
    fn label(&self) -> &'static str {
        T::label()
    }
    fn create_shader(&self) -> Shader {
        T::create_shader()
    }
    fn dyn_ray_tracer(&self, device: &Device) -> DynamicRayTracer {
        RayTracer::<T>::new(device).dynamic()
    }
}

pub fn pipeline_layout(
    device: &Device,
    blas_count: NonZeroU32,
    diffuse_count: NonZeroU32,
    emission_count: NonZeroU32,
    attribute_count: NonZeroU32,
    extra_bgls: &[BindGroupLayout],
) -> PipelineLayout {
    #[cfg(not(feature = "no-vertex-return"))]
    let entries = &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(BufferSize::new(44).unwrap()),
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(BufferSize::new(4).unwrap()),
            },
            count: Some(blas_count),
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::AccelerationStructure {
                vertex_return: true,
            },
            count: None,
        },
    ];
    #[cfg(feature = "no-vertex-return")]
    let entries = &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(BufferSize::new(44).unwrap()),
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(BufferSize::new(4).unwrap()),
            },
            count: Some(blas_count),
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::AccelerationStructure {
                vertex_return: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: Some(blas_count),
        },
        BindGroupLayoutEntry {
            binding: 4,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: Some(blas_count),
        },
    ];
    let mat_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries,
    });
    let mut bgls = Vec::with_capacity(extra_bgls.len() + 3);
    bgls.push(&mat_bgl);
    let out_bgl = out_bgl(device);
    bgls.push(&out_bgl);
    let texture_bgl = texture_bgl(device, [diffuse_count, emission_count, attribute_count]);
    bgls.push(&texture_bgl);
    bgls.extend(extra_bgls.iter());
    device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&mat_bgl, &out_bgl, &texture_bgl],
        push_constant_ranges: &[PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..4,
        }],
    })
}

pub(crate) fn texture_bgl(device: &Device, counts: [NonZeroU32; 3]) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(counts[0]),
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(counts[1]),
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(counts[2]),
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

pub(crate) fn out_bgl(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(BufferSize::new(128).unwrap()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}
