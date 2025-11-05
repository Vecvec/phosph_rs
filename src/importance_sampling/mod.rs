#[cfg(feature = "wip-features")]
use crate::Shader;
#[cfg(feature = "wip-features")]
use wgpu::Features;

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
        use wesl::include_wesl;

        Shader {
            base: include_wesl!("importance_sampler").to_string(),
            #[cfg(debug_assertions)]
            label: "Importance Sampler",
        }
    }
}

pub(crate) struct Reservoir {
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

pub(crate) struct Info {
    _emission: [f32; 4],
    _albedo: [f32; 4],
    _cam_loc: [f32; 4],
}
