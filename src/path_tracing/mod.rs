use wesl::include_wesl;

use crate::low_level::RayTracingShader;

/// A ray-tracing shader, note that this requires ~6-10x the samples of the [Medium] shader.
///
/// Differences from [Medium]
///  - Picks a wavelength of light and uses that as its colour
pub struct High;

unsafe impl RayTracingShader for High {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler() -> String {
        include_wesl!("high_path_tracing").to_string()
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Ray-Tracing High shader"
    }
}

/// A ray-tracing shader
///
/// Features
///  - Randomly picks refractive index between low and high refractive indices
///  - 2D roughness for surface and 2D roughness for coating.
pub struct Medium;

unsafe impl RayTracingShader for Medium {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler() -> String {
        include_wesl!("medium_path_tracing").to_string()
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Ray-Tracing Medium shader"
    }
}

/// A ray-tracing shader, note that this only needs ~0.9x the samples of the [Medium] shader.
///
/// Differences from [Medium]
///  - Ignores roughness
///  - Only uses one refractive index
pub struct Low;

unsafe impl RayTracingShader for Low {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler() -> String {
        include_wesl!("low_path_tracing").to_string()
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Ray-Tracing Low shader"
    }
}
