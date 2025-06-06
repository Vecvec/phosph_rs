use crate::low_level::RayTracingShader;
use std::ops::Add;

#[cfg(feature = "no-vertex-return")]
macro_rules! replace_get_vertex {
    ($i:expr) => {$i.replace("GET_COMMITTED_VERTEX_POSITIONS", "get_vertices(rayQueryGetCommittedIntersection(&rq))")};
}

#[cfg(not(feature = "no-vertex-return"))]
macro_rules! replace_get_vertex {
    ($i:expr) => {$i.replace("GET_COMMITTED_VERTEX_POSITIONS", "getCommittedHitVertexPositions(&rq)")};
}

macro_rules! include_general {
    () => {
        replace_get_vertex!(include_str!("general.wgsl"))
            .to_string()
            .add(include_str!("../importance_sampling/shared.wgsl"))
            .add(crate::bindings!())
    };
}

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
        include_general!().add(include_str!("high.wgsl"))
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
        include_general!().add(include_str!("medium.wgsl"))
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
        include_general!().add(include_str!("low.wgsl"))
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Ray-Tracing Low shader"
    }
}
