use crate::low_level::RayTracingShader;
use std::ops::Add;

/// Debugging shader to show whether the ray has hit the front face (green) or back face (red)
/// useful for translucent materials as they determine whether the ray is entering by whether
/// it hit the front face, to give some depth perception the brightness of the pixel is 1 / (depth + 1)
pub struct FrontFace;

unsafe impl RayTracingShader for FrontFace {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler() -> String {
        include_str!("front_face.wgsl")
            .to_string()
            .add(include_str!("../bindings.wgsl"))
            .add(include_str!("../importance_sampling/shared.wgsl"))
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Debugging shader"
    }
}

pub struct Reflectance;

unsafe impl RayTracingShader for Reflectance {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler() -> String {
        include_str!("reflectance.wgsl")
            .to_string()
            .add(include_str!("../bindings.wgsl"))
            .add(include_str!("../importance_sampling/shared.wgsl"))
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Debugging shader"
    }
}

pub struct Tangent;

unsafe impl RayTracingShader for Tangent {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler() -> String {
        include_str!("tangent.wgsl")
            .to_string()
            .add(include_str!("../bindings.wgsl"))
            .add(include_str!("../importance_sampling/shared.wgsl"))
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Debugging shader"
    }
}
