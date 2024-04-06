override SAMPLES = 5u;

struct Camera {
    // this way we can have any fov easily.
    projection_inverse: mat4x4<f32>,
    view_inverse: mat4x4<f32>,
}

struct Material {
    tex_pos_1: u32,
    tex_pos_2: u32,
    tex_pos_3: u32,
    tex_pos_recolour: u32,
    tex_idx_diffuse_emission: u32,
    tex_idx_attributes_ty: u32,
    emission_scale: f32,
    refractive_index: u32,
}

struct MaterialsIdx {
    materials: array<u32>,
}

@group(0) @binding(0)
var<storage> materials: array<Material>;
@group(0) @binding(1)
var<storage> material_idx: binding_array<MaterialsIdx>;
@group(0) @binding(2)
var acc_struct: acceleration_structure<vertex_return>;

@group(1) @binding(0)
var<uniform> camera: Camera;
@group(1) @binding(1)
var output: texture_storage_2d<rgba32float, write>;
@group(1) @binding(2)
var output_normal: texture_storage_2d<rgba32float, write>;
@group(1) @binding(3)
var output_albedo: texture_storage_2d<rgba32float, write>;
@group(1) @binding(4)
var bg: texture_cube<f32>;
@group(1) @binding(5)
var<storage, read_write> lights: WorkgroupLights;
@group(1) @binding(6)
var<storage> old_lights: WorkgroupLights;
@group(1) @binding(7)
var<storage, read_write> info: array<CompressedInfo>;

@group(2) @binding(0)
var sam:sampler;
@group(2) @binding(1)
var tex_diffuse: binding_array<texture_2d<f32>>;
@group(2) @binding(2)
var tex_emission: binding_array<texture_2d<f32>>;
@group(2) @binding(3)
// r: roughness
var tex_attributes: binding_array<texture_2d<f32>>;
@group(2) @binding(4)
var tex_recolour: texture_2d<f32>;

struct AABBIntersection {
    hit: bool,
    normal: vec3<f32>,
    tangent: vec3<f32>,
    t: f32,
}

fn unpack_2xu16(u:u32) -> vec2<u32> {
    let u1 = u & 0xFFFFu;
    let u2 = (u & 0xFFFF0000u) >> 16u;
    return vec2<u32>(u1, u2);
}