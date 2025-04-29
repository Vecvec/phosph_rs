fn get_normal(ray_direction:vec3<f32>, normal:vec3<f32>, tangent:vec3<f32>, seed:u32, roughness:vec2<f32>) -> vec3<f32> {
    return normal;
}

fn make_diffuse(hit_color:vec3<f32>, ray:Ray, intersection_normal:vec3<f32>, tangent:vec3<f32>, roughness:f32, new_dir: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0);
}

fn conv_rough(rough:f32) -> f32 {
    return 0.0;
}

fn pick_ref_idx(picked: ptr<function, PickedRefIdx>, low_idx:f32, high_idx:f32, seed:u32) -> vec4<f32> {
    return vec4<f32>(vec3<f32>(1.0), low_idx);
}