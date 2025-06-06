//! All bindings that depend on having no-vertex-return feature enabled
@group(0) @binding(2)
var acc_struct: acceleration_structure;

struct Vertices {
    geometry_stride: vec4<u32>,
    vertices: array<vec3<f32>>,
}

@group(0) @binding(3)
var<storage> vertices: binding_array<Vertices>;

struct Indices {
    indices: array<u32>,
}

@group(0) @binding(4)
var<storage> indices: binding_array<Indices>;

fn get_vertices(intersection: RayIntersection) -> array<vec3<f32>, 3> {
    let instance_idx = intersection.instance_index;
    let vertex_index = (vertices[instance_idx].geometry_stride.x * intersection.geometry_index) + (intersection.primitive_index * 3);
    let index_1 = select(indices[instance_idx].indices[vertex_index], vertex_index, arrayLength(&indices[instance_idx].indices) == 1);
    let index_2 = select(indices[instance_idx].indices[vertex_index + 1], vertex_index + 1, arrayLength(&indices[instance_idx].indices) == 1);
    let index_3 = select(indices[instance_idx].indices[vertex_index + 2], vertex_index + 2, arrayLength(&indices[instance_idx].indices) == 1);
    return array<vec3<f32>, 3>(vertices[instance_idx].vertices[index_1], vertices[instance_idx].vertices[index_2], vertices[instance_idx].vertices[index_3]);
}

alias ray_query_maybe_vertex_return = ray_query;