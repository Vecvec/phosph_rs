//! All bindings that depend on having no-vertex-return feature disabled
@group(0) @binding(2)
var acc_struct: acceleration_structure<vertex_return>;

alias ray_query_maybe_vertex_return = ray_query<vertex_return>;