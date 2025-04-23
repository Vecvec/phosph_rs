fn jacobian_determinant(q:Sample, r:Sample) -> f32 {
    let q1_to_q2 = q.sample_point - q.visible_point;
    let r1_to_q2 = r.sample_point - q.visible_point;
    let visible_normal = u32_to_normalised(q.visible_normal);
    let cos_theta_r = dot(normalize(r1_to_q2), visible_normal);
    let cos_theta_q = dot(normalize(q1_to_q2), visible_normal);
    return (safe_div((cos_theta_r) * len_sqrd(q1_to_q2), (cos_theta_q) * len_sqrd(r1_to_q2)));
}

const F32_MAX = 3.40282347e+38;

fn safe_div_vec3(numerator: vec3<f32>, denominator: f32) -> vec3<f32> {
    if (near_zero(denominator))  {
        return vec3<f32>(0.0);
    } else {
        return numerator / denominator;
    }
}

override IS_SAMPLES = 8u;
override IS_SPACE = 61u;
override DO_NOT_OVERRIDE = IS_SPACE * IS_SPACE;

const MIN_DIST = 0.00001;

@workgroup_size(8, 8, 1)
@compute
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) work_id: vec3<u32>) {
    let screen_size = textureDimensions(output);
    if (id.x > screen_size.x || id.y > screen_size.y) {
        return;
    }
    let idx = id.x + (id.y * screen_size.x);
    let x = f32(id.y)/f32(screen_size.y);
    var own_seed = rand_u32((id.x * id.y)) + rand_u32(seed_offset) + id.y;
    var pixel_color = vec3<f32>();
    var sample_is_samples = 0u;
    var is_color = vec3<f32>();
    var Rs = lights.samples[idx];
    var Z = 0u;
    let Rs_point = Rs.sample_point;
    var Rs_normal = u32_to_normalised(Rs.sample_normal);
    var Q = array<Reservoir, 129>();
    Q[0] = Rs;
    var i: u32;
    let input_M = unpack4xU8(Rs.confidence8_valid8).x;
    var selected = 0u;
    var w_sum = p_hat(sam_from_res(Rs));
    var out_radiance = vec3<f32>();
    var is_space = IS_SPACE;
    for (i = 0u; i < IS_SAMPLES; i++) {
        //Randomly choose a neighbor pixel qn
        let max_val = (is_space * is_space) - 1;
        let min_space = is_space / 2;
        let self_idx = min_space + (min_space * is_space);
        var rand = rand_u32(own_seed) % max_val;
        if (rand >= self_idx) {
            rand = rand + 1;
        }
        let x_neighbour = rand % is_space;
        let y_neighbour = rand / is_space;
        var neighbour = vec2<i32>(i32(x_neighbour), i32(y_neighbour)) - vec2<i32>(i32(min_space));
        own_seed = rand_u32(own_seed);
        var other_id_i = vec2<i32>(id.xy) + neighbour;
        if (other_id_i.x < i32(0) || other_id_i.y < i32(0) || (neighbour.x == i32(0) && neighbour.y == i32(0))) {
            //is_space = min(is_space / 2, 3);
            continue;
        }
        var other_id = vec2<u32>(other_id_i);
        if (other_id.x >= screen_size.x || other_id.y >= screen_size.y) {
            continue;
        }
        var work_idx = other_id.x + (other_id.y * screen_size.x);
        var Rn = lights.samples[work_idx];
        var Rn_normal = u32_to_normalised(Rn.sample_normal);
        //Calculate geometric similarity between q and qn
        let similarity = dot(Rn_normal, Rs_normal);
        //if similarity is lower than the given threshold then continue
        if (((similarity) < 0.9) || (Rn.ty != Rs.ty) || abs(sam_from_res(Rs).roughness - sam_from_res(Rn).roughness) > 0.1) {
            continue;
        }
        // Calculate |Jqn→q|
        var jacobian_determinant = jacobian_determinant(sam_from_res(Rn), sam_from_res(Rs));
        if (Rs.ty != DIFFUSE) {
            continue; //TODO
            //jacobian_determinant = 0.0;
        }
        //var Rn_ray = normalize(Rn.sample_point - Rn.visible_point);
        //ˆp′q ← ˆpq(Rn.z)/|Jqn→q|
        var p_hat_dashed_q = p_hat(sam_from_res(Rn)) * jacobian_determinant;
        var current_ray = normalize(Rs.sample_point - Rn.visible_point);
        let dist = length(Rs.sample_point - Rn.visible_point);
        // if Rn’s sample point is not visible to xv at q then ˆp′q ← 0
        /*if ((dist - MIN_DIST) > MIN_DIST) {
            var rq: ray_query;
            rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, MIN_DIST, dist - MIN_DIST, Rs.sample_point, current_ray));
            rayQueryProceed(&rq);
            let intersection = rayQueryGetCommittedIntersection(&rq);
            if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
                p_hat_dashed_q = 0.0;
                //continue;
            }
        }*/
        // Rs.MERGE(Rn, ˆp′q)
        let merge_return = merge(Rs, Rn, p_hat_dashed_q, own_seed);
        own_seed = rand_u32(own_seed);
        Rs = merge_return.reservoir;
        //out_radiance += (Rs.out_radiance * jacobian_determinant);
        if (merge_return.picked_r) {
            //textureStore(output, id.xy, vec4<f32>(0.0, 1.0, 0.0, 1.0));
            //Rs = merge_return.reservoir;
            Rs_normal = u32_to_normalised(Rs.sample_normal);
            selected = sample_is_samples + 1u;
            //return;
        } else {
            //textureStore(output, id.xy, vec4<f32>(1.0, 0.0, 0.0, 1.0));
            //return;
        }
        Q[sample_is_samples + 1u] = Rn;
        w_sum += p_hat_dashed_q;
        own_seed = rand_u32(own_seed);
        //Z = f32(unpack4xU8(Rs.M_valid).x);
        sample_is_samples = sample_is_samples + 1u;
    }
    var confidence_i_p_i = 0.0;
    var confidence_p_sum = 0.0;
    var j: u32;
    for (j = 0u; j < (sample_is_samples + 1u); j++) {
        // if ˆpqn (Rs.z) > 0 then Z ← Z + Rn.M
        let Rn = Q[j];
        let confidence_j_p_j = f32(unpack4xU8(Rn.confidence8_valid8).x) * p_hat(sam_from_res(Rn));
        if (selected == j) {
            confidence_i_p_i = confidence_j_p_j;
        }
        confidence_p_sum += confidence_j_p_j;
        if (p_hat(sam_from_res(Rn)) > 0.0) {
            Z = Z + (unpack4xU8(Rn.confidence8_valid8).x);
        }
    }
    let m_i = safe_div(confidence_i_p_i, confidence_p_sum);
    let Z_times_p_hat = f32(Z)
        //;
        //* pdf(Rs_normal, normalize(to_info(info[idx]).cam_loc - Rs_point), normalize(sam_from_res(Rs).visible_point - Rs_point), sam_from_res(Rs).ty, to_info(info[idx]).albedo, sam_from_res(Rs).roughness);
        *p_hat(sam_from_res(Rs));
    //TODO: this is biased if the distributions are different. We should get a better distribution (below is one that is not working but would be much better).
    Rs.W = safe_div(Rs.w, Z_times_p_hat);
    //Rs.W = (confidence_i_p_i * Rs.w) / (confidence_p_sum * p_hat(sam_from_res(Rs)));
    //storageBarrier();
    //lights.samples[idx] = Rs;
    //pixel_color = pixel_color + (lights.samples[idx].sample.out_radiance * pixel_albedo) + pixel_emission;
    if (sample_is_samples != 0u) {
        //let pix_is = fma(Rs.out_radiance * Rs.W, to_info(info[idx]).albedo, to_info(info[idx]).emission);
        //textureStore(output, id.xy, vec4<f32>(pix_is, 1.0));
    }
    if (unpack4xU8(Rs.confidence8_valid8).y == 0) {
        //return;
    }
    let pix_is = fma((Rs.out_radiance * Rs.W), to_info(info[idx]).albedo, to_info(info[idx]).emission);
    //let pix_is = fma(safe_div_vec3(out_radiance, f32(sample_is_samples)), to_info(info[idx]).albedo, to_info(info[idx]).emission);
    textureStore(output, id.xy, vec4<f32>(vec3(f32(sample_is_samples == 0)), 1.0));
    let diff = ((confidence_i_p_i * Rs.w) - (confidence_p_sum * p_hat(sam_from_res(Rs)))) / 24.0;
    textureStore(output, id.xy, vec4<f32>(vec3(max(diff, 0.0), max(-diff, 0.0), 0.0), 1.0));
    textureStore(output, id.xy, vec4<f32>(vec3(Rs.W / 1.0), 1.0));
    textureStore(output, id.xy, vec4<f32>((pix_is), 1.0));
}

var<push_constant> seed_offset:u32;

const U16_MAX = 0xFFFFu;