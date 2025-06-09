const BOUNCES = 5u;
const VERTICES = BOUNCES + 1u;

struct Ray {
    direction: vec3f,
    origin: vec3f,
}

struct Intersection {
    color: vec3f,
    t: f32,
    emission: vec3f,
    ty: u32,
    normal: vec3f,
    front_face: bool,
    tangent: vec3<f32>,
    hit: bool,
    tri_area: f32,
    roughness: vec2<f32>,
    roughness_transparent: vec2<f32>,
    refractive_index_low: f32,
    refractive_index_high: f32,
}


// WIP: Importance sampling

/*fn jacobian_determinant(current:Sample, other:Sample) -> f32 {
    let q1_to_q2 = other.sample_point - other.visible_point;
    let r1_to_q2 = current.sample_point - other.visible_point;
    let cos_theta_r = dot(other.visible_normal, r1_to_q2);
    let cos_theta_q = dot(other.visible_normal, q1_to_q2);
    return (clamp(abs(cos_theta_r), 0.01, 1.0) / clamp(abs(cos_theta_q), 0.01, 1.0)) * (len_sqrd(q1_to_q2) / len_sqrd(r1_to_q2));
}*/

// adds a light to the current frames list of lights
/*fn push(light: vec4<f32>) {
    if (near_zero(light.w)) {
        return;
    }
    //let old_len = atomicLoad(&lights.size);
    let val = atomicAdd(&lights.size, 1u);
    if val >= arrayLength(&lights.lights) - 1u {
        atomicStore(&lights.size, arrayLength(&lights.lights) - 1u);
    } else {
        lights.lights[val] = light;
    }
    return;
}

// gets a random light
fn get_rand(seed:u32) -> vec4<f32> {
    let maximum_new = atomicLoad(&lights.size);
    let maximum_old = atomicLoad(&old_lights.size);
    let maximum = maximum_old + maximum_new;
    let val = rand_u32(seed) % maximum;
    if val < maximum_new {
        let light = lights.lights[val];
        let brightness = brightness(f32(maximum), light.w);
        return vec4<f32>(light.xyz, brightness);
    } else {
        let light = old_lights.lights[val];
        let brightness = brightness(f32(maximum), light.w);
        return vec4<f32>(light.xyz, brightness);
    }
}*/

fn brightness(max:f32, odds: f32) -> f32 {
    return odds;
    //return odds * 32.0;
    //return odds * 400.0;
    //return odds * 4.0;
    //return odds * 80.0;
    //return 4.0;
}

var<push_constant> seed_offset:u32;

override IS_SAMPLES = 64u;

const RESAMPLE_MIN = 50u;
const RESAMPLE_MAX = 100u;

// TODO: Move this into a inter pixel buffer so all pixels can sample from it.
// S in the paper.
var<private> tentative_marcov_chain_state: MarcovChainState;

// Sc in the paper.
var<private> current_marcov_chain_state: MarcovChainState;

struct MarcovChainHolder {
    state: array<MarcovChainState, NUM_RESAMPLES>,
    sum_weight_sum: f32,
}

const NUM_RESAMPLES = 64;
var<private> other_pixels_marcov_chain_states: MarcovChainHolder;

fn shuffle_marcov_chain(next_subgroup:u32) {
    tentative_marcov_chain_state = get_tentative_marcov_chain(next_subgroup);
}

fn get_tentative_marcov_chain(subgroup_id:u32) -> MarcovChainState {
    return MarcovChainState(
        subgroupShuffle(tentative_marcov_chain_state.light_source, subgroup_id),
        subgroupShuffle(tentative_marcov_chain_state.mean_cosine, subgroup_id),
        subgroupShuffle(tentative_marcov_chain_state.weight_sum, subgroup_id),
        subgroupShuffle(tentative_marcov_chain_state.num_samples, subgroup_id),
        subgroupShuffle(tentative_marcov_chain_state.score, subgroup_id),
    );
}

fn fill_mc_states(own_seed: ptr<function, u32>, sub_size: u32) {
    for (var i = 0; i < NUM_RESAMPLES; i++) {
        let sub_id = rand_u32(*own_seed) % sub_size;
        let s = get_tentative_marcov_chain(sub_id);
        other_pixels_marcov_chain_states.state[i] = s;
        *own_seed = rand_u32(*own_seed);
    }
    //other_pixels_marcov_chain_states.state[0] = tentative_marcov_chain_state;
}

var<private> sum_mc: f32;

fn resample_mcmm(own_seed: ptr<function, u32>) -> MarcovChainState {
    var s = MarcovChainState();
    sum_mc = 0.0;
    for (var i = 0; i < NUM_RESAMPLES; i++) {
        var rand_candidate = rand_f32(*own_seed);
        *own_seed = rand_u32(*own_seed);
        let s_candidate = other_pixels_marcov_chain_states.state[i];
        sum_mc += s_candidate.weight_sum;
        var rand = rand_f32(*own_seed);
        if (rand * sum_mc) < s_candidate.weight_sum {
            s = s_candidate;
        }
        *own_seed = rand_u32(*own_seed);
    }
    other_pixels_marcov_chain_states.sum_weight_sum = sum_mc;
    return s;
}

fn marcov_chain_valid(state: MarcovChainState) -> bool {
    return state.weight_sum > 0.0;
}

// Moved the number in the paper outside to a constant.
const MARCOV_CHAIN_SAMPLE_CLAMP = 1024;
const FLOAT_MAX = 3.40282347E+38;

// x0 is never used so we don't include it.
//fn update_marcov_state(/* x0: vec3<f32>, */ x1: vec3<f32>, x2: vec3<f32>, pdf: f32, p_hat: f32, seed: u32, not_vmf: bool) {
//    let score = safe_div(p_hat, pdf);
//    let a = min(1.0, div_or(score, current_marcov_chain_state.score, 1.0));
//    let rand = rand_f32(seed);
//    var update = false;
//    if ((score > 0.0) && (!marcov_chain_valid(current_marcov_chain_state) || rand < a)) { // May want to convert this to be closer to the GI update.
//        if not_vmf {
//            tentative_marcov_chain_state = MarcovChainState();
//        }
//        tentative_marcov_chain_state.score = score;
//        update = true;
//    }
//    tentative_marcov_chain_state.num_samples = min(tentative_marcov_chain_state.num_samples + 1, MARCOV_CHAIN_SAMPLE_CLAMP);
//    let alpha = max(1.0 / f32(tentative_marcov_chain_state.num_samples), 0.1);
//    // WGSL mix is the same as the one in the paper.
//    let weight_sum = mix(tentative_marcov_chain_state.weight_sum, score, alpha);
//    tentative_marcov_chain_state.light_source = mix(tentative_marcov_chain_state.weight_sum * tentative_marcov_chain_state.light_source, score * x2, alpha) / weight_sum;
//    let mean = normalize(tentative_marcov_chain_state.light_source - x1);
//    // Instead of mean_cosine there was something else in the paper, but there is no such field in
//    // MarcovChainState and it isn't accessed anywhere else and r isn't accessed here so assuming it's r.
//    let omega = mix(tentative_marcov_chain_state.weight_sum * tentative_marcov_chain_state.mean_cosine * mean, score * normalize(x2 - x1), alpha);
//    tentative_marcov_chain_state.mean_cosine = (length(omega));
//    tentative_marcov_chain_state.weight_sum = weight_sum;
//    if update {
//        current_marcov_chain_state = tentative_marcov_chain_state;
//    }
//}

fn update_marcov_state(/* x0: vec3<f32>, */ x1: vec3<f32>, x2: vec3<f32>, pdf: f32, p_hat: f32, seed: u32, not_vmf: bool) {
    if not_vmf {
        tentative_marcov_chain_state = MarcovChainState();
    }
    let score = p_hat;
    let rand = rand_f32(seed);
    /*if near_zero(sum_mc) {
        tentative_marcov_chain_state.num_samples = min(tentative_marcov_chain_state.num_samples + 1, MARCOV_CHAIN_SAMPLE_CLAMP);
        let alpha = max(1.0 / f32(tentative_marcov_chain_state.num_samples), 0.1);
        tentative_marcov_chain_state.weight_sum = score;
        tentative_marcov_chain_state.light_source = score * x2;
        tentative_marcov_chain_state.mean_cosine = score;
    } else */if (rand * sum_mc < (p_hat * f32(NUM_RESAMPLES))) {
        if near_zero(length(get_light_dir(tentative_marcov_chain_state, x1))) {
            return;
        }
        tentative_marcov_chain_state.num_samples = min(tentative_marcov_chain_state.num_samples + 1, MARCOV_CHAIN_SAMPLE_CLAMP);
        let alpha = max(1.0 / f32(tentative_marcov_chain_state.num_samples), 0.1);
        let mean = normalize(get_light_dir(tentative_marcov_chain_state, x1));
        tentative_marcov_chain_state.weight_sum = mix(tentative_marcov_chain_state.weight_sum, score, alpha);
        tentative_marcov_chain_state.light_source = mix(tentative_marcov_chain_state.light_source, score * x2, alpha);
        //tentative_marcov_chain_state.light_source = mix(tentative_marcov_chain_state.light_source, x2, alpha);
        tentative_marcov_chain_state.mean_cosine = mix(tentative_marcov_chain_state.mean_cosine, score * dot(normalize(x2 - x1), mean), alpha);
    }
}

/// Returns an unnormalized vector from the `current` to the light
fn get_light_dir(state: MarcovChainState, current: vec3<f32>) -> vec3<f32> {
    return safe_div_vec3(state.light_source, state.weight_sum) - current;
    //return state.light_source - current;
}

@workgroup_size(64, 1, 1)
@compute
fn rt_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) work_id: vec3<u32>, @builtin(subgroup_size) sub_size:u32, @builtin(subgroup_invocation_id) sub_id: u32) {
    let screen_size = textureDimensions(output);
    if (id.x > screen_size.x || id.y > screen_size.y) {
        return;
    }
    let idx = id.x + (id.y * screen_size.x);
    tentative_marcov_chain_state = lights.samples[idx];
    let x = f32(id.y)/f32(screen_size.y);
    var own_seed = rand_u32((id.x * id.y)) + seed_offset + id.y;
    var pixel_color = vec3<f32>();
    var pixel_normal: vec3<f32>;
    var pixel_albedo: vec3<f32>;
    var pixel_emission: vec3<f32>;
    var pixel_ty: u32;
    var pixel_point: vec3<f32>;
    var pixel_roughness: f32;
    var is_samples = 0u;
    var out_radiance = vec3<f32>();
    var cam_loc: vec3<f32>;
    var sample: Sample;
    var sample_valid = false;
    var pdf = 0.0;
    let next_sub_id = (sub_id + 1) % sub_size;
    for (var i = 0u; i < SAMPLES; i++) {
        fill_mc_states(&own_seed, sub_size);
        let ray_sample = rt_sample(id.xy, own_seed, (i == 0u));
        cam_loc = ray_sample.cam_loc;
        pixel_normal = ray_sample.normal;
        pixel_albedo = ray_sample.albedo;
        pixel_emission = ray_sample.emission;
        pixel_ty = ray_sample.ty;
        pixel_color = pixel_color + ray_sample.color;
        out_radiance += ray_sample.out_radiance;
        pixel_roughness = ray_sample.roughness;
        pixel_point = ray_sample.point;
        //pixel_color = pixel_color + ray_sample.out_radiance;
        own_seed = rand_u32(own_seed);

        if (ray_sample.color.x < 0.0 || ray_sample.color.y < 0.0 || ray_sample.color.z < 0.0) {
            //textureStore(output, id.xy, vec4<f32>(0.0, 1.0, 0.0, 1.0));
            //return;
        }
        if (ray_sample.valid) {
            sample_valid = true;
            sample = Sample(ray_sample.visible_point, normalised_to_u32(ray_sample.visible_normal), ray_sample.point, normalised_to_u32(pixel_normal), out_radiance / f32(SAMPLES), pixel_ty, conv_rough(ray_sample.roughness), 1.0);
            pdf += ray_sample.pdf;
        }
        shuffle_marcov_chain(next_sub_id);
        //lights.samples[idx] = update(sample, 1.0, lights.samples[idx], own_seed);
    }
    pdf /= f32(SAMPLES);
//    if (sample.ty != SKY && sample_valid) {
//        var R: Reservoir;
//        let resam_R = old_lights.samples[idx];
//        R = resam_R;
//        //R.pdf = saturate(sample.pdf);
//        let w = p_hat(sample); // Don't think `/ pdf` should be here, in the paper it's here but it's probably part of the out radiance.
//        R = update(sample, w, R, own_seed, false);
//        own_seed = rand_u32(own_seed);
//        R.W = safe_div(R.w, (f32(unpack4xU8(R.confidence8_valid8).x) * p_hat(sam_from_res(R))));
//        //lights.samples[idx] = R;
//    }
    //info[idx] = from_info(Info(cam_loc, pixel_emission, pixel_albedo));
    //lights.samples[idx].out_radiance = out_radiance / f32(SAMPLES);
    //lights.samples[idx].W = lights.samples[idx].w / max(0.00001, (f32(unpack4xU8(lights.samples[idx].M_valid).x) * p_hat(sam_from_res(lights.samples[idx]), pixel_normal, lights.samples[idx].sample_point, pixel_albedo, -normalize(lights.samples[idx].sample_point - cam_loc))));
    lights.samples[idx] = tentative_marcov_chain_state;
    textureStore(output_normal, id.xy, vec4<f32>(pixel_normal, 1.0));
    textureStore(output_albedo, id.xy, vec4<f32>(pixel_albedo, 1.0));
    textureStore(output, id.xy, vec4<f32>(max(pixel_color / f32(SAMPLES), vec3<f32>(0.0)), 1.0));
    //textureStore(output, id.xy, vec4<f32>(vec3((lights.samples[idx].W / 5.0)), 1.0));
}

struct SampleReturn {
    color: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    point: vec3<f32>,
    emission: vec3<f32>,
    visible_point: vec3<f32>,
    visible_normal: vec3<f32>,
    out_radiance: vec3<f32>,
    ty:u32,
    roughness: f32,
    cam_loc: vec3<f32>,
    valid: bool,
    pdf:f32,
}

struct DiffuseReturn {
    brightness: vec3<f32>,
    direction: vec3<f32>,
    dot: f32,
}

struct PickedRefIdx {
    picked_idx: f32,
    picked: bool,
}

struct RISDirections {
    dir: vec3<f32>,
    weight: f32,
}

fn rt_sample(coord: vec2<u32>, own_seed: u32, should_push:bool) -> SampleReturn {
    var self_seed = own_seed;
    var ray = create_ray(coord);
    // add to sample when hitting a light source
    var color: vec3<f32> = vec3<f32>(1.0);
    var out_radiance_colour: vec3<f32> = vec3<f32>(1.0);
    var sample: SampleReturn = SampleReturn();
    sample.ty = SKY;
    sample.cam_loc = ray.origin;
    sample.pdf = 1.0;
    var previously_diffuse = false;
    var previous_brightness = 0.0;
    // like i but only starts counting after first diffuse surface or any surface that has a roughness > 0.2
    var i_diffuse = 0u;
    var accum_emission = vec3<f32>();
    var accum_albedo = vec3<f32>(1.0);
    var picked = PickedRefIdx();
    var cached_pdf: f32;
    var last_not_vmf = false;
    var vmf_weight = 1.0;
    for (var i = 0u; i < BOUNCES; i++) {
        let intersection = ray_hit(&ray);
        if (intersection.hit) {
            let new_pos = at(&ray, intersection.t);
            // We have at least hit the scene twice (this is our second time)
            if (i > 1u) && (length(intersection.emission) > 0.0) {
                update_marcov_state(ray.origin, new_pos, cached_pdf, length(intersection.emission), self_seed, last_not_vmf);
                self_seed++;
            }
            last_not_vmf = false;
            ray.origin = new_pos;
            accum_emission = fma(intersection.emission, accum_albedo, accum_emission);
            accum_albedo *= intersection.color;
            if (i == 0u) {
                sample.point = ray.origin;
                sample.emission = accum_emission;
                sample.ty = intersection.ty;
                //sample.roughness = intersection.roughness;
                sample.normal = intersection.normal;
                sample.albedo = accum_albedo;
            } else if (i == 1u) {
                sample.visible_point = ray.origin;
                sample.visible_normal = intersection.normal;
            }
            if ((intersection.tri_area > SMALL_VAL) && previously_diffuse) {
                //sample.color = vec3f(intersection.tri_area);
                //return sample;
            }
            /*if ((intersection.tri_area > SMALL_VAL) && should_push && previously_diffuse) {
                push(vec4<f32>(ray.origin, intersection.tri_area));
                previously_diffuse = false;
                //sample.color = vec3f(intersection.tri_area);
                //return sample;
            }*/
            sample.color = fma(color, intersection.emission, sample.color);
            if (!(i_diffuse == 0u)) {
                sample.out_radiance = fma(out_radiance_colour, intersection.emission, sample.out_radiance);
            }
            let assign_normal = get_normal(ray.direction, intersection.normal, intersection.tangent, self_seed, intersection.roughness);
            let transparent_normal = get_normal(ray.direction, intersection.normal, intersection.tangent, self_seed, intersection.roughness_transparent);
            var normal = transparent_normal;
            // if not reset this will cary the last loop's should_reflect
            var should_reflect = false;

            self_seed = self_seed + 1u;
            let colour_refractive_index = pick_ref_idx(&picked, intersection.refractive_index_low, intersection.refractive_index_high, self_seed);
            var refractive_index = colour_refractive_index.w;
            self_seed = self_seed + 1u;
            if ((!intersection.front_face) && intersection.ty == TRANSPARENT) {
                refractive_index = refractive_index;
            } else {
                refractive_index = 1.0 / refractive_index;
            }
            color *= colour_refractive_index.xyz;
            if (refractive_index != 1.0) {
                let cos_theta = min(dot(-ray.direction, normal), 1.0);
                let sin_theta = sqrt(1.0 - (cos_theta*cos_theta));
                let cannot_refract = (refractive_index * sin_theta) > 1.0;
                // REMEMBER: Looking from inside a material looks odd, but this seems to be correct.
                let pdf = reflectance(cos_theta, refractive_index);
                let rand = rand_f32(self_seed);
                if ((cannot_refract || (pdf > rand))) {
                    //sample.pdf *= select(pdf, 1.0, cannot_refract);
                    should_reflect = true;
                    //sample.color = vec3f(f32(cannot_refract));
                    //return sample;
                }
                //sample.pdf *= (1.0 - pdf);
                self_seed = self_seed + 1u;
                //sample.color = vec3f(reflectance(cos_theta, refractive_index));
            }

            if ((!should_reflect) || (intersection.ty == TRANSPARENT)) {
                color = color * intersection.color;
                if (!(i_diffuse == 0u)) {
                    out_radiance_colour = out_radiance_colour * intersection.color;
                }
            }

            if (!should_reflect) {
                normal = assign_normal;
                self_seed = self_seed + 1u;
            }

            var s = resample_mcmm(&self_seed);

            //sample.color = (normal);
            //return sample;
            //sample.color = color;
            //return sample;
            let reflecting = should_reflect || intersection.ty == METALLIC;
            if (!reflecting) {
                if (intersection.ty == DIFFUSE) {
                    // old direction code
                    /*let direction = rand_on_sphere(self_seed);
                    let dot = dot(direction, intersection.normal);
                    color = color * abs(dot);
                    if (dot < 0.0) {
                        ray.direction = -direction;
                    } else {
                        ray.direction = direction;
                    }*/
                    var dir: vec3<f32>;
                    // Prepare for markov chain
                    //s.light_source = vec3<f32>(0.0, 1.0, 0.0);
                    let mean = normalize(get_light_dir(s, ray.origin));
                    let kappa = get_kappa(s);
                    // Calculate cosine direction
                    let dir_cosine = cosine_weighted_hemisphere(self_seed, intersection.normal, intersection.tangent);
                    // Calculate markov chain direction
                    let theta = rand_f32(self_seed) * TAU;
                    let V = vec2<f32>(cos(theta), sin(theta));
                    self_seed = self_seed + 1u;
                    let xi = rand_f32(self_seed);
                    self_seed = self_seed + 1u;
                    let minus_2_kappa = -2.0 * kappa;
                    let W_minus_1 = ((log(1.0 + (xi * exp_minus_one(minus_2_kappa))) / kappa));
                    var W = 1.0 + W_minus_1;
                    // slightly more precise than `sqrt(1.0 - (W*W))`
                    let norm_V = sqrt(-fma(W_minus_1, W_minus_1, 2*W_minus_1));
                    let Xs = from_onb_no_tangent(vec3(V.x * norm_V, W, V.y * norm_V), mean);
                    let dir_markov = normalize(Xs);
                    // Calculate pdf of cosine for cosine direction
                    var unnormalied_confidences = array<f32, 2>(0.2, 0.8);
                    //var unnormalied_confidences = array<f32, 2>(0.8 , 0.2);
                    if (!marcov_chain_valid(s) || near_zero(length(get_light_dir(s, ray.origin))) || near_zero(s.weight_sum)) {
                        unnormalied_confidences[0] = 1.0;
                        unnormalied_confidences[1] = 0.0;
                    }
                    //unnormalied_confidences[0] = 1.0;unnormalied_confidences[1] = 0.0;
                    let confidence_sum = unnormalied_confidences[0] + unnormalied_confidences[1];
                    var confidences = array<f32, 2>(unnormalied_confidences[0] / confidence_sum, unnormalied_confidences[1] / confidence_sum);
                    var final_pdf: f32;
                    var norm_divisor = 1.0;
                    let weight = confidences[0];
                    let rand = rand_f32(self_seed);
                    self_seed = self_seed + 1u;
                    if (rand < weight) {
                        dir = dir_cosine;
                        //final_pdf = max(dot(dir, intersection.normal) / PI, 0.0);
                        last_not_vmf = true;
                    } else {
                        dir = dir_markov;
                        //final_pdf = vmf_pdf(mean, kappa, dir_markov);
                        //sample.color = (dir_markov + 1) / 2;
                        //return sample;
                    }
                    let cosine_pdf = max(dot(dir, intersection.normal) / PI, 0.0);
                    let avg_pdf = ((confidences[0] * cosine_pdf) + select(confidences[1] * all_vmf_pdfs(ray.origin, dir), 0.0, confidences[1] == 0.0));
                    final_pdf = avg_pdf;
                    norm_divisor = avg_pdf;
                    //vmf_weight *= avg_pdf;
                    //final_pdf = 1.0;
                    // Calculate final pdf
                    let diffuse = make_diffuse(intersection.color, ray, intersection.normal, intersection.tangent, intersection.roughness.x, dir);
                    //let dot = max(dot(dir, normal), 0.0) / dot(dir, intersection.normal);
                    //color = color * dot;
                    //ray.direction = dir;
                    /*if (diffuse.brightness < 0.0) {
                        sample.color = vec3f(0.0, 1.0, 0.0);
                        return sample;
                    }*/
                    cached_pdf = final_pdf;
                    let normalization_factor = abs(dot(dir, intersection.normal) / PI) / norm_divisor;
                    color = color * normalization_factor * diffuse;
                    if (!(i_diffuse == 0u)) {
                        out_radiance_colour = out_radiance_colour * normalization_factor * diffuse;
                    }
                    ray.direction = dir;
                    previously_diffuse = true;
                    if (i == 0) {
                        sample.pdf = final_pdf;
                    }
                    //sample.color = vec3f((diffuse.w));
                    //sample.color = ray.direction;
                    //return sample;
                } else if (intersection.ty == TRANSPARENT) {
                    //let k = 1.0 - refractive_index * refractive_index * (1.0 - dot(intersection.normal, ray.direction) * dot(intersection.normal, ray.direction));
                    ray.direction = refract(ray.direction, normal, refractive_index);
                    let dot = dot(-ray.direction, intersection.normal);
                    if (dot < 0.0) {
                        //sample.color = vec3f(1.0);
                        //return sample;
                    }
                    //sample.color = vec3f(sign(k), -sign(k), sign(dot(intersection.normal, ray.direction)));
                    //sample.color = vec3f((refractive_index * refractive_index) / 8.0, (1.0 - dot(intersection.normal, ray.direction) * dot(intersection.normal, ray.direction)), f32((1.0 - dot(intersection.normal, ray.direction) * dot(intersection.normal, ray.direction)) > 0.25));
                    //sample.color = -normal;
                    //return sample;
                }
            } else {
                ray.direction = reflect(ray.direction, normal);
                let dot = dot(ray.direction, intersection.normal);
                if (dot < 0.0) {
                    // if we would have hit the surface again, bounce off the hit surface
                    ray.direction = reflect(ray.direction, intersection.normal);
                }
                //sample.color = -normal;
                //sample.color = vec3f(1.0);
                //return sample;
            }
            if (((intersection.ty == DIFFUSE) && !reflecting) || (i_diffuse != 0u)) {

            }
            i_diffuse++;
            if (ray.direction.x == 0.0 && ray.direction.y == 0.0 && ray.direction.z == 0.0) {
                //sample.color = vec3f(1.0);
                //return sample;
            }
            //sample.color = ray.direction;
            //return sample;
        } else {
            let bg_color = background(&ray);
            // if it's our first hit it doesn't matter, but we should resample the sky sometimes
            if (i == 1u) {
                sample.visible_point = at(&ray, T_MAX);
                sample.visible_normal = -ray.direction;
            }
            if (!(i == 0u)) {
                sample.out_radiance = fma(out_radiance_colour, bg_color, sample.out_radiance);
            }
            sample.color = fma(color, bg_color, sample.color);
            break;
        }
    }
    sample.color /= vmf_weight;
    //let tex_coord = vec2<u32>((vec2<f32>(coord) / vec2<f32>(textureDimensions(output))) * 64.0);
    //sample.color = textureLoad(tex_attributes, tex_coord, 0u, 0).xyz;
    //sample.color = vec3<f32>(vec2<f32>(tex_sizes[1u].size[1]), 0.0);
    //sample.pdf = saturate(sample.pdf);
    sample.valid = true;//(i_diffuse > 1u);
    return sample;
}

fn all_vmf_pdfs(hit:vec3<f32>, direction:vec3<f32>) -> f32 {
    var pdfs = 0.0;
    if (sum_mc > 0.0) {
        for (var i = 0; i < NUM_RESAMPLES; i++) {
            let s = other_pixels_marcov_chain_states.state[i];
            if (!marcov_chain_valid(s)) {
                continue;
            }
            pdfs += (vmf_pdf(normalize(get_light_dir(s, hit)), get_kappa(s), direction)) * (s.weight_sum / sum_mc);
            //pdfs += vmf_pdf(normalize(s.light_source - hit), get_kappa(s), direction) / f32(NUM_RESAMPLES);
            //pdfs += 1.0;
        }
    }
    return pdfs;
}

fn get_kappa(state: MarcovChainState) -> f32 {
    const Np = 0.2;
    // TODO: use rp mix, that was the best in the paper.
    let rp = 0.9;
    // Not exactly sure what N is supposed to be in this case, but pretty sure it is this.
    let N = state.num_samples;
    let N_sqrd = f32(N * N);
    let r = safe_div(((N_sqrd * (state.mean_cosine / state.weight_sum)) + (Np * rp)), (N_sqrd + Np));
    // We don't want this to be zero or infinity
    // These values are large due to the large amount of floating point inacuracies there are
    //return max((3.0 * r) - (r*r*r) / max(1 - (r*r), 0.0001), 0.1);
    return 1000.0;
}

fn vmf_pdf(mean:vec3<f32>, kappa:f32, direction:vec3<f32>) -> f32 {
    let minus_2_kappa = -2.0 * kappa;
    let fraction = x_divided_by_exp_minus_one(minus_2_kappa) / (4.0 * PI);
    let cos_theta = dot(direction, mean);
    return max(fraction*exp(kappa * (cos_theta - 1.0)), 0.0);
}

fn exp_minus_one(x: f32) -> f32 {
    let u = exp(x);
    if (u == 1.0) {
        return x;
    }
    let y = u - 1.0;
    if (abs(x) < 1.0) {
        return (x * y) / log(u);
    }
    return y;
}

fn x_divided_by_exp_minus_one(x:f32) -> f32 {
    let u = exp(x);
    if (u == 1.0) {
        return 1.0;
    }
    let y = u - 1.0;
    if (abs(x) < 1.0) {
        return log(u) / y;
    }
    return x / y;
}

override T_MIN:f32 = 0.0001;
override T_MAX:f32 = 1000.0;

fn ray_hit(ray: ptr<function, Ray>) -> Intersection {
    var rq: ray_query_maybe_vertex_return;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, T_MIN, T_MAX, (*ray).origin, (*ray).direction));
    var aabb_normal: vec3<f32>;
    var aabb_tangent: vec3<f32>;
    while (rayQueryProceed(&rq)) {
        let intersection = rayQueryGetCandidateIntersection(&rq);
        if (intersection.kind == RAY_QUERY_INTERSECTION_TRIANGLE) {
            let material = materials[material_idx[intersection.instance_index].materials[intersection.primitive_index]];
            let bary = vec3<f32>(intersection.barycentrics, ((1.0 - intersection.barycentrics.x) - intersection.barycentrics.y));
            let tex_coords_float = fma(unpack2x16float(material.tex_pos_1), vec2<f32>(bary.z), fma(unpack2x16float(material.tex_pos_2), vec2<f32>(bary.x), (unpack2x16float(material.tex_pos_3) * bary.y)));
            let idx_diffuse = unpack_2xu16(material.tex_idx_diffuse_emission).x;
            if (!near_zero(textureSampleLevel(tex_diffuse[idx_diffuse], sam, tex_coords_float, intersection.t).w)) {
                rayQueryConfirmIntersection(&rq);
            }
        } else if (intersection.kind == RAY_QUERY_INTERSECTION_AABB) {
            let aabb_intersection = intersect(intersection);
            if (aabb_intersection.hit) {
                aabb_normal = aabb_intersection.normal;
                aabb_tangent = aabb_intersection.tangent;
                let t = aabb_intersection.t;
                rayQueryGenerateIntersection(&rq, t);
            }
        }
    }
    let intersection = rayQueryGetCommittedIntersection(&rq);

    var return_intersection = Intersection();
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        return_intersection.hit = true;
        return_intersection.t = intersection.t;
        return_intersection.front_face = intersection.front_face;

        let material = materials[material_idx[intersection.instance_index].materials[intersection.primitive_index]];
        let idx_diffuse_emission = unpack_2xu16(material.tex_idx_diffuse_emission);
        let idx_attributes_ty = unpack_2xu16(material.tex_idx_attributes_ty);
        return_intersection.ty = idx_attributes_ty.y;
        let ref_idx = unpack2x16float(material.refractive_index);
        return_intersection.refractive_index_low = ref_idx.x;
        return_intersection.refractive_index_high = ref_idx.y;

        var normal: vec3<f32>;
        var tangent: vec3<f32>;
        if (intersection.kind == RAY_QUERY_INTERSECTION_TRIANGLE) {
            let vertices = GET_COMMITTED_VERTEX_POSITIONS;
            tangent = correct_tangent(normalize(vertices[0] - vertices[1]));
            normal = normalize(cross(tangent, vertices[0] - vertices[2]));
        } else if (intersection.kind == RAY_QUERY_INTERSECTION_GENERATED) {
            tangent = aabb_tangent;
            normal = aabb_normal;
        }

        if (dot((*ray).direction, normal) > 0.0) {
            return_intersection.normal = -normal;
        } else {
            return_intersection.normal = normal;
        }
        return_intersection.tangent = tangent;

        let bary = vec3<f32>(intersection.barycentrics, ((1.0 - intersection.barycentrics.x) - intersection.barycentrics.y));
        let tex_coords_float = fma(unpack2x16float(material.tex_pos_1), vec2<f32>(bary.z), fma(unpack2x16float(material.tex_pos_2), vec2<f32>(bary.x), (unpack2x16float(material.tex_pos_3) * bary.y)));

        //return_intersection.color = vec3f(tex_coords_float, 0.0);
        return_intersection.color = textureSampleLevel(tex_diffuse[idx_diffuse_emission.x], sam, tex_coords_float, intersection.t).xyz;
        if (idx_diffuse_emission.y != 0xFFFFu) {
            return_intersection.emission = textureSampleLevel(tex_emission[idx_diffuse_emission.y], sam, tex_coords_float, intersection.t).xyz * material.emission_scale;
            //return_intersection.tri_area = spherical_size(vertices[0] - (*ray).origin, vertices[1] - (*ray).origin, vertices[2] - (*ray).origin);
        }
        if (idx_attributes_ty.x != 0xFFFFu) {
            let attributes = textureSampleLevel(tex_attributes[idx_attributes_ty.x], sam, tex_coords_float, intersection.t);
            return_intersection.roughness = attributes.xy;
            return_intersection.roughness_transparent = attributes.zw;
        }
    }
    return return_intersection;
}

fn correct_tangent(tangent: vec3<f32>) -> vec3<f32> {
    var dir = tangent;
    let abs_dir = abs(dir);
    let x_largest = abs_dir.x > abs_dir.y && abs_dir.x > abs_dir.z;
    let y_largest = abs_dir.y > abs_dir.x && abs_dir.y > abs_dir.z;
    var reverse_dir: bool;
    if (x_largest) {
        reverse_dir = dir.x < 0.0;
    } else if (y_largest) {
        reverse_dir = dir.y < 0.0;
    } else {
        reverse_dir = dir.z < 0.0;
    }
    if (reverse_dir) {
        dir = -dir;
    }
    return dir;
}

fn create_ray(coord: vec2<u32>) -> Ray {
    let screen_size = textureDimensions(output);
    let half_size = (vec2<f32>(screen_size) / 2.0);
    //the coordinates get converted from screenspace to between -1 and 1
    let normalized_coords = (vec2<f32>(coord) / half_size) - 1.0;
    // textures start at the top and go down so reverse
    let reverse_y_coords = vec2<f32>(normalized_coords.x, -normalized_coords.y);
    // since normal projection projects from world coords (rotated and translated already) to -1 and 1
    // inverting it will convert from [-1, 1] to world coords (currently still rotated, translated)
    let projected_coords = normalize(camera.projection_inverse * vec4<f32>(reverse_y_coords, 1.0, 1.0));
    // then we rotate these coordinates, but since the w location controls translation scale this will not translate
    let direction = (camera.view_inverse * vec4<f32>(projected_coords.xyz, 0.0)).xyz;
    // the position 0.0 gets rotated to 0.0, so only the tranlation gets applied
    let position = (camera.view_inverse * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    return Ray(direction, position);
}

// get the location at a distance down the ray
fn at(ray: ptr<function, Ray>, t: f32) -> vec3<f32> {
    return fma((*ray).direction, vec3<f32>(t), (*ray).origin);
}

fn background(ray: ptr<function, Ray>) -> vec3<f32> {
    return textureSampleLevel(bg, sam, (*ray).direction, 0.0).xyz;
}

// from https://math.stackexchange.com/a/1586185
// also see https://mathworld.wolfram.com/SpherePointPicking.html
fn rand_on_sphere(own_seed:u32) -> vec3<f32> {
    let u1 = rand_f32(own_seed);
    let u2 = rand_f32(own_seed + 1u);
    let latitude = acos(fma(2.0, u1, -1.0)) - HALF_PI;
    let longitude = TAU * u2;
    return to_sphere_from_lat_long(latitude, longitude);
}

fn rand_on_hemisphere(own_seed:u32) -> vec3<f32> {
    let u1 = rand_f32(own_seed);
    let u2 = rand_f32(own_seed + 1u);
    let latitude = acos(fma(2.0, u1, -1.0)) - HALF_PI;
    let longitude = TAU * u2;
    return vec3<f32>(cos(latitude) * cos(longitude), abs(sin(latitude)), cos(latitude) * sin(longitude));
}

fn to_sphere_from_lat_long(latitude: f32, longitude: f32) -> vec3<f32> {
    return vec3<f32>(cos(latitude) * cos(longitude), cos(latitude) * sin(longitude), sin(latitude));
}

fn reflectance(cosine:f32, ref_idx:f32) -> f32 {
    let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0_squared = r0*r0;
    return fma((1.0 - r0_squared), pow((1.0 - cosine), 5.0), r0_squared);
}

// converts from a orthonormal basis to Carteasian coordinates
fn from_onb(src:vec3<f32>, up:vec3<f32>, tangent: vec3<f32>) -> vec3<f32> {
    let s = tangent;
    let t = cross(up, s);
    return fma(vec3<f32>(src.x), s, fma(vec3<f32>(src.y), up, (src.z * t)));
}

fn from_onb_no_tangent(src:vec3<f32>, up:vec3<f32>) -> vec3<f32> {
    var other_angle: vec3<f32>;
    if (near_zero(up.x)) {
        other_angle = vec3(1.0, 0.0, 0.0);
    } else {
        other_angle = vec3(0.0, 1.0, 0.0);
    }
    let tangent = normalize(cross(up, other_angle));
    return from_onb(src, up, tangent);
}

fn onb_cosine_weighted_hemisphere(own_seed:u32) -> vec3<f32> {
    let r1 = rand_f32(own_seed);
    let r2 = rand_f32(own_seed + 1u);

    let phi = TAU * r1;
    let x = cos(phi) * sqrt(r2);
    let z = sin(phi) * sqrt(r2);
    let y = sqrt(1.0 - r2);
    return vec3<f32>(x, y, z);
}

fn cosine_weighted_hemisphere(own_seed:u32, normal: vec3<f32>, tangent:vec3<f32>) -> vec3<f32> {
    let hemisphere = onb_cosine_weighted_hemisphere(own_seed);
    return from_onb(hemisphere, normal, tangent);
}

fn sample_vndf_ggx(own_seed:u32, view_direction: vec3<f32>, roughness:vec2<f32>) -> vec3<f32> {
    let u1 = rand_f32(own_seed);
    let u2 = rand_f32(own_seed + 1u);

    var in_direction = vec3<f32>(view_direction.x * roughness.x, view_direction.y, view_direction.z * roughness.y);

    // minimize floating point error
    if (vec3_near_zero(in_direction)) {
        in_direction = vec3<f32>(0.0, -1.0, 0.0);
    }

    let warped_dir = normalize(in_direction);

    let wm_std = sample_ggx_interal(u1, u2, warped_dir);

    var out_normal = vec3<f32>(wm_std.x * roughness.x, wm_std.y, wm_std.z * roughness.y);

    if (vec3_near_zero(out_normal)) {
        out_normal = vec3<f32>(0.0, 1.0, 0.0);
    }

    let wm = normalize(out_normal);

    return wm;
}

fn sample_ggx_interal(u1: f32, u2:f32, wi:vec3<f32>) -> vec3<f32> {
    let phi = TAU * u1;
    let y = fma((1.0 - u2), (1.0 + wi.y), -wi.y);
    let sin_theta = sqrt(clamp(fma(y, -y, 1.0), 0.0, 1.0));
    let x = sin_theta * cos(phi);
    let z = sin_theta * sin(phi);
    let c = vec3<f32>(x, y, z);

    return c + wi;
}

fn oren_nayer_improved(in: vec3<f32>, seed:u32, roughness:f32, normal:vec3<f32>, tangent:vec3<f32>, color:vec3<f32>) -> DiffuseReturn {
    let incoming = -in;
    let reflected = onb_cosine_weighted_hemisphere(seed);
    let brightness = cosine_weighted_to_oren_nayer_improved(incoming, reflected, roughness, color);

    return DiffuseReturn(brightness, from_onb(reflected, normal, tangent), reflected.y);
}

const U16_MAX = 0xFFFFu;

fn normalised_to_u32(normalized:vec3<f32>) -> u32 {
    let v = normalized / (abs(normalized.x) + abs(normalized.y) + abs(normalized.z));
    var out = 0u;
    if (v.y >= 0.0) {
        out = pack2x16snorm(v.xz);
    } else {
        //out = pack2x16unorm(vec2<f32>((encode_axis(v.x)), (encode_axis(v.z))));
        //let xz = (1.0 - abs(v.zx)) * vec2<f32>(float_sign(v.x), float_sign(v.z));
        let xz = vec2<f32>(copy_sign(v.x, 1.0 - abs(v.z)), copy_sign(v.z, 1.0 - abs(v.x)));
        out = pack2x16snorm(xz);
    }
    return out;
}