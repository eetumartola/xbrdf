const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067154;
const TAU: f32 = 6.28318530717958647692;
const HALF_PI: f32 = 1.57079632679489661923;
const EPSILON: f32 = 0.0000001;
const HIT_EPSILON: f32 = 0.0001;
const STACK_SIZE: u32 = 64u;

struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    normal: vec4<f32>,
    color: vec4<f32>,
};

struct BvhNode {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    child_or_first: u32,
    child_b: u32,
    triangle_count: u32,
    _pad: u32,
};

struct Params {
    width: u32,
    height: u32,
    samples: u32,
    triangle_count: u32,
    node_count: u32,
    max_repeat_radius: u32,
    y_offset: u32,
    active_height: u32,
    sample_offset: u32,
    target_samples: u32,
    tile_min: vec2<f32>,
    tile_size: vec2<f32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    light_dir: vec4<f32>,
    material_color: vec4<f32>,
    material_kind: vec4<u32>,
    material_params: vec4<f32>,
};

struct Hit {
    found: bool,
    t: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    color: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output_pixels: array<vec4<f32>>;

fn direction_from_pixel(x: u32, y: u32) -> vec3<f32> {
    let u = (f32(x) + 0.5) / f32(params.width);
    let v = (f32(y) + 0.5) / f32(params.height);
    let azimuth = (u - 0.5) * TAU;
    let elevation = (1.0 - v) * HALF_PI;
    let horizontal = cos(elevation);
    return normalize(vec3<f32>(
        sin(azimuth) * horizontal,
        sin(elevation),
        cos(azimuth) * horizontal
    ));
}

fn hash_u32(value: u32) -> u32 {
    var x = value;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash_float(value: u32) -> f32 {
    return f32(hash_u32(value) & 0x00ffffffu) / 16777216.0;
}

fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn sample_2d(sample_index: u32, pixel_x: u32, pixel_y: u32) -> vec2<f32> {
    let global_sample = params.sample_offset + sample_index;
    let pixel_seed = pixel_x * 1973u + pixel_y * 9277u + params.target_samples * 26699u;
    let rotation = vec2<f32>(
        hash_float(pixel_seed),
        hash_float(pixel_seed ^ 0x9e3779b9u)
    );
    let base = vec2<f32>(
        (f32(global_sample) + 0.5) / f32(params.target_samples),
        radical_inverse_vdc(global_sample)
    );
    return fract(base + rotation);
}

fn repeat_radius(direction: vec3<f32>, axis: u32) -> i32 {
    let margin = params.material_params.y;
    let height = max(params.bounds_max.y - params.bounds_min.y + 2.0 * margin, margin);
    let dy = max(abs(direction.y), EPSILON);
    var tile_size = params.tile_size.x;
    var lateral = abs(direction.x);
    if (axis == 1u) {
        tile_size = params.tile_size.y;
        lateral = abs(direction.z);
    }
    let repeat = i32(ceil((lateral / dy) * height / tile_size)) + 2;
    return min(repeat, i32(params.max_repeat_radius));
}

fn shadow_repeat_radius(direction: vec3<f32>, axis: u32) -> i32 {
    let margin = params.material_params.y;
    let height = max(params.bounds_max.y - params.bounds_min.y + 2.0 * margin, margin);
    let dy = max(abs(direction.y), EPSILON);
    var tile_size = params.tile_size.x;
    var lateral = abs(direction.x);
    if (axis == 1u) {
        tile_size = params.tile_size.y;
        lateral = abs(direction.z);
    }
    let repeat = i32(ceil((lateral / dy) * height / tile_size));
    return min(repeat, i32(params.max_repeat_radius));
}

fn intersect_aabb_t(origin: vec3<f32>, inv_direction: vec3<f32>, bounds_min: vec3<f32>, bounds_max: vec3<f32>, max_t: f32) -> f32 {
    let tx1 = (bounds_min.x - origin.x) * inv_direction.x;
    let tx2 = (bounds_max.x - origin.x) * inv_direction.x;
    let ty1 = (bounds_min.y - origin.y) * inv_direction.y;
    let ty2 = (bounds_max.y - origin.y) * inv_direction.y;
    let tz1 = (bounds_min.z - origin.z) * inv_direction.z;
    let tz2 = (bounds_max.z - origin.z) * inv_direction.z;

    let tmin = max(max(min(tx1, tx2), min(ty1, ty2)), min(tz1, tz2));
    let tmax = min(min(max(tx1, tx2), max(ty1, ty2)), max(tz1, tz2));
    if (tmax >= max(tmin, 0.0) && tmin <= max_t) {
        return max(tmin, 0.0);
    }
    return 3.402823466e+38;
}

fn intersect_triangle(origin: vec3<f32>, direction: vec3<f32>, triangle: Triangle) -> f32 {
    let v0 = triangle.v0.xyz;
    let v1 = triangle.v1.xyz;
    let v2 = triangle.v2.xyz;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let p = cross(direction, edge2);
    let det = dot(edge1, p);

    if (abs(det) < EPSILON) {
        return -1.0;
    }

    let inv_det = 1.0 / det;
    let tvec = origin - v0;
    let u = dot(tvec, p) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }

    let q = cross(tvec, edge1);
    let v = dot(direction, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }

    let t = dot(edge2, q) * inv_det;
    if (t <= HIT_EPSILON) {
        return -1.0;
    }

    return t;
}

fn trace_bvh(origin: vec3<f32>, direction: vec3<f32>, offset: vec3<f32>) -> Hit {
    var hit = Hit(false, 3.402823466e+38, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(1.0));
    if (params.node_count == 0u) {
        return hit;
    }

    let local_origin = origin - offset;
    let inv_direction = 1.0 / direction;
    var stack: array<u32, 64>;
    var stack_size = 1u;
    stack[0] = 0u;

    loop {
        if (stack_size == 0u) {
            break;
        }

        stack_size = stack_size - 1u;
        let node_index = stack[stack_size];
        if (node_index >= params.node_count) {
            continue;
        }

        let node = bvh_nodes[node_index];
        if (intersect_aabb_t(local_origin, inv_direction, node.bounds_min.xyz, node.bounds_max.xyz, hit.t) == 3.402823466e+38) {
            continue;
        }

        if (node.triangle_count > 0u) {
            for (var i = 0u; i < node.triangle_count; i = i + 1u) {
                let triangle_index = node.child_or_first + i;
                if (triangle_index < params.triangle_count) {
                    let t = intersect_triangle(local_origin, direction, triangles[triangle_index]);
                    if (t > 0.0 && t < hit.t) {
                        hit.found = true;
                        hit.t = t;
                        hit.position = origin + direction * t;
                        hit.normal = normalize(triangles[triangle_index].normal.xyz);
                        hit.color = triangles[triangle_index].color.xyz;
                    }
                }
            }
        } else {
            if (stack_size + 2u <= STACK_SIZE) {
                let left = bvh_nodes[node.child_or_first];
                let right = bvh_nodes[node.child_b];
                let left_t = intersect_aabb_t(local_origin, inv_direction, left.bounds_min.xyz, left.bounds_max.xyz, hit.t);
                let right_t = intersect_aabb_t(local_origin, inv_direction, right.bounds_min.xyz, right.bounds_max.xyz, hit.t);
                if (left_t < right_t) {
                    stack[stack_size] = node.child_b;
                    stack[stack_size + 1u] = node.child_or_first;
                } else {
                    stack[stack_size] = node.child_or_first;
                    stack[stack_size + 1u] = node.child_b;
                }
                stack_size = stack_size + 2u;
            }
        }
    }

    return hit;
}

fn any_bvh_hit(origin: vec3<f32>, direction: vec3<f32>, offset: vec3<f32>) -> bool {
    if (params.node_count == 0u) {
        return false;
    }

    let local_origin = origin - offset;
    let inv_direction = 1.0 / direction;
    var stack: array<u32, 64>;
    var stack_size = 1u;
    stack[0] = 0u;

    loop {
        if (stack_size == 0u) {
            break;
        }

        stack_size = stack_size - 1u;
        let node_index = stack[stack_size];
        if (node_index >= params.node_count) {
            continue;
        }

        let node = bvh_nodes[node_index];
        if (intersect_aabb_t(local_origin, inv_direction, node.bounds_min.xyz, node.bounds_max.xyz, 3.402823466e+38) == 3.402823466e+38) {
            continue;
        }

        if (node.triangle_count > 0u) {
            for (var i = 0u; i < node.triangle_count; i = i + 1u) {
                let triangle_index = node.child_or_first + i;
                if (triangle_index < params.triangle_count) {
                    let t = intersect_triangle(local_origin, direction, triangles[triangle_index]);
                    if (t > 0.0) {
                        return true;
                    }
                }
            }
        } else {
            if (stack_size + 2u <= STACK_SIZE) {
                stack[stack_size] = node.child_or_first;
                stack[stack_size + 1u] = node.child_b;
                stack_size = stack_size + 2u;
            }
        }
    }

    return false;
}

fn trace_periodic(origin: vec3<f32>, direction: vec3<f32>) -> Hit {
    var hit = Hit(false, 3.402823466e+38, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(1.0));
    let rx = repeat_radius(direction, 0u);
    let rz = repeat_radius(direction, 1u);

    for (var oz = -rz; oz <= rz; oz = oz + 1) {
        for (var ox = -rx; ox <= rx; ox = ox + 1) {
            let offset = vec3<f32>(
                f32(ox) * params.tile_size.x,
                0.0,
                f32(oz) * params.tile_size.y
            );
            let candidate = trace_bvh(origin, direction, offset);
            if (candidate.found && candidate.t < hit.t) {
                hit = candidate;
            }
        }
    }

    return hit;
}

fn any_shadow_hit(origin: vec3<f32>, direction: vec3<f32>) -> bool {
    let rx = shadow_repeat_radius(direction, 0u);
    let rz = shadow_repeat_radius(direction, 1u);

    for (var oz = -rz; oz <= rz; oz = oz + 1) {
        for (var ox = -rx; ox <= rx; ox = ox + 1) {
            let offset = vec3<f32>(
                f32(ox) * params.tile_size.x,
                0.0,
                f32(oz) * params.tile_size.y
            );
            if (any_bvh_hit(origin, direction, offset)) {
                return true;
            }
        }
    }

    return false;
}

fn evaluate_material(n: vec3<f32>, light_dir: vec3<f32>, wo: vec3<f32>) -> vec3<f32> {
    let n_dot_l = max(dot(n, light_dir), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }

    let inv_macro_irradiance = params.material_params.x;
    if (params.material_kind.x == 1u) {
        let exponent = max(params.material_params.w, 1.0);
        let mirror = normalize(2.0 * n_dot_l * n - light_dir);
        let lobe = pow(max(dot(mirror, wo), 0.0), exponent);
        let brdf = ((exponent + 2.0) / (2.0 * PI)) * lobe;
        return params.material_color.xyz * (n_dot_l * brdf * inv_macro_irradiance);
    }

    return params.material_color.xyz * (n_dot_l * INV_PI * inv_macro_irradiance);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let local_y = id.y;
    if (x >= params.width || local_y >= params.active_height) {
        return;
    }

    let y = params.y_offset + local_y;
    if (y >= params.height) {
        return;
    }

    let out_index = y * params.width + x;
    let wo = direction_from_pixel(x, y);
    let ray_direction = -wo;
    let margin = params.material_params.y;
    let top_y = params.bounds_max.y + margin;
    let target_y = params.bounds_min.y - margin;
    var sum = vec3<f32>(0.0);

    for (var sample = 0u; sample < params.samples; sample = sample + 1u) {
        let uv = sample_2d(sample, x, y);
        let sample_target = vec3<f32>(
            params.tile_min.x + uv.x * params.tile_size.x,
            target_y,
            params.tile_min.y + uv.y * params.tile_size.y
        );
        let travel = (top_y - sample_target.y) / max(wo.y, EPSILON);
        let origin = sample_target + wo * travel;
        let hit = trace_periodic(origin, ray_direction);

        if (hit.found) {
            let n = hit.normal;
            if (dot(n, wo) > 0.0) {
                let contribution = hit.color * evaluate_material(n, params.light_dir.xyz, wo);
                if (any(contribution > vec3<f32>(0.0))) {
                    let shadow_origin = hit.position + params.light_dir.xyz * HIT_EPSILON * 4.0;
                    if (!any_shadow_hit(shadow_origin, params.light_dir.xyz)) {
                        sum = sum + contribution;
                    }
                }
            }
        }
    }

    let rgb = sum / f32(params.samples);
    output_pixels[out_index] = vec4<f32>(rgb, 1.0);
}
