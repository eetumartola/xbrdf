#version 330 core
const float TAU = 6.28318530717958647692;
const float HALF_PI = 1.57079632679489661923;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;

uniform sampler2D xbrdf_tex;
uniform vec3 camera_pos;
uniform vec3 preview_light;
uniform int mode;
uniform int camera_width;
uniform int camera_height;
uniform int light_width;
uniform int light_height;

out vec4 color;

vec2 dir_to_latlong(vec3 dir) {
    dir = normalize(dir);
    float u = atan(dir.x, dir.z) / TAU + 0.5;
    float v = 1.0 - asin(clamp(dir.y, 0.0, 1.0)) / HALF_PI;
    return vec2(fract(u), clamp(v, 0.0, 1.0));
}

vec4 fetch_atlas(ivec2 p) {
    ivec2 size = textureSize(xbrdf_tex, 0);
    p.x = ((p.x % size.x) + size.x) % size.x;
    p.y = clamp(p.y, 0, size.y - 1);
    return texelFetch(xbrdf_tex, p, 0);
}

vec4 sample_camera_tile(int light_x, int light_y, vec2 camera_uv) {
    float fx = camera_uv.x * float(camera_width) - 0.5;
    float fy = camera_uv.y * float(camera_height) - 0.5;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float tx = fract(fx);
    float ty = fract(fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int wx0 = ((x0 % camera_width) + camera_width) % camera_width;
    int wx1 = ((x1 % camera_width) + camera_width) % camera_width;
    y0 = clamp(y0, 0, camera_height - 1);
    y1 = clamp(y1, 0, camera_height - 1);

    ivec2 base = ivec2(light_x * camera_width, light_y * camera_height);
    vec4 a = fetch_atlas(base + ivec2(wx0, y0));
    vec4 b = fetch_atlas(base + ivec2(wx1, y0));
    vec4 c = fetch_atlas(base + ivec2(wx0, y1));
    vec4 d = fetch_atlas(base + ivec2(wx1, y1));
    return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

vec4 sample_iso_tile(int light_x, int light_y, float camera_v) {
    float fy = camera_v * float(camera_height) - 0.5;
    int y0 = clamp(int(floor(fy)), 0, camera_height - 1);
    int y1 = clamp(y0 + 1, 0, camera_height - 1);
    float ty = fract(fy);
    int x = ((light_x % light_width) + light_width) % light_width;
    int base_y = light_y * camera_height;
    return mix(fetch_atlas(ivec2(x, base_y + y0)), fetch_atlas(ivec2(x, base_y + y1)), ty);
}

vec4 sample_light_grid(vec3 light_dir, vec2 camera_uv, bool isotropic) {
    vec2 light_uv = dir_to_latlong(light_dir);
    float gx = light_uv.x * float(light_width) - 0.5;
    float gy = light_uv.y * float(light_height) - 0.5;
    int x0 = int(floor(gx));
    int y0 = int(floor(gy));
    float tx = fract(gx);
    float ty = fract(gy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    x0 = ((x0 % light_width) + light_width) % light_width;
    x1 = ((x1 % light_width) + light_width) % light_width;
    y0 = clamp(y0, 0, light_height - 1);
    y1 = clamp(y1, 0, light_height - 1);

    vec4 a = isotropic ? sample_iso_tile(x0, y0, camera_uv.y) : sample_camera_tile(x0, y0, camera_uv);
    vec4 b = isotropic ? sample_iso_tile(x1, y0, camera_uv.y) : sample_camera_tile(x1, y0, camera_uv);
    vec4 c = isotropic ? sample_iso_tile(x0, y1, camera_uv.y) : sample_camera_tile(x0, y1, camera_uv);
    vec4 d = isotropic ? sample_iso_tile(x1, y1, camera_uv.y) : sample_camera_tile(x1, y1, camera_uv);
    return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

vec3 stable_perpendicular(vec3 n) {
    vec3 helper = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    return normalize(cross(helper, n));
}

void main() {
    vec3 n = normalize(v_normal);
    vec3 t = normalize(v_tangent);
    vec3 b = normalize(v_bitangent);
    vec3 wo_world = normalize(camera_pos - v_position);
    vec3 wi_world = normalize(preview_light);

    if (mode == 2) {
        float wo_y = dot(wo_world, n);
        float wi_y = dot(wi_world, n);
        if (wo_y <= 0.0 || wi_y <= 0.0) {
            color = vec4(0.015, 0.015, 0.017, 1.0);
            return;
        }

        vec3 view_projected = wo_world - n * wo_y;
        vec3 iso_z = dot(view_projected, view_projected) > 1.0e-8
            ? normalize(view_projected)
            : stable_perpendicular(n);
        vec3 iso_x = normalize(cross(iso_z, n));
        vec3 wi = normalize(vec3(dot(wi_world, iso_x), wi_y, dot(wi_world, iso_z)));

        vec2 camera_uv = vec2(0.5, 1.0 - asin(clamp(wo_y, 0.0, 1.0)) / HALF_PI);
        vec4 response = sample_light_grid(wi, camera_uv, true);
        float rim = 0.08 * pow(1.0 - max(dot(n, wo_world), 0.0), 2.0);
        color = vec4(response.rgb * max(wi_y, 0.0) + vec3(rim), 1.0);
        return;
    }

    vec3 wo = normalize(vec3(dot(wo_world, t), dot(wo_world, n), dot(wo_world, b)));
    vec3 wi = normalize(vec3(dot(wi_world, t), dot(wi_world, n), dot(wi_world, b)));

    if (wo.y <= 0.0 || wi.y <= 0.0) {
        color = vec4(0.015, 0.015, 0.017, 1.0);
        return;
    }

    vec2 camera_uv = dir_to_latlong(wo);
    vec4 response;
    if (mode == 1) {
        response = sample_light_grid(wi, camera_uv, false);
    } else {
        response = sample_camera_tile(0, 0, camera_uv);
    }

    float macro_cosine = max(wi.y, 0.0);
    float rim = 0.08 * pow(1.0 - max(dot(n, wo_world), 0.0), 2.0);
    color = vec4(response.rgb * macro_cosine + vec3(rim), 1.0);
}
