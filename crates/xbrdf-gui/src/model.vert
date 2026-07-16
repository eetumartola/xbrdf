#version 330 core
in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec3 bitangent;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;

void main() {
    vec4 world = model * vec4(position, 1.0);
    mat3 basis = mat3(model);
    v_position = world.xyz;
    v_normal = normalize(basis * normal);
    v_tangent = normalize(basis * tangent);
    v_bitangent = normalize(basis * bitangent);
    gl_Position = projection * view * world;
}
