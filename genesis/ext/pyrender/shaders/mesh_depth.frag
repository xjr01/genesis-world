#version 330 core

out vec4 frag_color;

void main()
{
#ifdef ROUND_POINTS
    // Same sprite clipping as mesh.frag, so depth-only passes and shadow maps see the round particle outline too.
    vec2 sprite_coord = gl_PointCoord - vec2(0.5);
    if (dot(sprite_coord, sprite_coord) > 0.25) discard;
#endif

    frag_color = vec4(1.0);
}
