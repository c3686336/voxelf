#version 460

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 f_color;

void main() {
  vec2 c = uv * 4 - vec2(2.0);

  vec2 z = vec2(0.0);

  int i;
  for (i=0;i<100;i++) {
    z = vec2(z.x*z.x - z.y*z.y, 2 * z.x * z.y) + c;

    if (z.x * z.x + z.y * z.y > 4.0) {
      break;
    }
  }

  if (i == 100) {
    f_color = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
  }
}
