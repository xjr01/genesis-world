// Camera projection, software depth buffering and material picking.
export const add = (a, b) => a.map((v, i) => v + b[i]);
export const sub = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const scale = (a, s) => a.map((v) => v * s);
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const unit = (a) => scale(a, 1 / Math.hypot(...a));

export function multiply(a, b) {
  const [w, x, y, z] = a,
    [v, i, j, k] = b;
  return [
    w * v - x * i - y * j - z * k,
    w * i + x * v + y * k - z * j,
    w * j - x * k + y * v + z * i,
    w * k + x * j - y * i + z * v,
  ];
}
export function rotate(q, p) {
  return multiply(multiply(q, [0, ...p]), [q[0], -q[1], -q[2], -q[3]]).slice(1);
}
export function localRotation(q, angles) {
  for (let axis = 0; axis < 3; axis++) {
    const half = (angles[axis] * Math.PI) / 360,
      r = [Math.cos(half), 0, 0, 0];
    r[axis + 1] = Math.sin(half);
    q = multiply(q, r);
  }
  return q;
}
export function barycentric(x, y, A, B, C) {
  const det = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]);
  if (Math.abs(det) < 1e-10) return null;
  const u = ((B[1] - C[1]) * (x - C[0]) + (C[0] - B[0]) * (y - C[1])) / det;
  const v = ((C[1] - A[1]) * (x - C[0]) + (A[0] - C[0]) * (y - C[1])) / det;
  return [u, v, 1 - u - v];
}

export class SceneView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.mode = "xy";
    this.color = "height";
    this.paths = null;
    this.pose = null;
    this.robot = null;
    this.node = null;
  }
  project(v) {
    if (this.mode === "perspective") {
      const delta = sub(v, this.camera.pos_m),
        z = dot(delta, this.forward);
      return [
        this.width / 2 + (dot(delta, this.right) * this.focal) / z,
        this.height / 2 - (dot(delta, this.up) * this.focal) / z,
        z,
      ];
    }
    const [a, b, d] = this.axes;
    return [
      this.left + (v[a] - this.minA) * this.zoom,
      this.top + (this.maxB - v[b]) * this.zoom,
      -v[d],
    ];
  }
  dragPosition(x, y, original) {
    if (this.mode !== "perspective") {
      const p = [...original],
        [a, b] = this.axes;
      p[a] = (x - this.left) / this.zoom + this.minA;
      p[b] = this.maxB - (y - this.top) / this.zoom;
      return p;
    }
    const ray = add(
      this.forward,
      add(
        scale(this.right, (x - this.width / 2) / this.focal),
        scale(this.up, -(y - this.height / 2) / this.focal),
      ),
    );
    if (Math.abs(ray[2]) < 1e-8)
      throw Error("Camera ray is parallel to the editing plane.");
    return add(
      this.camera.pos_m,
      scale(ray, (original[2] - this.camera.pos_m[2]) / ray[2]),
    );
  }
  configure() {
    this.width = this.canvas.clientWidth;
    this.height = this.canvas.clientHeight;
    this.canvas.width = Math.round(this.width);
    this.canvas.height = Math.round(this.height);
    this.camera = this.data.camera;
    this.forward = unit(sub(this.camera.lookat_m, this.camera.pos_m));
    this.right = unit(cross(this.forward, this.camera.up));
    this.up = cross(this.right, this.forward);
    this.focal =
      this.height /
      (2 * Math.tan((this.camera.vertical_fov_deg * Math.PI) / 360));
    this.axes = { xy: [0, 1, 2], xz: [0, 2, 1], yz: [1, 2, 0] }[this.mode];
    if (!this.axes) return;
    const [a, b] = this.axes;
    let minA = Infinity,
      maxA = -Infinity,
      minB = Infinity,
      maxB = -Infinity;
    const vertices = [...this.data.vertices];
    if (this.paths) vertices.push(...this.paths.left, ...this.paths.right);
    if (this.pose)
      for (const f of this.pose.fingers) vertices.push(...f.vertices_world_m);
    if (this.robot)
      for (const f of this.robot) vertices.push(...f.vertices_world_m);
    for (const v of vertices) {
      minA = Math.min(minA, v[a]);
      maxA = Math.max(maxA, v[a]);
      minB = Math.min(minB, v[b]);
      maxB = Math.max(maxB, v[b]);
    }
    this.zoom = Math.min(
      (this.width - 60) / Math.max(maxA - minA, 0.01),
      (this.height - 60) / Math.max(maxB - minB, 0.01),
    );
    this.minA = minA;
    this.maxB = maxB;
    this.left = (this.width - (maxA - minA) * this.zoom) / 2;
    this.top = (this.height - (maxB - minB) * this.zoom) / 2;
  }
  draw() {
    if (!this.data) return;
    this.configure();
    const ctx = this.ctx,
      image = ctx.createImageData(this.canvas.width, this.canvas.height);
    const depth = new Float64Array(this.canvas.width * this.canvas.height);
    depth.fill(Infinity);
    const palette = [
      [72, 174, 201],
      [110, 198, 133],
      [230, 166, 97],
    ];
    const mesh = (verts, faces, colors) => {
      for (let index = 0; index < faces.length; index++) {
        const face = faces[index],
          ps = face.map((i) => verts[i]);
        if (this.mode === "perspective" && ps.some((p) => p[2] <= 0.001))
          continue;
        const [A, B, C] = ps,
          area = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]);
        if (Math.abs(area) < 1e-9) continue;
        const minX = Math.max(0, Math.floor(Math.min(A[0], B[0], C[0]))),
          maxX = Math.min(
            this.canvas.width - 1,
            Math.ceil(Math.max(A[0], B[0], C[0])),
          );
        const minY = Math.max(0, Math.floor(Math.min(A[1], B[1], C[1]))),
          maxY = Math.min(
            this.canvas.height - 1,
            Math.ceil(Math.max(A[1], B[1], C[1])),
          );
        const color = colors(index);
        for (let y = minY; y <= maxY; y++)
          for (let x = minX; x <= maxX; x++) {
            const u =
              ((B[1] - C[1]) * (x + 0.5 - C[0]) +
                (C[0] - B[0]) * (y + 0.5 - C[1])) /
              area;
            const v =
                ((C[1] - A[1]) * (x + 0.5 - C[0]) +
                  (A[0] - C[0]) * (y + 0.5 - C[1])) /
                area,
              w = 1 - u - v;
            if (u < 0 || v < 0 || w < 0) continue;
            const z =
              this.mode === "perspective"
                ? 1 / (u / A[2] + v / B[2] + w / C[2])
                : u * A[2] + v * B[2] + w * C[2];
            const pixel = y * this.canvas.width + x;
            if (z >= depth[pixel]) continue;
            depth[pixel] = z;
            for (let channel = 0; channel < 3; channel++)
              image.data[pixel * 4 + channel] = color[channel];
            image.data[pixel * 4 + 3] = 255;
          }
      }
    };
    const z = this.data.vertices.map((v) => v[2]);
    let low = Infinity,
      high = -Infinity;
    for (const value of z) {
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
    mesh(
      this.data.vertices.map((v) => this.project(v)),
      this.data.faces,
      (index) => {
        const face = this.data.faces[index];
        if (this.color !== "height")
          return palette[this.data.atlas[this.color][face[0]]];
        const value = face.reduce((sum, i) => sum + z[i], 0) / 3,
          t = (value - low) / Math.max(high - low, 0.001);
        return [55 + 45 * t, 110 + 60 * t, 160 + 40 * t];
      },
    );
    if (this.robot)
      for (const f of this.robot)
        mesh(
          f.vertices_world_m.map((v) => this.project(v)),
          f.faces,
          () => [113, 126, 145],
        );
    if (this.pose)
      for (const f of this.pose.fingers)
        mesh(
          f.vertices_world_m.map((v) => this.project(v)),
          f.faces,
          () => [225, 192, 105],
        );
    ctx.putImageData(image, 0, 0);
    if (this.paths)
      for (const [hand, color] of [
        ["left", "#63dfaa"],
        ["right", "#ff8c7d"],
      ]) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        this.paths[hand].forEach((v, i) => {
          const p = this.project(v);
          i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
        });
        ctx.stroke();
        for (
          let i = 0;
          i < this.paths[hand].length;
          i += Math.max(1, Math.floor(this.paths[hand].length / 25))
        )
          this.marker(this.paths[hand][i], color, 3);
      }
    if (this.node && this.paths)
      this.marker(this.paths[this.node.hand][this.node.index], "#fff", 6);
    if (this.selections)
      for (const [hand, p] of Object.entries(this.selections.points)) {
        const face = this.data.faces[p.face_index];
        const position =
          face && p.barycentric
            ? [0, 1, 2].map((axis) =>
                face.reduce(
                  (sum, vertex, i) =>
                    sum + p.barycentric[i] * this.data.vertices[vertex][axis],
                  0,
                ),
              )
            : p.material_world_m;
        this.marker(position, hand === "left" ? "#63dfaa" : "#ff8c7d", 6);
      }
    if (this.pose)
      for (const [axis, color] of [
        [0, "#ff6b6b"],
        [1, "#63dfaa"],
        [2, "#6f9fff"],
      ]) {
        const direction = [0, 0, 0];
        direction[axis] = 0.07;
        const a = this.project(this.pose.position_m),
          b = this.project(
            add(this.pose.position_m, rotate(this.pose.quat_wxyz, direction)),
          );
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.fillText("XYZ"[axis], b[0], b[1]);
      }
    ctx.fillStyle = "#d1e6f3";
    ctx.fillText(
      `Source frame ${this.data.source_frame} | ${this.mode} | Green: left, coral: right`,
      15,
      23,
    );
  }
  marker(v, color, radius) {
    const p = this.project(v);
    if (this.mode === "perspective" && p[2] <= 0.001) return;
    this.ctx.fillStyle = color;
    this.ctx.beginPath();
    this.ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
    this.ctx.fill();
  }
  pick(x, y) {
    const candidates = [];
    this.data.faces.forEach((face, faceIndex) => {
      const ps = face.map((i) => this.project(this.data.vertices[i]));
      if (this.mode === "perspective" && ps.some((p) => p[2] <= 0.001)) return;
      let weights = barycentric(x, y, ...ps);
      if (!weights || Math.min(...weights) < -1e-7) return;
      if (this.mode === "perspective") {
        weights = weights.map((w, i) => w / ps[i][2]);
        const sum = weights.reduce((a, b) => a + b, 0);
        weights = weights.map((w) => w / sum);
      }
      const interpolate = (vertices) =>
        [0, 1, 2].map((axis) =>
          face.reduce((s, j, k) => s + weights[k] * vertices[j][axis], 0),
        );
      const world = interpolate(this.data.vertices),
        rest = interpolate(this.data.rest_vertices);
      candidates.push({
        face_index: faceIndex,
        barycentric: weights,
        material_world_m: world,
        rest_raw_xyz: rest,
        source_frame: this.data.source_frame,
        depth: this.project(world)[2],
        tag_vertex: face[weights.indexOf(Math.max(...weights))],
      });
    });
    return candidates.sort((a, b) => a.depth - b.depth);
  }
}
