import { SceneView, add, sub, localRotation, rotate } from "./render.js";

const $ = (id) => document.getElementById(id);
const view = new SceneView($("cloth"));
let data = null,
  candidates = [],
  currentPath = null,
  sourcePose = null,
  drag = null;
const selections = {
  schema: "workbench-material-regrasp-v1",
  source_frame: 0,
  points: {},
  placement_targets_world_m: {},
};
const status = (message) => ($("status").textContent = message);
const json = (value) => JSON.stringify(value, null, 2);

async function request(path, value) {
  const response = await fetch(
    path,
    value === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(value),
        },
  );
  const result = await response.json();
  if (!response.ok) throw Error(result.error || "Request failed");
  return result;
}
function action(button, task) {
  $(button).onclick = async () => {
    $(button).disabled = true;
    try {
      await task();
    } catch (error) {
      status(error.message);
    } finally {
      $(button).disabled = false;
    }
  };
}
function updateActions() {
  const plan = JSON.parse($("plan").value);
  $("poseAction").replaceChildren(
    ...plan.actions.map((a) => new Option(a.id, a.id)),
  );
}
async function load(index = 0) {
  data = await request("/api/state?index=" + index);
  if (!$("frame").options.length) {
    $("frame").replaceChildren(
      ...data.frames.map((frame, i) => new Option(frame, i)),
    );
    $("plan").value = json(data.plan);
    selections.source_frame = data.start_frame;
    selections.cloth_sha256 = data.cloth_sha256;
    updateActions();
    $("reference").disabled = !data.reference_path;
  }
  $("frame").value = index;
  view.data = data;
  view.pose = null;
  sourcePose = null;
  view.robot = null;
  $("showRobot").checked = false;
  view.selections = selections;
  view.draw();
  status(
    `Frame ${data.source_frame}; plan anchor ${data.start_frame}. ${data.is_physics_ready ? "Native execution state." : "Observation workspace; physics requires a matching native checkpoint."}`,
  );
}
function showPoints() {
  $("points").textContent = json(selections);
  view.draw();
}
function setPath(result) {
  currentPath = result.compiled;
  view.paths = {
    left: result.left_pos,
    right: result.right_pos,
    frames: result.source_frames,
  };
  $("nodeFrame").replaceChildren(
    ...result.source_frames.map((frame, i) => new Option(frame, i)),
  );
  $("nodeFrame").value = Math.min(30, result.source_frames.length - 1);
  $("stageStart").value = result.source_frames[0];
  $("stageEnd").value = result.source_frames.at(-1);
  selectNode();
  status(
    `PATH ${result.accepted ? "accepted" : "rejected"}\n${result.saved || currentPath}${result.required_duration_scale ? `\nSuggested duration multiplier: ${result.required_duration_scale.toFixed(3)}` : ""}`,
  );
}
function selectNode() {
  if (!view.paths) return;
  view.node = { hand: $("hand").value, index: Number($("nodeFrame").value) };
  view.draw();
}
function appendCorrection() {
  if (!view.paths) throw Error("Compile or load a path first.");
  const edits = JSON.parse($("edits").value),
    frame = view.paths.frames[Number($("nodeFrame").value)];
  const delta = ["dx", "dy", "dz"].map((id) => Number($(id).value) / 1000);
  const node = { hand: $("hand").value, frame, delta_m: delta };
  const key =
    $("editMode").value === "semantic" ? "semantic_keyframes" : "edits";
  edits[key] ||= [];
  const existing = edits[key].findIndex(
    (edit) => edit.hand === node.hand && edit.frame === node.frame,
  );
  if (key === "edits") {
    node.support_radius = Number($("radius").value);
    node.mode = $("editMode").value;
  }
  if (existing < 0) edits[key].push(node);
  else edits[key][existing] = node;
  edits.stage_range = [
    Number($("stageStart").value),
    Number($("stageEnd").value),
  ];
  $("edits").value = json(edits);
  status("Correction recorded. Save corrected path to compile it.");
}
function screenPoint(event) {
  const rect = $("cloth").getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}
$("cloth").onpointerdown = (event) => {
  if ($("mode").value !== "node" || !view.paths) return;
  const [x, y] = screenPoint(event),
    hand = $("hand").value;
  let best = -1,
    distance = 12;
  const selected = Number($("nodeFrame").value);
  const marker = view.project(view.paths[hand][selected]);
  // Stationary samples overlap; the timeline selection identifies the intended node.
  if (Math.hypot(x - marker[0], y - marker[1]) < distance) {
    best = selected;
  } else {
    view.paths[hand].forEach((position, index) => {
      const point = view.project(position),
        d = Math.hypot(x - point[0], y - point[1]);
      if (d < distance) {
        distance = d;
        best = index;
      }
    });
  }
  if (best < 0) return;
  if (best === 0) {
    status("The starting TCP stays fixed.");
    return;
  }
  $("nodeFrame").value = best;
  selectNode();
  drag = { original: [...view.paths[hand][best]], hand, index: best };
  $("cloth").setPointerCapture(event.pointerId);
};
$("cloth").onpointermove = (event) => {
  if (!drag) return;
  try {
    const position = view.dragPosition(...screenPoint(event), drag.original);
    const delta = sub(position, drag.original);
    ["dx", "dy", "dz"].forEach(
      (id, i) => ($(id).value = (delta[i] * 1000).toFixed(3)),
    );
    // Keep the fitted camera stable while dragging by drawing only the marker.
    view.marker(position, "#ffffff", 4);
  } catch (error) {
    status(error.message);
  }
};
$("cloth").onpointerup = () => {
  if (!drag) return;
  drag = null;
  try {
    appendCorrection();
    view.draw();
  } catch (error) {
    status(error.message);
  }
};
$("cloth").onpointercancel = () => {
  drag = null;
  view.draw();
};
$("cloth").onclick = (event) => {
  if ($("mode").value === "node") return;
  const [x, y] = screenPoint(event);
  if ($("mode").value === "placement") {
    if (!["xy", "perspective"].includes(view.mode)) {
      status("Use overhead or camera view for placement.");
      return;
    }
    const p = view.dragPosition(x, y, [0, 0, Number($("z").value) / 1000]);
    ["x", "y", "z"].forEach(
      (id, i) => ($(id).value = (p[i] * 1000).toFixed(3)),
    );
    return;
  }
  candidates = view.pick(x, y);
  $("layers").replaceChildren();
  const width = ["negative X", "center", "positive X"],
    length = ["hem", "torso", "shoulder/collar"],
    surface = ["table-up material", "table-facing material"];
  candidates.forEach((p, i) => {
    const vertex = p.tag_vertex,
      tags = [
        width[data.atlas.width_band[vertex]],
        length[data.atlas.length_zone[vertex]],
        surface[data.atlas.surface_layer[vertex]],
      ];
    $("layers").add(
      new Option(
        `Face ${p.face_index} | Z ${(p.material_world_m[2] * 1000).toFixed(2)} mm | ${tags.join(", ")}`,
        i,
      ),
    );
  });
  status(
    `${candidates.length} cloth candidates. Confirm the intended folded layer.`,
  );
};
action("confirm", async () => {
  if (data.source_frame !== data.start_frame)
    throw Error("Switch to the plan start before selecting a grasp.");
  const candidate = candidates[Number($("layers").value)];
  if (!candidate) return;
  const { depth, tag_vertex, ...point } = candidate;
  selections.points[$("hand").value] = point;
  showPoints();
});
action("placement", async () => {
  const p = ["x", "y", "z"].map((id) => Number($(id).value) / 1000);
  if (["x", "y", "z"].some((id) => !$(id).value) || !p.every(Number.isFinite))
    throw Error("Enter finite placement XYZ.");
  selections.placement_targets_world_m[$("hand").value] = p;
  showPoints();
});
action("save", async () =>
  status("Saved " + (await request("/api/points", selections)).saved),
);
action("compile", async () => {
  setPath(await request("/api/compile", JSON.parse($("plan").value)));
  updateActions();
});
action("reference", async () =>
  setPath(await request("/api/path", { compiled: data.reference_path })),
);
action("addNode", async () => appendCorrection());
action("clearNodes", async () => {
  $("edits").value = json({ edits: [], semantic_keyframes: [] });
  status("Corrections cleared.");
});
action("applyNodes", async () => {
  if (!currentPath) throw Error("Compile or load a path first.");
  const edits = JSON.parse($("edits").value);
  edits.stage_range = [
    Number($("stageStart").value),
    Number($("stageEnd").value),
  ];
  setPath(await request("/api/edit", { compiled: currentPath, edits }));
  $("edits").value = json({ edits: [], semantic_keyframes: [] });
});
action("draft", async () => {
  const plan = await request("/api/default-plan", {
    plan: JSON.parse($("plan").value),
    pull_axis: $("pullAxis").value,
    pull_sign: Number($("pullSign").value),
  });
  $("plan").value = json(plan);
  updateActions();
  status("Draft ready. Review it, then compile.");
});
function updatePose() {
  if (!sourcePose) return;
  const quat = localRotation(
    sourcePose.quat_wxyz,
    ["rx", "ry", "rz"].map((id) => Number($(id).value)),
  );
  view.pose = {
    ...sourcePose,
    quat_wxyz: quat,
    fingers: sourcePose.fingers.map((f) => ({
      ...f,
      vertices_world_m: f.vertices_local_m.map((p) =>
        add(sourcePose.position_m, rotate(quat, p)),
      ),
    })),
  };
  $("poseInfo").textContent =
    `World WXYZ: ${quat.map((v) => v.toFixed(6)).join(", ")}\n${sourcePose.geometry_source}\nPreview only; check IK and physics after applying.`;
  view.draw();
}
action("preview", async () => {
  sourcePose = await request(
    `/api/pose?index=${$("frame").value}&hand=${$("hand").value}`,
  );
  ["rx", "ry", "rz"].forEach((id) => ($(id).value = 0));
  updatePose();
  status("Measured gripper pose loaded.");
});
action("poseApply", async () => {
  if (!view.pose) throw Error("Load a measured pose first.");
  const plan = JSON.parse($("plan").value),
    action = plan.actions.find((a) => a.id === $("poseAction").value);
  if (!action) throw Error("Select an action.");
  action[$("hand").value].quat = view.pose.quat_wxyz;
  $("plan").value = json(plan);
  status("Pose applied. Compile the updated plan before IK.");
});
action("poseSave", async () => {
  if (!view.pose) throw Error("Load a measured pose first.");
  const result = await request("/api/pose", {
    source_frame: data.source_frame,
    hand: $("hand").value,
    role: $("mode").value === "placement" ? "placement" : "grasp",
    quat_wxyz: view.pose.quat_wxyz,
  });
  status("Saved " + result.saved);
});
action("ik", async () => {
  if (!currentPath) throw Error("Compile or save a path first.");
  status("Checking robot on CPU...");
  const result = await request("/api/ik", {
    compiled: currentPath,
    options: JSON.parse($("ikOptions").value),
  });
  status(
    `IK ${result.is_accepted ? "accepted" : "rejected"}\nFailed checks: ${result.failed_checks.join(", ") || "none"}\n${result.saved}`,
  );
});
$("showRobot").onchange = async () => {
  try {
    view.robot = $("showRobot").checked
      ? (await request(`/api/robot?index=${$("frame").value}`)).meshes
      : null;
    view.draw();
  } catch (error) {
    status(error.message);
    $("showRobot").checked = false;
  }
};
["rx", "ry", "rz"].forEach((id) => ($(id).oninput = updatePose));
$("frame").onchange = () =>
  load(Number($("frame").value)).catch((error) => status(error.message));
$("view").onchange = () => {
  view.mode = $("view").value;
  view.draw();
};
$("color").onchange = () => {
  view.color = $("color").value;
  view.draw();
};
$("nodeFrame").onchange = selectNode;
$("hand").onchange = () => {
  view.pose = null;
  sourcePose = null;
  selectNode();
};
$("plan").onchange = () => {
  try {
    updateActions();
  } catch (error) {
    status(error.message);
  }
};
action("anchor", async () => {
  const index = data.frames.indexOf(data.start_frame);
  if (index < 0) throw Error("Replay segment does not contain the plan start.");
  await load(index);
});
window.onresize = () => view.draw();
load().catch((error) => status(error.message));
