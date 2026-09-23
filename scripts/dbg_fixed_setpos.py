# probe: does set_pos/set_quat(relative=True) move a FIXED rigid entity?
import os, sys
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))
import numpy as np
import genesis as gs

gs.init(backend=gs.gpu, precision="32", seed=0, logging_level="warning")
scene = gs.Scene(show_viewer=False)
e = scene.add_entity(
    morph=gs.morphs.Box(pos=(0.0, 0.0, 1.0), size=(0.1, 0.1, 0.1), fixed=True),
    material=gs.materials.Rigid(gravity_compensation=1.0),
)
scene.build()
p0 = e.get_pos().cpu().numpy().copy()
e.set_pos(np.array([0.5, 0.0, 1.0]), relative=True, zero_velocity=True)
e.set_quat(np.array([0.9239, 0.0, 0.3827, 0.0]), relative=True, zero_velocity=True)
scene.step()
p1 = e.get_pos().cpu().numpy()
q1 = e.get_quat().cpu().numpy()
print("pos before:", p0, " after:", p1, " quat after:", q1)
print("MOVED" if abs(p1[0] - 0.5) < 1e-4 else "NOT MOVED")
