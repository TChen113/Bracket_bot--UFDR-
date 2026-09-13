import glob
import os
import json
import time
import numpy as np
from scipy.optimize import minimize
import rerun as rr
import visualize_urdf as vu

# 1. Initialize Rerun and Log the Whiteboard Once
rr.init("bracketbot_drawing", spawn=True)
rr.log("world/whiteboard", rr.Boxes3D(centers=[[0.49, 0.0, 1.10]], half_sizes=[[0.01, 0.915, 0.61]], colors=[[245, 245, 250, 180]]), static=True)
rr.log("world/whiteboard/canvas_target", rr.Boxes3D(centers=[[0.481, -0.15, 1.05]], half_sizes=[[0.001, 0.25, 0.25]], colors=[[50, 150, 255, 120]]), static=True)

# 2. Setup the IK Kinematic Tree
urdf_path = r"D:\Battle_of_the_School\Bracket_bot (UFDR)\chopped_urdf_v2\chopped_urdf_v2\urdf\chopped_urdf_v2.urdf"
links, joints, roots = vu.parse_urdf(urdf_path)
order = vu.build_paths(links, joints, roots)
joint_names = ["rj0", "rj1", "rj2", "rj3", "rj4", "rj5", "rj6"]

def get_right_eef_position(q_array):
    q_values = {name: val for name, val in zip(joint_names, q_array)}
    transforms = {"": np.eye(4)}
    for link, path, joint in order:
        parent_path = "" if "/" not in path else path.rsplit("/", 1)[0]
        T_parent = transforms.get(parent_path, np.eye(4))
        if joint is None:
            transforms[path] = T_parent
            continue
        t, r = vu.joint_transform(joint, q_values.get(joint["name"], 0.0))
        T_local = np.eye(4)
        T_local[:3, :3] = r
        T_local[:3, 3] = t
        T_global = T_parent @ T_local
        transforms[path] = T_global
        if link == "right_eef":
            return T_global[:3, 3]
    return np.zeros(3)

global_target = np.zeros(3)
def cost_function(q_array):
    return np.linalg.norm(global_target - get_right_eef_position(q_array))

# 3. The IK Solver with REAL-TIME Logging
bnds = [(-1.03, 0.0)] + [(-2.09, 2.09)] * 6
current_angles = np.array([-0.3, 0.392, -0.071, 0.868, 0.007, -0.286, 0.705])

def run_ik_solver(target_pos_array):
    global global_target, current_angles
    global_target = target_pos_array
    result = minimize(cost_function, current_angles, method="L-BFGS-B", bounds=bnds)
    
    if result.success or result.fun < 0.005:
        current_angles = result.x
        # Log the robot's pose on the default real-time clock
        vu.log_robot(urdf_path, {name: val for name, val in zip(joint_names, result.x)})
        # Slow down Python slightly so we can watch the drawing happen live!
        time.sleep(0.002)  

# 4. Automatically find and load the JSON file
json_files = glob.glob("**/*_strokes.json", recursive=True) + glob.glob("*_strokes.json")
if not json_files:
    raise FileNotFoundError("No *_strokes.json file found! Run the vectorizer script first.")
latest_json = max(json_files, key=os.path.getmtime)
print(f"Automatically loading stroke file: {latest_json}")

with open(latest_json, "r") as f:
    strokes = json.load(f)["strokes"]

# 5. Execute the Drawing Loop
PEN_UP_X = 0.3
PEN_DOWN_X = 0.45

def map_coords(u, v):
    phys_y = 0.10 - (u * 0.50)
    phys_z = 0.80 + (v * 0.50)
    return phys_y, phys_z

print(f"Calculating {len(strokes)} strokes and animating live...")

drawn_segments = []

for stroke in strokes:
    start_y, start_z = map_coords(stroke[0][0], stroke[0][1])
    
    # Hover
    run_ik_solver(np.array([PEN_UP_X, start_y, start_z]))
    current_stroke_pts = []  
    
    # Plunge
    run_ik_solver(np.array([PEN_DOWN_X, start_y, start_z]))
    current_stroke_pts.append([PEN_DOWN_X, start_y, start_z])
    
    # Draw path
    for point in stroke[1:]:
        y, z = map_coords(point[0], point[1])
        run_ik_solver(np.array([PEN_DOWN_X, y, z]))
        current_stroke_pts.append([PEN_DOWN_X, y, z])
        
        # Live trace update
        if len(current_stroke_pts) > 1:
            current_all_segments = drawn_segments + [current_stroke_pts]
            rr.log("world/drawing_trace", rr.LineStrips3D(current_all_segments, colors=[[0, 0, 0, 255]], radii=[0.003]))

    # Save stroke
    if len(current_stroke_pts) > 1:
        drawn_segments.append(current_stroke_pts)

    # Retract
    end_y, end_z = map_coords(stroke[-1][0], stroke[-1][1])
    run_ik_solver(np.array([PEN_UP_X, end_y, end_z]))

# 6. Lock the final drawing to the canvas permanently
rr.log("world/drawing_trace", rr.LineStrips3D(drawn_segments, colors=[[0, 0, 0, 255]], radii=[0.003]), static=True)

print("Done! The drawing is permanently locked to the canvas.")