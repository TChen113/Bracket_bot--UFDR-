https://www.youtube.com/watch?v=x1wyK6OXQsU

This repository contains the custom image processing and inverse kinematics pipeline developed for "Stroke Bot" at the Battle of the School hackathon. 
The system takes a line image, extracts continuous centerlines, solves the inverse kinematics for a 7-DOF robotic arm, and simulates the drawing process in real-time.
Note: The 3D URDF and mesh files for the robotic arm were provided by the hackathon sponsors and are not included in this repository to save space

System Architecture: The pipeline consists of three main Python scripts:
1. image_to_svg_vectorize_en.py (Vision & Path Generation):
  - Loads a black-and-white image and uses OpenCV skeletonization (cv2.ximgproc.thinning) to extract single-pass, 1-pixel-wide centerline strokes to prevent the robot from retracing paths.
  - Normalizes the stroke coordinates to a [0, 1] unit square.
  - Exports the normalized path data to a *_strokes.json file.

2. visualize_urdf.py (Kinematic Tree & Rerun Integration):
  - Parses the sponsor-provided URDF file and builds the kinematic tree.
  - Logs the 3D meshes, joint transforms, and the target whiteboard canvas directly to the Rerun viewer.

3. solve_ik.py (Inverse Kinematics & Execution):
  - Automatically loads the most recent *_strokes.json file.
  - Maps the normalized [0, 1] coordinates to the physical Y and Z dimensions of the simulated whiteboard.
  - Uses SciPy's minimize function (L-BFGS-B method) to solve the inverse kinematics for the 7-DOF arm.
  - Executes the drawing loop in Rerun, moving the end-effector between hovering (X = 0.3) and drawing (X = 0.45) positions while logging a live line trace of the drawing.

Library Requirements:
  - numpy
  - scipy
  - opencv-contrib-python
  - rerun-sdk

How to Run the Simulation
Step 1: Generate the Stroke Data
Run the image vectorizer on your target image. This will output a JSON file containing the stroke arrays in the same directory.

```:
python image_to_svg_vectorize_en.py <path_to_image>
```

Step 2: Run the IK Solver and Simulation
Ensure you have the Rerun viewer installed. Run the solver script, which will automatically find the generated JSON file, launch the Rerun viewer, and begin the live drawing animation.


```:
python solve_ik.py
```
