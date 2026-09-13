import numpy as np

def convert_radians_to_motor_turns(urdf_pos_array):
    """
    Converts 7-DOF URDF joint angles (radians/meters) into physical motor turns.
    Extracted from BracketBot constants.py.
    """
    # Mechanical constants from the hardware team
    ik_sign = np.array([-1, 1, -1, 1, 1, -1, -1], dtype=np.float32)
    wheel_radius = 0.0465
    
    q = np.array(urdf_pos_array, dtype=np.float32).copy()
    
    # Apply motor mounting direction flips
    q[:7] = q[:7] * ik_sign
    
    # Convert linear Z-axis movement (meters) on the lift mast to wheel rotations
    q[0] = q[0] / wheel_radius
    
    # Convert radians to full rotations (turns)
    q = q / (2 * np.pi)
    
    return q