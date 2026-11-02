def parameters(animal=None):

    gaits = [
        # {"type": "walk", "phase": "double_tripod", "direction": 1, "period": 0.25, "reward": "joint_based", "params": {}},
        {"type": "walk", "phase": "double_tripod", "direction": -1, "period": 0.25, "reward": "joint_based", "params": {}},
        {"type": "walk", "phase": "bound", "direction": -1, "period": 0.25, "reward": "joint_based", "params": {}},
    ]

    multi_gait = True if len(gaits) > 1 else False
    
    P = {
        "dt": 0.001, # Physics time step [s].
        "dt_opt": 0.005, # Optimisation time step [s].
        
        "gaits": gaits,
        
        "time_rounding": 3, # [*3*]. Number of decimal digits.
        "animal": animal,
        "animal_model": animal,
        
        "env": "insect_env",
        "solver": "RK4",
        
        # Flags
        "multi_gait": multi_gait,
        "object_mode": False, # True if gait["type"] == "walk" and not gait["reward"] == "contact_based" else False,
        
        "object_type": "tunnel_simple", # [None, tunnel, tunnel_simple, wall_simple].
        
        "rendering": {
            "camera_names": ["worldview_0", "worldview_1", "robot_view_0", "robot_view_1"],
            "render_fps": 100, # Number of frames to render per simulation second.
        },
        
        "physical_model": {
            "file_name": "insect",
        },
    }

    P["frame_skip"] = int(P["dt_opt"] / P["dt"]) # Number of physics steps to skip per optimisation/control step.
    
    return P