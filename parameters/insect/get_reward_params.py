import numpy as np

def get_reward_params(animal, gait_type, reward_type, params, dt_opt, dt0, T0, T, multi_gait):
    
    if(gait_type == "walk"):
        if(reward_type == "contact_based"):
            
            reward_weights = {
                "step_reward": 1 * (dt_opt / dt0), #  * (T0 / T) * (dt_opt / dt0),
                
                # "progress": 5 / params["total_length"], # TODO: this might also require scaling by the gait period.
                "progress": 1, # [*1*, 10]. Scaled automatically by the scale of the forward/backward progress.
                
                "contact_time": -20 * (T0 / T), # [*-20*, -2]. Minimize the std of contact time within each leg set, and maximize the std across leg sets.
                "contact_time_intra_inter": [1, 5], # *[1, 2]*, [1, 0.5].
                
                "orientation": -2, # [*-2*]. Keeps the yaw, pitch and roll angles close to zero.
                "action_smoothness": 0, # [*-1*]. Minimize the different between consecutive actions. Action diff is implicitly normalized actions are smaller for smaller time steps.
                "left_right_symmetry": 0, # [*-2*, -5, -10]. Minimize the difference between matching left and right joints.
                
                "qpos": 0, # [*-1*]. Scaled automatically by the scale of the change in joint angle (smaller period leads to larger angle diffs, while a smaller dt_opt leads to smaller angle diffs).
                "angular_velocity": 0 * (dt_opt / dt0), # [-0.001, -0.005, *-0.0001*]. Normalized by the control time step since angular velocity is in rad/s (hence more time steps mean more penalty).
                
                "leg_angles_stats": 0, # [*-50*]. This is used to constrain the leg angles to specific mean and std values.
                "leg_angles_range": 0, # [*-100*].
                "interjoint_phase": 0,
                "contact_force_smoothness": 0, # [-10]. Minimize the difference between consecutive contact forces.
            }
            
            thresholds = {
                "max_roll_angle": 60 * (np.pi/180), # Rotation around the x-axis (lateral).
                "max_pitch_angle": 60 * (np.pi/180), # Rotation around the y-axis (longitudinal, head-to-abdomen).
                "max_yaw_angle": 20 * (np.pi/180), # Rotation around the z-axis.
                "max_z_pos": 1.1 * params["body_pos"][2],
                # "max_z_pos": 0.5 if animal == "desert_locust" else 0.8 if animal == "mole_cricket" else np.inf, # [*1.1*, 0.8]. Maximum z position of the body.
            }
        elif(reward_type == "joint_based"):
            reward_weights = {
                "step_reward": 1 * (dt_opt / dt0), #  * (T0 / T) * (dt_opt / dt0),
                "progress": 1, # [*1*, 10]. Scaled automatically by the scale of the forward/backward progress.
                
                "qpos": -20, # [*-1*]. Scaled automatically by the scale of the change in joint angle (smaller period leads to larger angle diffs, while a smaller dt_opt leads to smaller angle diffs).
                "qpos_subweights": {"phase": 0.25, "qpos": 5, "qvel": 0},
                "angular_velocity": -0.1, #  * (dt_opt / dt0), # [*-0.05*]. Normalized by the control time step since angular velocity is in rad/s (hence more time steps mean more penalty).
                
                "orientation": -1, # [*-1*]. Keeps the yaw, pitch and roll angles close to zero.
                "action_smoothness": 0, # [*-1*]. Minimize the different between consecutive actions. Action diff is implicitly normalized actions are smaller for smaller time steps.                
                "left_right_symmetry": 0, # Minimize the difference between matching left and right joints.

                "contact_time": 0,
                "leg_angles_stats": 0, # [*-50*]. This is used to constrain the leg angles to specific mean and std values.
                "interjoint_phase": 0,
                "contact_force_smoothness": 0, # [-10]. Minimize the difference between consecutive contact forces.
                "jump": 0,
                "pre_jump_contact": 0,
            }

            thresholds = {
                "max_roll_angle": 60 * (np.pi/180), # Rotation around the x-axis (lateral).
                "max_pitch_angle": 60 * (np.pi/180), # Rotation around the y-axis (longitudinal, head-to-abdomen).
                "max_yaw_angle": 40 * (np.pi/180), # [*20*, 40]. Rotation around the z-axis.
                "max_z_pos": 1.1 * params["body_pos"][2],
            }
    elif(gait_type == "jump"):
        reward_weights = {
            "step_reward": 0,
            "jump": 0.001,
            "contact_time": 0, # Minimize the std of contact time within each leg set, and maximize the std across leg sets.
            "contact_time_intra_inter": [1, 2],
        }

        thresholds = {
            "max_roll_angle": 20 * (np.pi/180), # Rotation around the x-axis (lateral).
            "max_pitch_angle": 30 * (np.pi/180), # Rotation around the y-axis (longitudinal, head-to-abdomen).

            # TODO: the initial yaw is currently opposite (+180) for the locust model.
            "max_yaw_angle": 20 * (np.pi/180), # Rotation around the z-axis.
            # "z_range": 0.65,
        }
    elif(gait_type == "run"):
        if(reward_type == "contact_based"):
            reward_weights = {
                # "contact_weight": 4,
                "air_weight": 3, # [*3*].
                "forward_weight": 1, # [4, *1*].
                "control_cost_weight": 0.1,
                "smoothness_weight": 1.7,
                "step_reward": 0.3,
                "time_difference": 35, # [*35*].
                
                "collision_penalty": 0.01,
                # "termination_penalty": 30 / 10,
                "min_y_angle": -0.4, # [rad].
                "max_y_angle": 0.35, # [rad].
                # "max_z_angle": 0.45, # change to 0.3 
                "min_z_pos": -0.2,
                "max_time_difference_period_ratio": 1,
                "contact_force_range":(-1, 1),

                "reset_noise_scale": 0.01, # 0.1 yields poorer results.
            }

            thresholds = {}

        else:
            reward_weights = {
                # "contact_threshold": 0.1,
                "min_y_angle": -np.pi/5, # [rad]. [-np.pi/5].
                "max_y_angle": np.pi/5, # [rad]. [np.pi/5].
                "step_reward": 0, # 0.1
                "forward_weight": 1,
                "control_cost_weight": 0.1,
                "smoothness_weight": 1,
                "contact_force_range": (-1, 1),
                "collision_penalty": 0.01,
                "reset_noise_scale": 0.1,
            }

            thresholds = {}

    thresholds["target_yaw_angle"] = np.pi if animal == "desert_locust" else 0 # TODO: the initial yaw is currently opposite (+180) for the locust model.

    return reward_weights, thresholds