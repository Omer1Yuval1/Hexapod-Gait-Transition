import numpy as np
from parameters.insect.get_reward_params import get_reward_params

def calculate_sigmoid(x, slope, offset):
    return 1 / (1 + np.exp(-slope * x + offset))

def get_gait_params(animal, gait, T, dt_opt, params, multi_gait):
    
    # dt_opt is the time step for the optimisation (= dt * frame_skip).
    # Angle_offsets only affect the control, not the joint itself (they are not used in the ref attribute of the joint).
    
    T0 = 0.25 # Baseline period [s].
    dt0 = 0.01 # Baseline optimisation time step [s].
    
    if(animal == "desert_locust" or animal == "mole_cricket"):
        gait_params = {
            # "reward_mode": 0, # 0. The phase is time-dependent. 1. Local control with a phase lag constraint. 2. Contact-based.
            # "Adaptive_global_phase": True, # This resets the phase every step to the circular mean of the current phase with the circular mean of all current leg phases. It can be thought of as phase smoothing. However it also allows the agent the stay behind (or in front) of the external clock.
            # "qpos_threshold": 32 * (np.pi / 180), # [32°=0.558[rad]]. Mean absolute difference between current and reference joint angles. Above this threshold the episode terminates. The maximum joint angle is used.
            "qpos_threshold_amplitude_ratio": 0.5, # [0.5, *1*].
            "adaptive_gait_phase_circular_mean_weights": (0.95, 0.05), # (global phase, pose-based phase). [(0.8, 0.2), (0.9, 0.1)].
            
            "joints_per_leg": 2,
            # "optimized_joints": [7,8,11,12, 15,16,19,20, 24,25,28,29], # J0+J1 for fore and middle legs, and J1+J2 for hind legs.
            # "optimized_joints": [7,8,9, 11,12,13, 15,16,17, 19,20,21, 23,24,25, 27,28,29], # J0+J1+J2 for all legs.
            "leg_base_tip_indices": np.array([[4,7], [8,11], [12,15], [16,19], [20,23], [24,27]]),
            
            "horizontal_joints": {"qpos": [7,11, 15,19, 25,29], "qvel": [6,10, 14,18, 24,28]},
            # "vertical_joints": {"qpos": [8,12, 16,20, 24,28], "qvel": [7,11, 15,19, 23,27]},
            "vertical_joints": {"qpos": [8,12, 16,20, 24,28,  23,27], "qvel": [7,11, 15,19, 23,27,  22,26]}, # Includes the hind J0 joints.
            
            "hind_joints": {"qpos": [23,27, 24,28, 25,29], "qvel": [22,26, 23,27, 24,28]}, # Hind legs J0, J1, J2.
            
            # "function_guided_joints": {"qpos": [7,11, 15,19, 25,29, 8,12, 16,20, 24,28], "qvel": [6,10, 14,18, 24,28, 7,11, 15,19, 23,27]}, # TODO: put here all joint params in a 1d array for each property.
            
            "qpos_noise_scale": 0.01, # [0.1].
            "qvel_noise_scale": 0.01, # [0.1].
            "contact_force_range": 1, # Used to clip and rescale the contact forces.

            # Gait transition parameters
            "gait_transition": {
                "location_marker": 10, # [m].
                "location_noise": 3,
                "time_marker": 2, # [s].
                "time_noise": 0.5, # [s].
                "sigmoid_function": calculate_sigmoid, # lambda x, slope, offset: 1 / (1 + np.exp(-slope * x - offset)),
                "slope": 10, # [*10*, 18]. At negative distances (before marker) f(x) ~ 1, at positive distances (after marker) f(x) ~ 0.
                "offset": 0, # [*0*, 5].
                # "slope"
                # "slope": 10 if gait["reward"] == "joint_based" else 30, # [*10*]. At negative distances (before marker) f(x) ~ 1, at positive distances (after marker) f(x) ~ 0.
                # "offset": 0 if gait["reward"] == "joint_based" else 5
                # [slope=10, offset=0]: intersecting at y=0.5, x \in [-0.25,0.25].
                # [slope=30, offset=5]: non-intersecting, x \in [-0.25,0.25].
            },
        }

        gait_params["reward_weights"], gait_params["thresholds"] = get_reward_params(animal, gait["type"], gait["reward"], params, dt_opt, dt0, T0, T, multi_gait)


    else:
        gait_params = {}
    
    if(animal == "mole_cricket" or animal == "desert_locust"):
        if(gait["phase"] == "double_tripod"):
            
            base_phases = np.array([0, np.pi, np.pi, 0, np.pi, 0]) # L1, R1, L2, R2, L3, R3.
            if(gait["direction"] == 1): # Forward.
                leg_phases = {"horizontal": base_phases, "vertical": base_phases + np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, -np.pi/2, -np.pi/2])}
            elif(gait["direction"] == -1): # Backward.
                leg_phases = {"horizontal": base_phases, "vertical": base_phases + np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2])}
            
            leg_phases["vertical"] = np.concatenate((leg_phases["vertical"], base_phases[4:6] + np.array([np.pi, np.pi]))) # Add phases for the hind J0 joints. Hind J0 phase = base_phases[[4,5]] + pi.
            
            if(gait["direction"] == 1): # Forward.
                
                leg_angles_mean = np.array([-6, 6, 72, -72, 159, -159]) * np.pi/180
                leg_angles_std = np.array([6, 6, 39, 39, 12, 12]) * np.pi/180
                
                amplitudes = {"horizontal": np.array([10, 10, 25, 25, 50, 50]) * np.pi / 180, "vertical": np.array([20, 20, 30, 30, 20, 20]) * np.pi / 180} # Amplitudes for the horizontal and vertical joints.
                # "horizontal": np.array([10, 10, 80, 80, 50, 50])
                offsets = {"horizontal": np.array([-20,-20, 30, 30, 0,0]) * np.pi / 180, "vertical": np.array([0,0, 0,0, 0,0]) * np.pi / 180} # Offsets for the horizontal and vertical joints.

                amplitudes["vertical"] = np.concatenate((amplitudes["vertical"], np.array([20, 20]) * np.pi / 180)) # Add amplitudes for the hind J0 joints.
                offsets["vertical"] = np.concatenate((offsets["vertical"], np.array([0, 0]) * np.pi / 180)) # Add amplitudes for the hind J0 joints.

            elif(gait["direction"] == -1): # Backward.
                
                leg_angles_mean = np.array([3, 3, 55, -55, 160, -160]) * np.pi/180
                leg_angles_std = np.array([8, 8, 16, 16, 18, 18]) * np.pi/180
                
                amplitudes = {"horizontal": np.array([10, 10, 25, 25, 60, 60]) * np.pi / 180, "vertical": np.array([20, 20, 30, 30, 20, 20]) * np.pi / 180} # Amplitudes for the horizontal and vertical joints.
                offsets = {"horizontal": np.array([-30,-30, -10,-10, 0,0]) * np.pi / 180, "vertical": np.array([0,0, 0,0, 0,0]) * np.pi / 180} # Offsets for the horizontal and vertical joints.
                
                amplitudes["vertical"] = np.concatenate((amplitudes["vertical"], np.array([30, 30]) * np.pi / 180)) # Add amplitudes for the hind J0 joints.
                offsets["vertical"] = np.concatenate((offsets["vertical"], np.array([10, 10]) * np.pi / 180)) # Add offsets for the hind J0 joints.

            gait_params.update({
                "leg_angles_mean": leg_angles_mean,
                "leg_angles_std": leg_angles_std,
                # "z_range": (0, 1.5), # Note that this depends on the size of the model.
                "amplitudes": amplitudes,
                "offsets": offsets, # L1, R1, L2, R2, L3, R3.
                "leg_phases": leg_phases,
            })
        if(gait["phase"] == "bound"):

            base_phases = np.array([0, 0, np.pi, np.pi, np.pi, np.pi]) # Each left-right leg pair is in-phase and the middle legs are anti-phase with the hind legs.
            # base_phases = np.array([0, 0, np.pi, np.pi, 0, 0]) # Each left-right leg pair is in-phase and the middle legs are anti-phase with the hind legs.
            if(gait["direction"] == 1): # Forward.
                leg_phases = {"horizontal": base_phases, "vertical": base_phases + np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, -np.pi/2, -np.pi/2])}
            elif(gait["direction"] == -1): # Backward.
                leg_phases = {"horizontal": base_phases, "vertical": base_phases + np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2])}
            
            leg_phases["vertical"] = np.concatenate((leg_phases["vertical"], base_phases[4:6] + np.array([np.pi, np.pi]))) # Add phases for the hind J0 joints. Hind J0 phase = base_phases[[4,5]] + pi.
            
            leg_angles_mean = np.array([-5, 5, 50, -50, 155, -155]) * np.pi/180
            leg_angles_std = np.array([7, 7, 18, 18, 25, 25]) * np.pi/180

            amplitudes = {"horizontal": np.array([10, 10, 25, 25, 60, 60]) * np.pi / 180, "vertical": np.array([20, 20, 30, 30, 20, 20]) * np.pi / 180} # Amplitudes for the horizontal and vertical joints.
            offsets = {"horizontal": np.array([-30,-30, -10,-10, 0,0]) * np.pi / 180, "vertical": np.array([0,0, 0,0, 0,0]) * np.pi / 180} # Offsets for the horizontal and vertical joints.
            
            amplitudes["vertical"] = np.concatenate((amplitudes["vertical"], np.array([30, 30]) * np.pi / 180)) # Add amplitudes for the hind J0 joints.
            offsets["vertical"] = np.concatenate((offsets["vertical"], np.array([10, 10]) * np.pi / 180)) # Add offsets for the hind J0 joints.

            gait_params.update({
                "leg_angles_mean": leg_angles_mean,
                "leg_angles_std": leg_angles_std,
                # "z_range": (0, 1.5), # Note that this depends on the size of the model.
                "amplitudes": amplitudes,
                "offsets": offsets, # L1, R1, L2, R2, L3, R3.
                "leg_phases": leg_phases,
            })

    if(animal == "mole_cricket"):
        gait_params["terminal_geoms"] = np.array(["L1_tibia_geom", "R1_tibia_geom", "L2_tarsus_geom", "R2_tarsus_geom", "L3_tarsus_geom", "R3_tarsus_geom"])
    elif(animal == "desert_locust" or animal == "carpenter_ant"):
        gait_params["terminal_geoms"] = np.array(["L1_tarsus_geom", "R1_tarsus_geom", "L2_tarsus_geom", "R2_tarsus_geom", "L3_tarsus_geom", "R3_tarsus_geom"])
    elif(animal == "half_cheetah"):
        gait_params["terminal_geoms"] = np.array(["ffoot", "bfoot"])
    elif(animal == "biped"):
        gait_params["terminal_geoms"] = np.array(["Geom_L1_femur", "Geom_L3_femur"])
    elif(animal == "triped"):
        gait_params["terminal_geoms"] = np.array(["Geom_L1_femur", "Geom_R1_femur", "Geom_L3_femur"])
    
    # Set leg sets as grouped indices of terminal geoms
    if(len(gait_params["terminal_geoms"]) == 6):
        if(gait["phase"] == "double_tripod"):
            gait_params["leg_sets"] = [[0, 3, 4], [1, 2, 5]]
        elif(gait["phase"] == "bound"):
            # gait_params["leg_sets"] = [[0,1], [2,3], [4,5]] # Fore, middle, hind.
            gait_params["leg_sets"] = [[0,1,4,5], [2,3]] # Fore+hind, middle.
        else:
            gait_params["leg_sets"] = [[0, 1, 2, 3, 4, 5]]
    elif(len(gait_params["terminal_geoms"]) == 2):
        gait_params["leg_sets"] = [[0], [1]] # The interleg phase is in P["gaits"][0]["phase"].
    
    return gait_params