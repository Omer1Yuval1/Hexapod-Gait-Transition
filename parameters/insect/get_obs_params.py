def get_obs_params(gait, multi_gait=False):
    
    if(gait["type"] == "walk"):
        if(gait["reward"] == "contact_based"):
            P = {
                "exclude_xy_from_obs": True,
                "use_contact_time": True,

                "use_contact_forces": False, # True if multi_gait else False,
                "use_gait_phase": False, # Use the gait phase in the observation space.
                "use_joint_phases": False, # Use the joint phases in the observation space.
                "use_body_state_randomizer": False,
                "use_leg_angles_std": False,
                "use_collision": False,
            }
        elif(gait["reward"] == "joint_based"):
            P = {
                "use_gait_phase": True, # Use the gait phase in the observation space.
                "use_joint_phases": False, # Use the joint phases in the observation space.
                "use_body_state_randomizer": True,
                "use_contact_time": False,
                "use_contact_forces": True, # True if not multi_gait else False,
                "exclude_xy_from_obs": True,
                "use_leg_angles_std": False,
                "use_collision": False,
            }
    elif(gait["type"] == "jump"):
        P = {
            "exclude_xy_from_obs": True,
            "use_gait_phase": True, # Use the gait phase in the observation space.
            "use_contact_forces": True,
            
            "use_joint_phases": False, # Use the joint phases in the observation space.
            "use_body_state_randomizer": False,
            "use_contact_time": False,
            "use_leg_angles_std": False,
            "use_collision": False,
        }
    elif(gait["type"] == "run"):
        P = {

        }
    elif(gait["type"] == "dig"):
        P = {
            "exclude_xy_from_obs": False,
            "use_contact_forces": False,
            
            "use_contact_time": False,

            "use_gait_phase": False, # Use the gait phase in the observation space.
            "use_joint_phases": False, # Use the joint phases in the observation space.
            "use_body_state_randomizer": False,
            "use_leg_angles_std": False,
            "use_collision": False,
        }

    # Body structure
    leg_num = 6 # Number of legs.
    torso_param_num = 3 + 4 + 3 + 3 # 3=(x,y,z-position), 4=(xyzw orientation). 3=(xyz velocity), 3=(xyz angular velocity).
    hinge_joints_per_leg = 4
    
    hinge_param_num = 2 * leg_num * hinge_joints_per_leg # Angle and angular velocity of each joint. 48.
    
    obs_shape = torso_param_num + hinge_param_num
    # obs_shape = torso_param_num + hinge_param_num + terminal_contacts + leg_tip_contact_time
    
    if P["exclude_xy_from_obs"]:
        obs_shape -= 2
    if P["use_gait_phase"]:
        if(multi_gait):
            obs_shape += 4 # sin+cos for each gait.
        else:
            obs_shape += 2 # Gait phase: sin and cos of the gait phase.
    if P["use_joint_phases"]:
        obs_shape += 24 # Joint phases: sin and cos of the joint phases. TODO: replace with the number of optimized joints defined in gait_params.
    if P["use_collision"]:
        obs_shape += 0
    if P["use_contact_forces"]:
        obs_shape += 29*6 # nbody * 6. External forces: force in {x, y, z} and torque around {x, y, z} applied to the CoM of each body.
    if(P["use_contact_time"]):
        obs_shape += 6
    """ if(P["use_angular_velocity_sign"]):
        obs_shape += 6 """
    if(P["use_leg_angles_std"]):
        obs_shape += 6
    
    if(multi_gait):
        obs_shape += 2 # Transition flag (boolean) and distance to the next transition point.
    
    P["size"] = obs_shape

    return P