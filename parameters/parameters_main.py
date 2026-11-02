def parameters_main(animal=None, T=None):
    
    if(animal is None):
        animal = "mole_cricket" # ["mole_cricket", "worm", "half-cheetah"].
    
    if(animal == "mole_cricket" or animal == "desert_locust" or animal == "biped" or animal == "triped" or animal == "carpenter_ant"):
        from parameters.insect.parameters import parameters
        from parameters.insect.training_params import training_params
        from parameters.insect.get_gait_params import get_gait_params
        from parameters.insect.get_model_params import get_model_params
        from parameters.insect.get_obs_params import get_obs_params
        from parameters.insect.object_params import object_params

        train = True
        
        P = parameters(animal=animal)
        
        P["model"] = {}
        P["model"]["params"], P["model"]["joints"], P["model"]["body_parts"] = get_model_params(P["animal"], P["gaits"][0])
        
        for i, gait in enumerate(P["gaits"]):
            P["gaits"][i]["params"] = get_gait_params(animal, gait, T=gait["period"], dt_opt=(P["dt"]*P["frame_skip"]), params=P["model"]["params"], multi_gait=P["multi_gait"])
        
        P["observation"] = get_obs_params(P["gaits"][0], P["multi_gait"])
        P["training"] = training_params(use_gpu=train, T=P["gaits"][0]["period"], dt=P["dt_opt"], animal=P['animal'], multi_gait=P["multi_gait"])

        if(P["object_mode"]):
            P["objects"] = object_params(P["object_type"])
        else:
            P["objects"] = []
    
    elif(animal == "worm"):
        from parameters.worm.parameters import parameters
        P = parameters()
    elif(animal == "half_cheetah"):
        from parameters.half_cheetah.parameters import parameters
        from parameters.half_cheetah.get_obs_params import get_obs_params
        from parameters.half_cheetah.get_gait_params import get_gait_params
        from parameters.half_cheetah.training_params import training_params
        P = parameters()
        P["observation"] = get_obs_params(P["object_mode"], P["impose_leg_phases"], P["gait_transition"])
        P["gait_params"] = get_gait_params(P["dt_opt"], P["T"], P["object_mode"], P["impose_leg_phases"])
        
        P["training"] = training_params(use_gpu=True, T=P["T"], dt=P["dt_opt"], animal=P['animal'], gait=P["gait"], gait_transition=P["gait_transition"], object_mode=P["object_mode"])
        P["evaluation"] = {
            "steps_per_episode": int(16 / P["dt_opt"]) + 1,
            "episode_num": 30,
        }
    elif(animal == "mole_cricket_dig"):
        print("*** mole_cricket_dig ***")
        from parameters.insect.parameters import parameters
        from parameters.insect.get_obs_params import get_obs_params
        from parameters.insect.training_params import training_params
        # from parameters.insect.get_gait_params import get_gait_params

        P = parameters(animal=animal)

        P["gaits"] = [{"type": "dig", "phase": "", "direction": 1, "period": 0.25, "reward": "", "params": {}}]
        P["env"] = "dig_env"
        P["physical_model"]["file_name"] = "mole_cricket_jelly_control"

        P["observation"] = get_obs_params(P["gaits"][0], False)
        P["training"] = training_params(use_gpu=True, T=0.25, dt=P["dt_opt"], animal=animal, multi_gait=False)
        
        P["training"]["steps_per_episode"] = 1600 # 400.
        # P["total_timesteps"] = 5*(10**6)
    return P