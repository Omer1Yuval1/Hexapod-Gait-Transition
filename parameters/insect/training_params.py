import torch

def training_params(use_gpu=False, T=0.5, dt=0.01, animal=None, multi_gait=False):
    
    # dt is the optimisation time step (i.e., the physics time step x frame_skip).
    
    P = {
        "policy_network": [128, 128], # [128, 128] default in PPO. [256, 256].
        "value_network": [128, 128], # [128, 128] default in PPO. [256, 256].
        "activation_function": torch.nn.Tanh, # `torch.nn.ReLU` (default in SAC). `torch.nn.Tanh` (default in PPO).
        "algorithm": "PPO", # PPO, SAC, TD3, DDPG.
        "gamma": 0.99, # Default=0.99. [*0.99*]. Higher values mean the agent takes future rewards more seriously.
        "ent_coef": 0, # Default=0.0. [*0*, 0.001] Higher values mean more exploration.
        
        "verbose": 1,
        "tensorboard_log_path": f"resources/results/{animal}/tensorboard_log/",
    }
    
    if(use_gpu and torch.cuda.is_available()):
        P["device"] = "cuda"
        P["n_envs"] = 16 # 16
        
        if(multi_gait):
            P["total_timesteps"] = 20*(10**6) # [20].
        else:
            P["total_timesteps"] = 10*(10**6)
        
        P["steps_per_batch"] = P["total_timesteps"]
        P["batch_num"] = int(P["total_timesteps"] / P["steps_per_batch"])
        
        print("Using GPU")
    else:
        P["device"] = "cpu"
        P["n_envs"] = 2
        
        P["total_timesteps"] = 3*(10**6)
        P["steps_per_batch"] = 1*(10**6)
        P["batch_num"] = 1
        
        print("Using CPU")
    
    P["steps_per_period"] = round(T / dt)

    periods_per_episode = 8 if T == 0.25 else 4 # Number of periods in one episode.
    
    if(multi_gait):
        periods_per_episode *= 2 # Double the episode length for multi-gait training.
        
    P["steps_per_episode"] = periods_per_episode * P["steps_per_period"] + 1
    
    return P