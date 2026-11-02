import os
import sys

import gymnasium as gym
from stable_baselines3 import PPO # [PPO, SAC] Import RL agent.
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from envs.mujoco.insectEnv import InsectEnv

def get_env_and_model(P, model_path, model_file, main_dir, seed=0):

    env = SubprocVecEnv([lambda: make_env(P, main_dir) for _ in range(P["training"]["n_envs"])]) # Set RL environment(s).
    
    env = VecMonitor(env, P["training"]["tensorboard_log_path"])
    print("Action space:", env.action_space)
    print("Obs space:", env.observation_space)
    
    # Set model
    if(len(sys.argv) >= 3): # If an existing model is provided.
        model = PPO.load(model_path + os.sep + model_file, env=env) # Load the saved model.
        print("Loading an existing PPO model.")
    else: # Create new model.
        policy_kwargs = dict(activation_fn=P["training"]["activation_function"], net_arch=dict(pi=P["training"]["policy_network"], vf=P["training"]["value_network"]))
        
        model = PPO("MlpPolicy", env, seed=seed, gamma=P["training"]["gamma"], ent_coef=P["training"]["ent_coef"], device=P["training"]["device"], policy_kwargs=policy_kwargs, verbose=P["training"]["verbose"], tensorboard_log=P["training"]["tensorboard_log_path"])
        
        print("Creating a new model and training a PPO agent.")
    # print("Policy:", model.policy)
    
    return env, model

def make_env(P, main_dir):
    xml_path = os.path.join(main_dir, 'envs', 'mujoco', 'assets', P["physical_model"]["file_name"] + ".xml")
    
    match P["animal"]:
        case "mole_cricket" | "desert_locust" | "carpenter_ant":
            return gym.make(P["env"], P=P, xml_file=xml_path, max_episode_steps=P["training"]["steps_per_episode"])
        case "mole_cricket_dig":
            from envs.mujoco.digEnv import digEnv
            return gym.make('dig_env', P=P, xml_file=xml_path, max_episode_steps=P["training"]["steps_per_episode"])
        case "half_cheetah":
            # from envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
            from gymnasium.envs.registration import register
            register(id='half_cheetah_env', entry_point='envs.mujoco.half_cheetah_v5:HalfCheetahEnv')
            return gym.make(P["env"], P=P, xml_file=xml_path, max_episode_steps=P["training"]["steps_per_episode"])
        case "worm":
            from envs.mujoco.WormEnvSimple import WormEnvSimple # Import RL environment.
            return gym.make(P["env"], P=P, max_episode_steps=P["training"]["steps_per_episode"])
        case _:
            return "Unknown input. Cannot create environment."