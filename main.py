import os
import sys
import time
import datetime
import pickle

import random
import numpy as np
import torch

from parameters.parameters_main import parameters_main
from functions.generate_mujoco_xml_main import generate_mujoco_xml_main
from functions.get_env_and_model import get_env_and_model
from output.save_results_main import save_results_main

from stable_baselines3.common.callbacks import BaseCallback

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Extract the info dictionary from the first environment
        info = self.locals["infos"][0]
        
        if "reward" in info:
            # Log standard singular values
            self.logger.record("reward_components/y_pos", info["reward"]["y_pos"])
            self.logger.record("reward_components/z_pos", info["reward"]["z_pos"])
            
            if isinstance(info["reward"]["progress"], list):
                self.logger.record("reward_components/progress_gait_0", info["reward"]["progress"][0])
                self.logger.record("reward_components/progress_gait_1", info["reward"]["progress"][1])
            else:
                self.logger.record("reward_components/progress", info["reward"]["progress"])
            
            if isinstance(info["reward"]["contact_time"], list):
                self.logger.record("reward_components/contact_time_gait_0", info["reward"]["contact_time"][0])
                self.logger.record("reward_components/contact_time_gait_1", info["reward"]["contact_time"][1])
            else:
                self.logger.record("reward_components/contact_time", info["reward"]["contact_time"])

            if isinstance(info["reward"]["intra_set_contact_std"], list):
                self.logger.record("reward_components/contact_std_intra_set_gait_0", info["reward"]["intra_set_contact_std"][0])
                self.logger.record("reward_components/contact_std_intra_set_gait_1", info["reward"]["intra_set_contact_std"][1])
            else:
                self.logger.record("reward_components/contact_std_intra_set", info["reward"]["intra_set_contact_std"])
            
            if isinstance(info["reward"]["inter_set_contact_std"], list):
                self.logger.record("reward_components/contact_std_inter_set_gait_0", info["reward"]["inter_set_contact_std"][0])
                self.logger.record("reward_components/contact_std_inter_set_gait_1", info["reward"]["inter_set_contact_std"][1])
            else:
                self.logger.record("reward_components/contact_std_inter_set", info["reward"]["inter_set_contact_std"])
                
        return True

def main(P=None):

    # Set random seed for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    main_dir = os.path.dirname(__file__)
    
    # Set animal model
    if(len(sys.argv) >= 2): # If an animal name is provided.
        animal = sys.argv[1]
    else:
        animal = None
    
    # Set directories and parameters
    if(len(sys.argv) >= 3): # If an existing model is provided.
        path_split = os.path.split(sys.argv[2])
        model_path = path_split[0]
        model_file = path_split[1]
        
        # Set parameters
        if(os.path.exists(f'{model_path}/parameters.pkl')): # If an existing parameters file is provided.
            with open(f'{model_path}/parameters.pkl', 'rb') as f:
                P = pickle.load(f)
                print("Loaded an existing parameters file.")
                print(P["physical_model"]["file_name"])

                from parameters.insect.training_params import training_params
                P["training"] = training_params(use_gpu=False, T=P["gaits"][0]["period"], dt=P["dt_opt"], animal=P['animal'])
        else:
            P = parameters_main()
            
            from parameters.insect.training_params import training_params
            P["training"] = training_params(use_gpu=False, T=P["gaits"][0]["period"], dt=P["dt_opt"], animal=P['animal'])
            
            print("No parameters file found. Using default parameters.")
    else: # Create a new model.
        if(P is None):
            P = parameters_main(animal)
            print("Creating a new model with default parameters.")
        else:
            print("Creating a new model using UI parameters.")
        
        gaits = ("forward" if P["gaits"][0]["direction"] == 1 else "backward") + "_" +  str(P['gaits'][0]['phase'])
        if(len(P["gaits"]) == 2):
            gaits += "-" + ("forward" if P["gaits"][1]["direction"] == 1 else "backward") + "_" + str(P['gaits'][1]['phase'])
        
        name = f"{P['animal']}_{gaits}_T={str(P['gaits'][0]['period'])}_dt={str(P['dt'])}_dt-opt={str(P['dt_opt'])}"

        print("Model name =", name)
        
        current_datetime = datetime.datetime.now()
        model_file = f"model_{name}_{current_datetime.strftime('%Y%m%d%H%M%S')}"
        model_path = f"resources/results/{P['animal']}/locomotion/{model_file}/"

    # Generate xml model
    generate_mujoco_xml_main(P=P, animal=P["animal"]) # , main_dir=main_dir, model_file_name=P["physical_model"]["file_name"])
    time.sleep(2) # A short delay to ensure the xml file has been created.
    
    env, model = get_env_and_model(P, model_path, model_file, main_dir, seed=seed)

    print("model_file", P["physical_model"]["file_name"])
    
    # Train the model
    if(P["training"]["device"] == "cuda"):
        tensorboard_callback = CustomTensorboardCallback()
        for i in range(P["training"]["batch_num"]):
            print("Training...")
            model.learn(total_timesteps=P["training"]["steps_per_batch"], callback=tensorboard_callback, tb_log_name=name, reset_num_timesteps=(i==0))
            
            model.save(model_path + os.sep + model_file + '.zip') # Save the model.
            print("Done training. Model Saved.")
    
    # Save the parameters to a file
    with open(f'{model_path}/parameters.pkl', 'wb') as f:
        pickle.dump(P, f)
    
    env.close()

    # Make plots and videos
    save_results_main(model=model, P=P, model_path=model_path, model_file=model_file)
    
if __name__ == '__main__':
	main()