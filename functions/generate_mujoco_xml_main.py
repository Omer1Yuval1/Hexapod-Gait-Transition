import os
# import gymnasium as gym

def generate_mujoco_xml_main(P=None, animal=None, main_dir="", model_file_name=""):
    
    if(animal is None):
        animal = "mole_cricket"
    
    # source_file = os.path.join(main_dir, 'envs', 'mujoco', 'assets', animal + "_template.xml")
    target_file = os.path.join(main_dir, 'envs', 'mujoco', 'assets', animal + ".xml")
    
    if(animal == "mole_cricket" or animal == "desert_locust"):
        from functions.generate_mujoco_xml_insect import generate_mujoco_xml_insect
        generate_mujoco_xml_insect(P=P)
    elif(animal == "worm"):
        print("Currently no xml file generation required for the worm environment.")
        # from envs.mujoco.WormEnvSimple import WormEnvSimple # Import RL environment.
        # gym.register(P["env"], entry_point=WormEnvSimple)
        # from functions.generate_mujoco_xml_worm import generate_mujoco_xml_worm
        # from envs.mujoco.WormEnvMuJoCo import WormEnvMuJoCo # Import RL environment.
        # gym.register(P["env"], entry_point=WormEnvMuJoCo)
        # generate_mujoco_xml_worm(P=P, target_file=target_file)
    elif(animal == "half_cheetah"):
        # print(P["animal"])
        # from envs.mujoco.half_cheetah_v5 import HalfCheetahEnv # Import RL environment and register.
        from functions.generate_mujoco_xml_half_cheetah import generate_mujoco_xml_half_cheetah
        # gym.register(P["env"], entry_point=HalfCheetahEnv)
        generate_mujoco_xml_half_cheetah(P=P, target_file=target_file)