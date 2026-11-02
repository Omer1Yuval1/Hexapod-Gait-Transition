def save_results_main(model=0, P=None, model_path=None, model_file=None, ep=0):
    
    if(P is None):
        animal = "mole_cricket"
    else:
        animal = P["animal"]
    
    if(animal == "mole_cricket" or animal == "desert_locust" or animal == "carpenter_ant"):
        from output.insect.save_agent_video import save_agent_video
        save_agent_video(model=model, animal=animal, P=P, model_path=model_path, model_file=model_file, ep=ep)
    elif(animal == "worm"):
        from output.worm.save_worm_video import save_worm_video
        save_worm_video(model=model, P=P, model_path=model_path, model_file=model_file, ep=ep)
    elif(animal == "half_cheetah"):
        from output.half_cheetah.save_agent_video import save_agent_video
        save_agent_video(model=model, P=P, model_path=model_path, model_file=model_file, ep=ep)
    elif(animal == "mole_cricket_dig"):
        from output.insect.save_agent_video_dig import save_agent_video
        save_agent_video(model=model, animal=animal, P=P, model_path=model_path, model_file=model_file, ep=ep)