def parameters_vibration(animal=None):

    P = {
        "tilt_angle": 0, # [deg]. [5].
        "face_angle": 0, # [deg]. [120].
        "steps_per_episode": 15000, # ant=[15000]. biped=[20000]. desert_locust=[25000].
        "frequencies": [80], # Hz. [62.5Hz = 0.016. ***80Hz = 0.0125***. 90Hz = 0.111, 20Hz = 0.05].
        "amplitude": 0.5, # 10Hz=1.5, 20Hz=[3,*6*], 40Hz=6, 80Hz=1.
        "flags": {
            "add_sticks": False
        },
        "actuator": {
            "ctrlrange": "-.02 .02",
            "kp": 500000, # [80Hz = 500000, 60Hz = 300000, 20Hz=]
        },
        "rendering": {
            "camera_pos": "-30 0 5", # locust="-60 0 15"
            "camera_name": "world_view_0",
            "camera_euler": "0 -80 -90",
        },
        "floor": {
            "geom": {
            },
            "joint": {
                "springdamper": "0 0",
                "armature": 0.001,
                "damping": 0.001,
                "frictionloss": 0,
                "range": "-.02 .02",
            }
        }
    }
    print(animal)
    if(animal == "triped"):
        P["amplitude"] = 0.1
        P["actuator"]["kp"] = 450000
    
    return P