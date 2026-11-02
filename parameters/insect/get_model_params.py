import numpy as np

def get_model_params(animal_model, gait, tilt_angle=0, face_angle=0):
    
    '''
        All units are in CGS.
        "com" stands for center of mass.
    '''

    SI = False
    # CGS = False
    
    default_mass = 0.001
    gravity = "0 0 -980" # "0 0 -9.8"
    noslip_iterations = 3
    
    if(animal_model == "mole_cricket"):
        
        if(SI):
            r = 0.2 * (1.35 / 0.75)
            gravity = "0 0 -9.81"
            
            stiffness = 50 # [100].
            damping = 4 # [*20*].
            armature = 0.1
            frictionloss = 0

            geom_margin = "0.002"
            solref = "0.02 1"
            solimp = "0.0 0.99 0.01"
            friction = [2, 0.1, 0.1] # [2, 0.1, 0.1].

            gear_ratio_high = 25 # Used for horizontal joints.
            gear_ratio_low = 25 # Vertical joints.
            gear_ratio = 25 # Other joints.

            total_mass = 12 # [12].
            total_length = 1
        else:
            r = 1.35 / 0.75
            stiffness = 30 # [x10x, 20, *60*, *80*].
            damping = 20 # [*20*].
            armature = 0.5 # [0.01, x0.1x *0.5*].
            frictionloss = 0.1 # [0.1, *0.7*].
            
            geom_margin = "0.01"
            solref = "0.0002 1"
            solimp = "0.95 0.99 0.01"
            friction = [1, 0.005, 0.0001] # ***[0.5, 0.005, 0.0001]***, [2, 0.005, 0.0001], [0.45, x0.5x, *2*, 2.6].

            gear_ratio_low = 200 # [*200*].
            gear_ratio = 200 # [*200*].
            gear_ratio_high = 200 # [200].
        
            total_mass = 0.1 # [-1]. Here the total mass is 0.62 and the mean inertia is 0.54.
            total_length = 5
        
        
        params = {
            "animal": "mole_cricket",
            "version": "v2",
            "mesh_rescale": np.array([0.75, 0.75, 0.75]) * r, # np.array([1.35, 1.35, 1.35])
            
            "body_pos": np.array([0.0, 0.0, 3 * (0.75 / 1.35)]) * r if gait == "dig" else np.array([0.0, 0.0, 0.8 * (0.75 / 1.35)]) * r, # np.array([0.0, 0.0, 0.8])
            "body_euler": [tilt_angle, 0, face_angle + 0],

            "gravity": gravity,
            "total_mass": total_mass,
            "total_length": total_length,
            "noslip_iterations": noslip_iterations,
            
            "default": {
                "joint": {
                    "armature": 1,
                    "frictionloss": 0.1,
                    "stiffness": "20",
                    "damping": "2",
                    "limited": "true",
                    "margin": 0, ## 0.01.
                    "ctrlrange": [-1, 1],
                },
                "geom": {
                    "density": None,
                    "mass": "0.01",
                    "solref": solref,
                    "solimp": solimp,
                    "friction": "0",
                    "margin": geom_margin,
                    "gap": "0",

                    "rgba": "0.8 0.6 0.4 1",
                    "contype": 2, # Use contype=4 and conaffinity=0 to exclude all contacts by default.
                    "conaffinity": 1,
                },
                "actuator": {
                    # "control_range": "-1 1",
                    # "gear_ratio": 100,
                },
                "floor": {
                    "mass": 4,
                    "friction": friction,
                    "contype": 1,
                    "conaffinity": 2,
                    "tilt_angle": tilt_angle,
                    "euler": f"{str(tilt_angle)} 0 0",
                },
            },
        }
        
        body_parts = {
            "head": {
                "mass": 0.14, # 0.14 [g]. r=1 => [0.01]
            },
            "thorax": {
                "mass": 0.85, # 0.85 [g], r=6.07 => [0.06].
            },
            "abdomen": {
                "mass": 1.05, # 1.05 [g]. r=7.5 => [0.075].
                "com": 0.95
            },
            "forelegs": {
                "coxa": {
                    "mass": 0.04
                },
                "femur": {
                    "mass": 0.04
                },
                "tibia": {
                    "mass": 0.02,
                    "friction": friction,
                },
                "tarsus": {
                    "mass": default_mass,
                    "size": [0.01, 0.01, 0.01],
                    "L1_pos": np.array([-0.326, 1.11, -0.16]) * r,
                    "R1_pos": np.array([0.326, 1.11, -0.16]) * r,
                },
            },
            "middlelegs": {
                "coxa": {
                    "mass": 0.014
                },
                "femur": {
                    "mass": 0.013
                },
                "tibia": {
                    "mass": 0.055,
                },
                "tarsus": {
                    "mass": default_mass,
                    "friction": friction,
                }
            },
            "hindlegs": {
                "coxa": {
                    "mass": 0.013
                },
                "femur": {
                    "mass": 0.03
                },
                "tibia": {
                    "mass": 0.01
                },
                "tarsus": {
                    "mass": default_mass,
                    "friction": friction,
                },
            }
        }
        
        joints = {
            "L1": {
                "J0": {
                    "pos": np.array([-0.2, 0.75, -0.05]) * r,
                    "axis": [0, 0, 1],
                    "range": [-60, 10], # [-60, -20]
                    "ref": 0,
                    "springref": 0, # -40
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_high,
                    "ctrlrange": [-1, 1],
                },
                "J1": {
                    "pos": np.array([-0.2, 0.75, -0.05]) * r,
                    "axis": [-1, -0.2, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_low,
                    "ctrlrange": [-1, 1],
                },
                "J2": {
                    "pos": np.array([-0.35, 1, 0.03]) * r,
                    "axis": [-1, -0.2, -0.2],
                    "range": [-40, 40],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                },
                "J3": {
                    "pos": np.array([-0.35, 1.11, -0.04]) * r,
                    "axis": [-1, -0.2, 0],
                    "range": [-1, 1],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                    "tip_pos": np.array([-0.6, 1.97, -0.36]),
                }
            },
            "R1": {
                "J0": {
                    "pos": np.array([0.2, 0.75, -0.05]) * r,
                    "axis": [0, 0, -1],
                    "range": [-60, 10], # [-60, -20]
                    "ref": 0,
                    "springref": 0, # -40
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_high,
                    "ctrlrange": [-1, 1],
                },
                "J1": {
                    "pos": np.array([0.2, 0.75, -0.05]) * r,
                    "axis": [-1, 0.2, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_low,
                    "ctrlrange": [-1, 1],
                },
                "J2": {
                    "pos": np.array([0.35, 1, 0.03]) * r,
                    "axis": [-1, 0.2, -0.2],
                    "range": [-40, 40],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                },
                "J3": {
                    "pos": np.array([0.35, 1.11, -0.04]) * r,
                    "axis": [-1, 0.2, 0],
                    "range": [-1, 1],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                    "tip_pos": np.array([0.6, 1.97, -0.36]),
                }
            },
            "L2": {
                "J0": {
                    "pos": np.array([-0.17, 0.23, -0.2]) * r,
                    "axis": [0, 0, 1],
                    "range": [-80, 120], ## [-40, 100], [-20, 40].
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_high,
                    "ctrlrange": [-1, 1],
                },
                "J1": {
                    "pos": np.array([-0.17, 0.23, -0.2]) * r,
                    "axis": [0, -1, 0],
                    "range": [-40, 40],
                    "ref": 0,
                    "frictionloss": 0.1,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_low,
                    "ctrlrange": [-1, 1],
                },
                "J2": {
                    "pos": np.array([-0.44, 0.38, 0.05]) * r,
                    "axis": [-1, -0.9, 0],
                    "range": [-10, 10],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                },
                "J3": {
                    "pos": np.array([-0.46, 0.42, -0.17]) * r,
                    "axis": [0.5, 1, 0.1],
                    "range": [-10, 10],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                    "tip_pos": np.array([-1.10, 1.0, -0.38]),
                }
            },
            "R2": {
                "J0": {
                    "pos": np.array([0.17, 0.23, -0.2]) * r,
                    "axis": [0, 0, -1],
                    "range": [-80, 120], ## [-40, 100], [-20, 40].
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_high,
                    "ctrlrange": [-1, 1],
                },
                "J1": {
                    "pos": np.array([0.17, 0.23, -0.2]) * r,
                    "axis": [0, 1, 0],
                    "range": [-40, 40],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_low,
                    "ctrlrange": [-1, 1],
                },
                "J2": {
                    "pos": np.array([0.44, 0.38, 0.05]) * r,
                    "axis": [-1, 0.9, 0],
                    "range": [-10, 10],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                },
                "J3": {
                    "pos": np.array([0.46, 0.42, -0.17]) * r,
                    "axis": [0.5, -1, 0.1],
                    "range": [-10, 10],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                    "tip_pos": np.array([1.10, 1.0, -0.38]),
                }
            },
            "L3": {
                "J0": {
                    "pos": np.array([-0.21, -0.22, -0.1]) * r,
                    "axis": [0, 0, 1],
                    "range": [-30, 30],
                    "ref": 20,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                },
                "J1": {
                    "pos": np.array([-0.21, -0.22, -0.1]) * r,
                    "axis": [1, -0.5, 0],
                    "range": [-20, 20], ## [-30, 30].
                    "ref": 0, ## -20,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_low,
                    "ctrlrange": [-1, 1],
                },
                "J2": {
                    "pos": np.array([-0.27, -0.75, 0.30]) * r,
                    "axis": [1, -0.1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_high,
                    "ctrlrange": [-1, 1],
                },
                "J3": {
                    "pos": np.array([-0.4, -0.82, -0.18]) * r,
                    "axis": [-1, 0.3, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                    "tip_pos": np.array([-0.8, -2, -0.34]),
                }
            },
            "R3": {
                "J0": {
                    "pos": np.array([0.21, -0.22, -0.1]) * r,
                    "axis": [0, 0, -1],
                    "range": [-30, 30],
                    "ref": 20,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                },
                "J1": {
                    "pos": np.array([0.21, -0.22, -0.1]) * r,
                    "axis": [1, 0.5, 0],
                    "range": [-20, 20], ## [-30, 30].
                    "ref": 0, ## -20,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_low,
                    "ctrlrange": [-1, 1],
                },
                "J2": {
                    "pos": np.array([0.27, -0.75, 0.30]) * r,
                    "axis": [1, 0.1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio_high,
                    "ctrlrange": [-1, 1],
                },
                "J3": {
                    "pos": np.array([0.4, -0.82, -0.18]) * r,
                    "axis": [-1, -0.3, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "ctrlrange": [-1, 1],
                    "tip_pos": np.array([0.6, 1.97, -0.36]),
                    "tip_pos": np.array([0.8, -2, -0.34]),
                }
            }
        }
    elif(animal_model == "desert_locust"):
        
        if(SI):
            r = 0.2
            
            gravity = "0 0 -9.81"
            
            stiffness = 50 # [100].
            damping = 4 # [*20*].
            armature = 0.1
            frictionloss = 0

            geom_margin = "0.002"
            solref = "0.02 1"
            solimp = "0.0 0.99 0.01"
            friction = [2, 0.1, 0.1] # [2, 0.1, 0.1].

            gear_ratio_high = 25 # Used for horizontal joints.
            gear_ratio_low = 25 # Vertical joints.
            gear_ratio = 25 # Other joints.

            total_mass = 12 # [12].
            total_length = 1
        else:
            r = 1

            gear_ratio_low = 120 # [*200*].
            gear_ratio = 120 # [*200*].
            gear_ratio_high = 120 # [200].

            stiffness = 30 # [x10x, 20, *60*, *80*].
            damping = 20 # [*2*].
            armature = 0.5 # [0.01, x0.1x *0.5*].
            frictionloss = 0.1 # [0.1, *0.7*].
            
            hind_ft_mode = False
            if(hind_ft_mode):
                stiffness_hind_ft = 700
                damping_hind_ft = 30
                frictionloss_hind_ft = 100
                armature_hind_ft = 0.5
                gear_ratio_hind_ft = 800
            else:
                stiffness_hind_ft = stiffness
                damping_hind_ft = damping
                armature_hind_ft = armature
                frictionloss_hind_ft = frictionloss
                gear_ratio_hind_ft = gear_ratio
            
            geom_margin = "0.01"
            solref = "0.0002 1"
            solimp = "0.95 0.99 0.01"
            friction = [1, 0.005, 0.0001] # ***[0.5, 0.005, 0.0001]***, [2, 0.005, 0.0001], [0.45, x0.5x, *2*, 2.6].

            total_mass = 0.06 # [-1]. The locust's mean mass is 1.5g, and the mole cricket's is 2.5g. In the simulation the MC mass is set to 0.1. Keeping the same ratio (1.5/2.5=0.6) gives 0.6*0.1 = 0.06.
            total_length = 5
        
        params = {
            "animal": "desert_locust",
            "version": "v4", # "v3".
            "mesh_rescale": np.array([0.775, 0.775, 0.775]) * r,
            "body_pos": np.array([0.0, 0.0, 1.05]) * r, # v2 = [0.0, 0.0, 0.2].
            "body_euler": [tilt_angle, 0, face_angle+180],

            "gravity": gravity, # Gravity,
            "total_mass": total_mass, # * 0.001, # 1.5g = 0.0015kg. [1.5].
            "total_length": total_length,
            "noslip_iterations": noslip_iterations,
            
            "default": {
                "joint": {
                    "stiffness": str(stiffness),
                    "damping": str(damping),
                    "armature": 1,
                    "frictionloss": 0.1,
                    "limited": "true",
                    "margin": 0,
                    "ctrlrange": [-1, 1],
                },
                "geom": {
                    "density": None,
                    "mass": "0.01",
                    "solref": solref,
                    "solimp": solimp,
                    "friction": "0",
                    "margin": geom_margin,
                    "gap": "0",

                    "rgba": "0.8 0.6 0.4 1",
                    "contype": 2,
                    "conaffinity": 1,
                },
                "actuator": {
                    "control_range": "-1 1",
                    # "gear_ratio": 100,
                },
                "floor": {
                    "mass": 4,
                    "friction": friction,
                    "contype": 1,
                    "conaffinity": 2,
                    "tilt_angle": tilt_angle,
                    "euler": f"{str(tilt_angle)} 0 0",
                },
            },
        }
        
        body_parts = {
            "head": {
                "mass": 0.125, ## [m=0.125 [g], 0.05]  r=1. x=0.01.
            },
            "thorax": {
                "mass": 0.6, ## [m=0.6 [g], 0.05]. r=4.8. r*x=0.048.
            },
            "abdomen": {
                "mass": 0.45, ## [m=0.45 [g], 0.05]. r=3.6. r*x=0.036.
                "com": 1.15,
            },
            "forelegs": {
                "coxa": {
                    "mass": 0.002,
                },
                "femur": {
                    "mass": 0.004
                },
                "tibia": {
                    "mass": 0.002
                },
                "tarsus": {
                    "mass": default_mass,
                    "friction": friction
                },
            },
            "middlelegs": {
                "coxa": {
                    "mass": 0.003
                },
                "femur": {
                    "mass": 0.01
                },
                "tibia": {
                    "mass": 0.03
                },
                "tarsus": {
                    "mass": default_mass,
                    "friction": friction
                }
            },
            "hindlegs": {
                "coxa": {
                    "mass": 0.008
                },
                "femur": {
                    "mass": 0.07
                },
                "tibia": {
                    "mass": 0.01
                },
                "tarsus": {
                    "mass": default_mass,
                    "friction": friction
                },
            }
        }
        
        joints = {
            "L1": {
                "J0": {
                    "pos": np.array([0.48, -1.58, -0.18]) * r,
                    "axis": [0, 0, 1],
                    "range": [-60, 60], # [-60, 20].
                    "ref": 30, # 0.
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": np.array([0.48, -1.58, -0.18]) * r,
                    "axis": [0, 1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "springref": -10,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": np.array([1.25, -1.65, -0.05]) * r,
                    "axis": [0.2, 1, 0],
                    "range": [-30, 30],
                    "ref": 0, # Consider changing this to -30.
                    "springref": 20, # Consider changing this to 0.
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J3": {
                    "pos": np.array([1.55, -1.63, -0.6]) * r,
                    "axis": [-0.05, -1, 0],
                    "range": [-20, 20],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "tip_pos": np.array([1.95, -1.58, -0.65]),
                }
            },
            "R1": {
                "J0": {
                    "pos": np.array([-0.48, -1.58, -0.18]) * r,
                    "axis": [0, 0, -1],
                    "range": [-60, 60], # [-60, 20].
                    "ref": 30, # 0.
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": np.array([-0.48, -1.58, -0.18]) * r,
                    "axis": [0, -1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "springref": -10,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": np.array([-1.25, -1.65, -0.05]) * r,
                    "axis": [0.2, -1, 0],
                    "range": [-30, 30],
                    "ref": 0, # Consider changing this to -30.
                    "springref": 20, # Consider changing this to 0.
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J3": {
                    "pos": np.array([-1.55, -1.63, -0.6]) * r,
                    "axis": [-0.05, 1, 0],
                    "range": [-20, 20],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "tip_pos": np.array([-1.95, -1.58, -0.65]),
                }
            },
            "L2": {
                "J0": {
                    "pos": np.array([0.58, -0.93, -0.13]) * r,
                    "axis": [0, 0, 1],
                    "range": [-50, 60], # [-40, 100], [-20, 40].
                    "ref": 0, # -30.
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": np.array([0.58, -0.93, -0.13]) * r,
                    "axis": [-0.6, 1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": np.array([1.15, -0.44, 0.21]) * r,
                    "axis": [-0.8, 1, 0],
                    "range": [-60, 60],
                    "ref": 0, # 
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J3": {
                    "pos": np.array([1.15, -0.454, -0.64]) * r,
                    "axis": [1, -1, 0],
                    "range": [-20, 20],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "tip_pos": np.array([1.4, -0.1, -0.68]),
                }
            },
            "R2": {
                "J0": {
                    "pos": np.array([-0.58, -0.93, -0.13]) * r,
                    "axis": [0, 0, -1],
                    "range": [-50, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": np.array([-0.58, -0.93, -0.13]) * r,
                    "axis": [-0.6, -1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": np.array([-1.15, -0.44, 0.21]) * r,
                    "axis": [-0.8, -1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J3": {
                    "pos": np.array([-1.15, -0.454, -0.64]) * r,
                    "axis": [1, 1, 0],
                    "range": [-20, 20],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "tip_pos": np.array([-1.4, -0.1, -0.68]),
                }
            },
            "L3": {
                "J0": {
                    "pos": np.array([0.55, -0.3, -0.15]) * r,
                    "axis": [0, 0, 1],
                    "range": [-15, 15],
                    "ref": -10,
                    "springref": -10,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": np.array([0.55, -0.3, -0.15]) * r,
                    "axis": [-1, 0.7, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "springref": 20,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": np.array([1.05, 1.12, 1.63]) * r,
                    "axis": [-1, 0.4, 0],
                    "range": [-60, 60],
                    "ref": 0,

                    "stiffness": stiffness_hind_ft,
                    "damping": damping_hind_ft,
                    "frictionloss": frictionloss_hind_ft,
                    "armature": armature_hind_ft,
                    "gear_ratio": gear_ratio_hind_ft,
                },
                "J3": {
                    "pos": np.array([1.66, 2.1, -0.23]) * r,
                    "axis": [1, -0.5, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "tip_pos": np.array([1.88, 2.61, -0.04]),
                }
            },
            "R3": {
                "J0": {
                    "pos": np.array([-0.55, -0.3, -0.15]) * r,
                    "axis": [0, 0, -1],
                    "range": [-15, 15],
                    "ref": -10,
                    "springref": -10,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": np.array([-0.55, -0.3, -0.15]) * r,
                    "axis": [-1, -0.7, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "springref": 20,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": np.array([-1.05, 1.12, 1.63]) * r,
                    "axis": [-1, -0.4, 0],
                    "range": [-60, 60],
                    "ref": 0,

                    "stiffness": stiffness_hind_ft,
                    "damping": damping_hind_ft,
                    "frictionloss": frictionloss_hind_ft,
                    "armature": armature_hind_ft,
                    "gear_ratio": gear_ratio_hind_ft,
                },
                "J3": {
                    "pos": np.array([-1.66, 2.1, -0.23]) * r,
                    "axis": [1, 0.5, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                    "tip_pos": np.array([-1.88, 2.61, -0.04]),
                }
            }
        }
    elif(animal_model == "carpenter_ant"):
        # TODO: use 5mg (=0.005 g) to set mass of body parts.
        default_mass = 0.0001
        
        stiffness = 180 # [20, 45, 200].
        damping = 2 # 2.
        frictionloss = 0.1 # [0.1, 0.7].
        armature = 0.01 # [0.01, 0.5].
        friction = 2 # [0.45, 2.6].
        gear_ratio = 250
        
        params = {
            "animal": animal_model,
            "version": "v0",
            "mesh_rescale": 1.0,
            "body_pos": [0.0, 0.0, 0.1],
            "body_euler": [tilt_angle, 0, face_angle+180],

            "gravity": gravity,
            "noslip_iterations": noslip_iterations,

            "settotalmass": 0.005,
            "inertiafromgeom": "false",
            
            "default": {
                "joint": {
                    "armature": 0.01,
                    "frictionloss": 0.1,
                    "limited": "true",
                    "margin": 0.01,
                },
                "geom": {
                    # "mass": "0.01",
                    "density": 0.5, # CGS.
                    "solref": "0.0002 1",
                    "solimp": "0.95 0.99 0.01",
                    "friction": "0",
                    "margin": "0.01", # "0.0005",
                    "gap": "0", # "0.0005",
                    "rgba": "0.8 0.6 0.4 1",
                    "contype": 2,
                    "conaffinity": 1,
                },
                "actuator": {
                    "control_range": "-1 1",
                    # "gear_ratio": 100,
                },
                "floor": {
                    "mass": 4,
                    "friction": friction,
                    "contype": 1,
                    "conaffinity": 2,
                    "tilt_angle": tilt_angle,
                    "euler": f"{str(tilt_angle)} 0 0",
                },
            },
        }
        
        body_parts = {
            "head": {
                "mass": 0.125, ## [*0.125*, 0.05]  r=1.
            },
            "thorax": {
                "mass": 0.6, ## [*0.6*, 0.05]. r=4.8.
            },
            "abdomen": {
                "mass": 0.45, ## [*0.45*, 0.05]. r=3.6.
                "com": 1.15,
            },
            "forelegs": {
                "coxa": {
                    "mass": 0.0002,
                },
                "femur": {
                    "mass": 0.0004
                },
                "tibia": {
                    "mass": 0.0002
                },
                "tarsus": {
                    "mass": default_mass,
                    "friction": friction
                },
            },
            "middlelegs": {
                "coxa": {
                    "mass": 0.0003
                },
                "femur": {
                    "mass": 0.001
                },
                "tibia": {
                    "mass": 0.03
                },
                "tarsus": {
                    "mass": default_mass,
                    "friction": friction
                }
            },
            "hindlegs": {
                "coxa": {
                    "mass": 0.0008
                },
                "femur": {
                    "mass": 0.007
                },
                "tibia": {
                    "mass": 0.001
                },
                "tarsus": {
                    "mass": default_mass,
                    "friction": friction
                },
            }
        }
        
        joints = {
            "L1": {
                "J0": {
                    "pos": [0.042, -0.15, 0.45],
                    "axis": [0, 0, 1],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": [0.042, -0.15, 0.45],
                    "axis": [1, 0.65, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": [0.135, -0.31, 0.39],
                    "axis": [1, 0.8, 0],
                    "range": [-10, 10],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J3": {
                    "pos": [0.165, -0.362, 0.2025],
                    "axis": [1, 0.8, 0],
                    "range": [-20, 20],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                }
            },
            "R1": {
                "J0": {
                    "pos": [-0.042, -0.15, 0.45],
                    "axis": [0, 0, -1],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": [-0.042, -0.15, 0.45],
                    "axis": [1, -0.65, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": [-0.135, -0.31, 0.39],
                    "axis": [1, -0.8, 0],
                    "range": [-10, 10],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J3": {
                    "pos": [-0.165, -0.362, 0.2025],
                    "axis": [1, -0.8, 0],
                    "range": [-20, 20],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                }
            },
            "L2": {
                "J0": {
                    "pos": [0.1, -0.078, 0.378],
                    "axis": [0, 0, 1],
                    "range": [-60, 60], # [-40, 100], [-20, 40].
                    "ref": 0, # -30.
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": [0.1, -0.078, 0.378],
                    "axis": [0, 1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": [0.26, -0.065, 0.375],
                    "axis": [0, 1, 0],
                    "range": [-60, 60],
                    "ref": 0, # 
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J3": {
                    "pos": [0.383, -0.057, 0.185],
                    "axis": [0, 1, 0],
                    "range": [-20, 20],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                }
            },
            "R2": {
                "J0": {
                    "pos": [-0.1, -0.078, 0.378],
                    "axis": [0, 0, -1],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": [-0.1, -0.078, 0.378],
                    "axis": [0, -1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J2": {
                    "pos": [-0.26, -0.065, 0.375],
                    "axis": [0, -1, 0],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J3": {
                    "pos": [-0.383, -0.057, 0.185],
                    "axis": [0, -1, 0],
                    "range": [-20, 20],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                }
            },
            "L3": {
                "J0": {
                    "pos": [0.074, 0.038, 0.37],
                    "axis": [0, 0, 1],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": [0.074, 0.038, 0.37],
                    "axis": [-0.7, 1, 0.2],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": 2*gear_ratio,
                },
                "J2": {
                    "pos": [0.219, 0.132, 0.387],
                    "axis": [-0.7, 1, 0.2],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": 2*gear_ratio,
                },
                "J3": {
                    "pos": [0.37, 0.28, 0.205],
                    "axis": [-0.2, 1, 0.2],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                }
            },
            "R3": {
                "J0": {
                    "pos": [-0.074, 0.038, 0.37],
                    "axis": [0, 0, -1],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                },
                "J1": {
                    "pos": [-0.074, 0.038, 0.37],
                    "axis": [-0.7, -1, 0.2],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": 2*gear_ratio,
                },
                "J2": {
                    "pos": [-0.219, 0.132, 0.387],
                    "axis": [-0.7, -1, 0.2],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": 2*gear_ratio,
                },
                "J3": {
                    "pos": [-0.37, 0.28, 0.205],
                    "axis": [-0.2, -1, -0.2],
                    "range": [-60, 60],
                    "ref": 0,
                    "frictionloss": frictionloss,
                    "armature": armature,
                    "stiffness": stiffness,
                    "damping": damping,
                    "gear_ratio": gear_ratio,
                }
            }
        }
    elif(animal_model == "biped"):
        friction = 2.6
        params = {
            "body_pos": [0.0, 0.0, 0.25],
            "body_euler": [0, 0, 0],
            
            "gravity": gravity,
            "noslip_iterations": noslip_iterations,

            "default": {
                "joint": {
                    "armature": 0.01,
                    "frictionloss": 0.1,
                    "limited": "true",
                    "margin": 0.01,
                },
                "geom": {
                    "density": None,
                    "mass": "0.01",
                    "solref": "0.0002 1",
                    "solimp": "0.95 0.99 0.01",
                    "friction": "0",
                    "margin": "0.0", # "0.0005",
                    "gap": "0", # "0.0005",
                    "rgba": "0.8 0.6 0.4 1",
                    "contype": 2,
                    "conaffinity": 1,
                },
                "floor": {
                    "mass": 4,
                    "friction": friction,
                    "contype": 1,
                    "conaffinity": 2,
                    "tilt_angle": tilt_angle,
                    "euler": f"{str(tilt_angle)} 0 0",
                },

            },
        }
        
        body_parts = {
            "foreleg": {
                "femur": {
                    "size": [0.04, 0.2],
                    "euler": [-20, 0, 0],
                    "friction": friction,
                    "mass": 0.01,
                }
            },
            "hindleg": {
                "femur": {
                    "size": [0.04, 0.2],
                    "euler": [0, 0, 0],
                    "friction": friction,
                    "mass": 0.01,
                }
            }
        }
        
        joints = {
            "foreleg": {
                "J0": {
                    "frictionloss": 0.1,
                    "armature": 0.01,
                    # "springdamper": [0.01, 0.1],
                    "stiffness": 45,
                    "damping": 4,
                    "pos": [0, 0.07, 0.2],
                    "axis": [1, 0, 0],
                    "range": [-30, 30],
                    "ref": -5,
                }
            },
            "hindleg": {
                "J0": {
                    "frictionloss": 0.1,
                    "armature": 0.01,
                    # "springdamper": [0, 0],
                    "stiffness": 100,
                    "damping": 4,
                    "pos": [0, 0, 0.2],
                    "axis": [1, 0, 0],
                    "range": [-1, 1],
                    "ref": 0,
                }
            }
        }
    elif(animal_model == "triped"):
        friction = 2.6
        params = {
            "body_pos": [0.0, 0.0, 0.35],
            "body_euler": [0, 0, 0],
            
            "gravity": gravity,
            "noslip_iterations": noslip_iterations,

            "default": {
                "joint": {
                    "armature": 0.01,
                    "frictionloss": 0.1,
                    "limited": "true",
                    "margin": 0.01,
                },
                "geom": {
                    "density": None,
                    "mass": "0.01",
                    "solref": "0.0002 1",
                    "solimp": "0.95 0.99 0.01",
                    "friction": "0",
                    "margin": "0.0", # "0.0005",
                    "gap": "0", # "0.0005",
                    "rgba": "0.8 0.6 0.4 1",
                    "contype": 2,
                    "conaffinity": 1,
                },
                "floor": {
                    "mass": 4,
                    "friction": friction,
                    "contype": 1,
                    "conaffinity": 2,
                    "tilt_angle": tilt_angle,
                    "euler": f"{str(tilt_angle)} 0 0",
                },
                
            },
        }
        
        body_parts = {
            "L1": {
                "femur": {
                    "size": [0.03, 0.1],
                    "euler": [-20, 0, 0],
                    "friction": friction,
                    "mass": 0.01,
                }
            },
            "R1": {
                "femur": {
                    "size": [0.03, 0.1],
                    "euler": [-20, 0, 0],
                    "friction": friction,
                    "mass": 0.01,
                }
            },
            "hindleg": {
                "femur": {
                    "size": [0.03, 0.1],
                    "euler": [0, 0, 0],
                    "friction": friction,
                    "mass": 0.01,
                }
            }
        }
        
        joints = {
            "L1": {
                "J0": {
                    "frictionloss": 0.1,
                    "armature": 0.01,
                    # "springdamper": [0.01, 0.1],
                    "stiffness": 45,
                    "damping": 4,
                    "pos": [0, 0.04, 0.1],
                    "axis": [1, 0, 0],
                    "range": [-30, 30],
                    "ref": -5,
                }
            },
            "R1": {
                "J0": {
                    "frictionloss": 0.1,
                    "armature": 0.01,
                    # "springdamper": [0.01, 0.1],
                    "stiffness": 45,
                    "damping": 4,
                    "pos": [0, 0.04, 0.1],
                    "axis": [1, 0, 0],
                    "range": [-30, 30],
                    "ref": -5,
                }
            },
            "hindleg": {
                "J0": {
                    "frictionloss": 0.1,
                    "armature": 0.01,
                    # "springdamper": [0, 0],
                    "stiffness": 100,
                    "damping": 4,
                    "pos": [0, 0, 0.1],
                    "axis": [1, 0, 0],
                    "range": [-1, 1],
                    "ref": 0,
                }
            }
        }
    elif(animal_model == "half_cheetah"):
        params = {
            "gravity": "0 0 -9.81",

            "default": {
                "joint": {
                    "armature": ".1",
                    "damping": ".01",
                    "solreflimit": ".02 1",
                    "solimplimit": "0 .8 .03",
                    "stiffness": "8",
                },
                
                "geom": {
                    "solref": "0.02 1",
                    "solimp": "0.0 0.8 0.01",
                    "friction": ".4 .1 .1",
                    "margin": "0",
                }
            }
        }

        body_parts = {}

        joints = {
            "front_leg": {
                "thigh": {
                    "range": "-1 .7",
                    "stiffness": "180",
                    "damping": "4.5",
                    "gear": "120",
                },
                "shin": {
                    "range": "-1.2 .87",
                    "stiffness": "120",
                    "damping": "3",
                    "gear": "60",
                },
                "foot": {
                    "range": "-.5 .5",
                    "stiffness": "60",
                    "damping": "1.5",
                    "gear": "30",
                },
            },
            "back_leg": {
                "thigh": {
                    "range": "-.52 1.05",
                    "stiffness": "240",
                    "damping": "6",
                    "gear": "120",
                },
                "shin": {
                    "range": "-.785 .785",
                    "stiffness": "180",
                    "damping": "4.5",
                    "gear": "90",
                },
                "foot": {
                    "range": "-.4 .785",
                    "stiffness": "120",
                    "damping": "3",
                    "gear": "60",
                },
            },
        }
    
    return params, joints, body_parts