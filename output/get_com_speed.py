import numpy as np
import os
import matplotlib.pyplot as plt

def get_com_speed(dt_opt, qpos, t_max):
    
    # x = qpos[:,0]
    y = qpos[:,1]
    
    signed_speed = np.diff(y) / dt_opt  # Calculate the speed in the y direction.
    
    # speed = (np.diff(x)**2 + np.diff(y)**2)**0.5
    # signed_speed = speed * np.sign(np.diff(y))
    
    return signed_speed