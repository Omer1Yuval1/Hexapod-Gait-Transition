import numpy as np
from scipy.spatial.transform import Rotation as R

def get_leg_angles(anterior_body_point, posterior_body_point, leg_base_tip_coordinates):
    
    '''
        if the thorax is the anterior body point, is it correct to use it as a rotation origin for the leg tip vector?
        verify all this by plotting before and after rotation.
    '''

    leg_angles = np.empty(len(leg_base_tip_coordinates),)
    leg_tip_x_positions = np.empty(len(leg_base_tip_coordinates),)

    main_body_axes = anterior_body_point - posterior_body_point
    main_body_axis_angle = np.arctan2(main_body_axes[1], main_body_axes[0])

    rotation_mat = R.from_euler('z', -main_body_axis_angle, degrees=False)
    rotation_mat = rotation_mat.as_matrix()[:2, :2]

    for i, leg in enumerate(leg_base_tip_coordinates): # For each leg.
        leg_base = leg[0][:2] # x,y of leg base.
        leg_tip = leg[1][:2] # x,y of leg tip.
        leg_vector = leg_tip - leg_base
        
        leg_vector = rotation_mat @ leg_vector
        leg_angles[i] = np.arctan2(leg_vector[1], leg_vector[0])

        leg_tip_vector = leg_tip - anterior_body_point[:2]
        leg_tip_vector = rotation_mat @ leg_tip_vector

        leg_tip_x_positions[i] = leg_tip_vector[0]

    return leg_angles, leg_tip_x_positions