import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from envs.mujoco.functions.get_leg_angles import get_leg_angles
from envs.mujoco.functions.plot_leg_angles import plot_leg_angles
from envs.mujoco.functions.compute_stance import compute_stance
from output.insect.plot_swing_stance import plot_swing_stance

def get_leg_angles_all(geom_xpos, qpos, objects_num, P=None, t=None, save_path=None, plot=False, name=0):
    
    # geom_xpos is a 2d array of coordinates (n x [x,y,z]).

    test = False

    i0 = 1 + objects_num # The first object is the floor, followed by walls.
    
    # Set object indices in geom_xpos
    # leg_base_tip_indices = np.array([[4,7], [8,11], [12,15], [16,19], [20,23], [24,27]])
    leg_base_tip_indices = np.array([[i0+3,i0+6], [i0+7,i0+10], [i0+11,i0+14], [i0+15,i0+18], [i0+19,i0+22], [i0+23,i0+26]])

    # print("leg_base_tip_indices =", leg_base_tip_indices)
    # print("geom_xpos =", geom_xpos.shape)
    
    leg_angles = np.empty([geom_xpos.shape[0], leg_base_tip_indices.shape[0]]) # shape=[time, legs].
    leg_tip_x_positions = np.empty([geom_xpos.shape[0], leg_base_tip_indices.shape[0]])
    # print(i0, leg_base_tip_indices)
    
    horizontal_joint_indices = np.array([7,11,15,19,25,29])
    vertical_joint_indices =   np.array([8,12,16,20,24,28])
    
    if(test):
        # plt.figure(figsize=(8, 6))
        fig, ax0 = plt.subplots()
        ax0.axis('equal')
    
    for i, x in enumerate(geom_xpos): # For each step.
        # print("geom_xpos_i=", x)
        head = x[i0]
        thorax = x[i0+1]
        abdomen = x[i0+2]
        
        '''
        main_body_axes = thorax - abdomen
        main_body_axis_angle = np.arctan2(main_body_axes[1], main_body_axes[0])
        
        rotation_mat = R.from_euler('z', -main_body_axis_angle, degrees=False)
        rotation_mat = rotation_mat.as_matrix()[:2, :2]
        
        for j, l in enumerate(leg_base_tip_indices): # For each leg.
            leg_base = x[l[0]][:2]
            leg_tip = x[l[1]][:2]
            leg_vector = leg_tip - leg_base
            
            leg_vector = rotation_mat @ leg_vector
            leg_angles[i,j] = np.arctan2(leg_vector[1], leg_vector[0])

            leg_tip_thorax_vector = leg_tip - thorax[:2]
            leg_tip_thorax_vector = rotation_mat @ leg_tip_thorax_vector

            leg_tip_x_positions[i,j] = leg_tip_thorax_vector[0]
        '''

        leg_base_tip_coordinates = [(x[idx[0]], x[idx[1]]) for idx in leg_base_tip_indices]
        leg_angles[i,:], leg_tip_x_positions[i,:] = get_leg_angles(thorax, abdomen, leg_base_tip_coordinates)

        if(test):
            # print(head, thorax, abdomen)
            ax0.plot(head[0], head[1], 'ro', markersize=0.1)
            ax0.plot(thorax[0], thorax[1], 'go', markersize=0.1)
            ax0.plot(abdomen[0], abdomen[1], 'bo', markersize=0.1)
            # plt.annotate('', xy=leg_vector, xytext=[0,0], arrowprops=dict(facecolor='red', arrowstyle='->', lw=.5))
            # plt.arrow(0, 0, leg_vector[0], leg_vector[1], width = 0.05)
    
    if(test):       
        plt.show()
    
    leg_angles_mean = np.empty([leg_angles.shape[1], ])
    leg_angles_std = np.empty([leg_angles.shape[1], ])

    if(not save_path is None):
        qpos_horizontal_mean = np.empty([horizontal_joint_indices.shape[0], ])
        qpos_horizontal_std = np.empty([horizontal_joint_indices.shape[0], ])

        # print("circular std of joint angles=", np.round(circstd(qpos[:, horizontal_joint_indices], high=np.pi*2, low=0, axis=0)*180/np.pi, 2))
        # print("linear std of joint angles=", np.round(np.std(qpos[:, horizontal_joint_indices], axis=0)*180/np.pi, 2))
        
        qpos_vertical_mean = np.empty([vertical_joint_indices.shape[0], ])
        qpos_vertical_std = np.empty([vertical_joint_indices.shape[0], ])

        stance = np.empty([leg_base_tip_indices.shape[0], geom_xpos.shape[0]])
    
    for i in range(leg_angles.shape[1]): # For each leg.
        leg_angles_i = leg_angles[:, i]
        leg_angles_mean[i] = circmean(leg_angles_i, high=np.pi*2, low=0)
        leg_angles_std[i] = circstd(leg_angles_i, high=np.pi*2, low=0)
        
        if(not save_path is None):
            qpos_horizontal = qpos[:, horizontal_joint_indices[i]]
            qpos_horizontal_mean[i] = circmean(qpos_horizontal, high=np.pi*2, low=0)
            qpos_horizontal_std[i] = circstd(qpos_horizontal, high=np.pi*2, low=0)
            
            qpos_vertical = qpos[:, vertical_joint_indices[i]]
            qpos_vertical_mean[i] = circmean(qpos_vertical, high=np.pi*2, low=0)
            qpos_vertical_std[i] = circstd(qpos_vertical, high=np.pi*2, low=0)

            stance[i,:] = compute_stance(leg_tip_x_positions[:,i], P["gaits"][0]["direction"])
    
    
    leg_phases_tip_position, leg_tip_x_positions_normalized = plot_leg_angles(P["dt_opt"], t, P["multi_gait"], save_path, plot, name, leg_angles, leg_tip_x_positions, qpos[:, horizontal_joint_indices], qpos[:, vertical_joint_indices], leg_angles_mean, leg_angles_std, qpos_horizontal_mean, qpos_horizontal_std, qpos_vertical_mean, qpos_vertical_std)

    if(not save_path is None):
    
        # Plot swing-stance based on leg tip positions
        stance[stance == 0] = 0.3
        plot_swing_stance(stance.T, save_path, f"leg_tip_ep={name}")

    '''
    if(not save_path is None):
        # plt.show()
        fig.savefig(os.path.join(save_path, f"leg_angles_{name}.png"), bbox_inches='tight')
        fig1.savefig(os.path.join(save_path, f"horizontal+vertical_joint_angles_{name}.png"), bbox_inches='tight')
        fig3.savefig(os.path.join(save_path, f"leg_tip_x_positions_{name}_win={win_size}.png"), bbox_inches='tight')
    '''

    return leg_angles_mean, leg_angles_std, leg_angles[-1,:], leg_phases_tip_position, leg_tip_x_positions_normalized