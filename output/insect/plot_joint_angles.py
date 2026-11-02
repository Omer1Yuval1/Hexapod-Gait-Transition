import numpy as np
import os
# from scipy.stats import circstd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from envs.mujoco.functions.apply_pca_and_plot_limit_cycle import apply_pca_and_plot_limit_cycle

def plot_joint_angles(gaits, dt_opt, t, t_max, qpos, qvel, actions, colormap, save_dir, ep):

    qpos = qpos * 180 / np.pi
    qvel = qvel * 180 / np.pi
    
    gaits[0]["params"]["hind_joints"] = {"qpos": [23,27, 24,28, 25,29], "qvel": [22,26, 23,27, 24,28]} # Hind legs J0, J1, J2.
    
    for i in (0, 2): # For horizontal, vertical and hind joints.
        if(i == 0):
            J = gaits[0]["params"]["horizontal_joints"]["qpos"]
            dJ = gaits[0]["params"]["horizontal_joints"]["qvel"]

            """ if(not gaits[0]["type"] == "jump"):
                A = gaits[0]["amplitudes"]["horizontal"] * 180 / np.pi
                offsets = gaits[0]["offsets"]["horizontal"] * 180 / np.pi """
            
            joint_type = "J0"
        elif(i == 1):
            J = gaits[0]["params"]["vertical_joints"]["qpos"]
            dJ = gaits[0]["params"]["vertical_joints"]["qvel"]
            
            """ if(not gaits[0]["type"] == "jump"):
                A = gaits[0]["amplitudes"]["vertical"] * 180 / np.pi
                offsets = gaits[0]["offsets"]["vertical"] * 180 / np.pi """
            
            joint_type = "J1"
        elif(i == 2): # Hind leg joints.
            J = gaits[0]["params"]["hind_joints"]["qpos"]
            dJ = gaits[0]["params"]["hind_joints"]["qvel"]
            # A = gaits[0]["params"]["amplitudes"]["vertical"] * 180 / np.pi
            # offsets = gaits[0]["params"]["offsets"]["vertical"] * 180 / np.pi
            joint_type = "hind_J0+J1+J2"

        fig1 = plt.figure(figsize=(260, 195))
        fig1.tight_layout()
        plt.gcf().set_size_inches(12, 9)
        plt.rcParams.update({'font.size': 14})
        axes_1 = [fig1.add_subplot(311 + j) for j in range(3)]
        
        plt.subplots_adjust(left=0.09, right=0.91, top=0.98, bottom=0.06)
        
        for j, ax in enumerate(axes_1): # For each leg pair.
            ax.plot(t, qpos[:, J[2 * j]], '-', c=colormap(j)) # Left joint.
            # ax.scatter(t, qpos[:, J[2 * j]], color='k', s=5, alpha=0.7) # Also plot dots over the line of the left joint to view its resolution.
            ax.plot(t, qpos[:, J[2 * j + 1]], '--', c=colormap(j)) # Right joint.
            
            if(i < 2 and not gaits[0]["type"] == "jump"):
                # Plot a 2Hz sine wave with amplitude of A[2 * j] and offset of offsets[2 * j]
                """ sine_wave = A[2 * j] * np.sin(2 * np.pi * 2 * t) + offsets[2 * j]
                ax.plot(t, sine_wave, '-', c='gray', alpha=0.5) """
                # print("circular std of sine wave=", np.round(circstd(sine_wave*np.pi/180, high=np.pi*2, low=0)*180/np.pi, 2))
                # print("linear std of sine wave=", np.round(np.std(sine_wave), 2))

                """ if(not gaits[0]["type"] == "jump"):
                    ax.set_yticks(np.arange(-A[2 * j] + offsets[2 * j], A[2 * j] + offsets[2 * j] + A[2 * j] / 2, A[2 * j] / 2))
                    ax.set_ylim(-A[2 * j] + offsets[2 * j], A[2 * j] + offsets[2 * j]) """
            
            ax.set_xlim(t[0], t_max)
            ax.set_ylabel('Joint angle [°]')
            # print(-A[2 * j] + offsets[2 * j] , A[2 * j] + offsets[2 * j], np.arange(-A[2 * j] + offsets[2 * j], A[2 * j] + offsets[2 * j] + A[2 * j] / 2, A[2 * j] / 2))
        
        axes_1[-1].set_xlabel('Time')
        
        # plt.show()
        plt.savefig(os.path.join(save_dir, f"joint_angles_pairwise_{joint_type}_ep={ep}.png"), bbox_inches='tight')
        plt.close(fig1)
        
        # Plot joint velocities
        # ---------------------

        fig3 = plt.figure(figsize=(260,195))
        
        fig3.tight_layout()
        plt.gcf().set_size_inches(12, 9)
        plt.rcParams.update({'font.size': 14})
        
        axes_2 = [fig3.add_subplot(311 + j) for j in range(3)]
        
        plt.subplots_adjust(left=0.09, right=0.91, top=0.98, bottom=0.06)

        for j, ax in enumerate(axes_2):
            ax.plot(t, qvel[:, dJ[2 * j]], '-', c=colormap(j))
            ax.scatter(t, qvel[:, dJ[2 * j]], color='k', s=5, alpha=0.7)
            ax.plot(t, qvel[:, dJ[2 * j + 1]], '--', c=colormap(j))
            ax.set_xlim(t[0], t_max)
            y_min = min(np.min(qvel[:, dJ[2 * j]]), np.min(qvel[:, dJ[2 * j + 1]]))
            y_max = max(np.max(qvel[:, dJ[2 * j]]), np.max(qvel[:, dJ[2 * j + 1]]))
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel('Joint velocity [°/s]')
        
        axes_2[-1].set_xlabel('Time')
        
        # plt.show()
        plt.savefig(os.path.join(save_dir, f"joint_velocity_pairwise_{joint_type}_ep={ep}.png"), bbox_inches='tight')
        plt.close(fig3)

    # Plot limit cycle of joint angle pairs
    # -------------------------------------
    fig2, axes_2 = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), subplot_kw={'aspect': 'equal'})
    fig2.tight_layout()
    plt.gcf().set_size_inches(12, 9)
    plt.rcParams.update({'font.size': 14})

    J = gaits[0]["params"]["horizontal_joints"]["qpos"]
    for i, ax in enumerate(axes_2):
        x = qpos[:, J[2 * i]]
        y = qpos[:, J[2 * i+1]]
        
        
        # Format data for LineCollection using x and y (NOT t)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='turbo') # Create the LineCollection.

        # Map the time variable 't' to the color gradient
        lc.set_array(t)
        lc.set_linewidth(2)
    
        # Add the collection and auto-scale
        ax.add_collection(lc)
        ax.autoscale()

    plt.savefig(os.path.join(save_dir, f"limit_cycle_of_horizontal_joint_angles_pairwise_ep={ep}.png"), bbox_inches='tight')
    plt.close(fig2)

    # Plot limit cycle of joint angles and velocities
    # -----------------------------------------------
    J_hor = gaits[0]["params"]["horizontal_joints"]["qpos"]
    # dJ_hor = gaits[0]["params"]["horizontal_joints"]["qvel"]

    J_ver = gaits[0]["params"]["vertical_joints"]["qpos"]
    # dJ_ver = gaits[0]["params"]["vertical_joints"]["qvel"]

    J_combined = np.concatenate([J_hor, J_ver]) # , dJ_hor, , dJ_ver

    apply_pca_and_plot_limit_cycle(qpos[:, J_hor], dt_opt, save_dir, f"limit_cycle_of_PCA_of_horizontal_joint_angles_{ep}")
    
    apply_pca_and_plot_limit_cycle(qpos[:, J_combined], dt_opt, save_dir, f"limit_cycle_of_PCA_of_optimized_joint_angles_{ep}")

    apply_pca_and_plot_limit_cycle(qpos[:, 1:], dt_opt, save_dir, f"limit_cycle_of_PCA_of_qpos_{ep}") # z-pos, body quaternion, joint angles.

    # return qpos[:, J_combined]