import numpy as np
# import pandas as pd
from scipy import interpolate
import os
import matplotlib.pyplot as plt

def plot_joint_torques(P, t, t_max, actions, colormap, save_dir, ep):
    
    # window = 25
    smoothing_factor = 120
    # smoothing_factor_1 = 150 # [*100*, 150].
    # smoothing_factor_2 = 120 # [*100*, 150].
    
    # Joint torques
    J0 = [0, 4, 9, 12, 16, 20] # J0 torques.
    J1 = [1, 5, 8, 13, 17, 21] # J1 torques.
    
    for i in range(2): # Horizontal and vertical joints.
        if(i == 0):
            J = J0
            joint_type = "J0"
        else:
            J = J1
            joint_type = "J1"

        fig1, ax = plt.subplots(3, 1, figsize=(260,195))
        # fig3.tight_layout()
        plt.gcf().set_size_inches(12, 9)
        plt.rcParams.update({'font.size': 14})
        plt.subplots_adjust(left=0.09, right=0.91, top=0.98, bottom=0.06)

        for j in range(len(J)//2): # For each leg pair.
            for k in range(2): # Left and right legs.
                if(k == 0):
                    marker = '-'
                else:
                    marker = '--'

                jj = 2*j + k

                action = actions[:,J[jj]]
            
                action = action - np.mean(action)
                action = action / np.std(action)
            
                smoothed_action = interpolate.UnivariateSpline(t[:-1], action, s=smoothing_factor)
                # smoothed_action.set_smoothing_factor(150)
                # action = pd.Series(action).rolling(window=window).mean().to_numpy()
            
                ax[j].plot(t, smoothed_action(t), marker, lw=1, c=colormap(j))
                ax[j].scatter(t, smoothed_action(t), c='k', s=2, alpha=0.6)

                ax[j].set_ylim(-1.5, 1.5)
                ax[j].set_ylim(-1.5, 1.5)
                ax[j].set_ylim(-1.5, 1.5)
            
            ax[j].set_xlim(t[0], t_max) # t[0], t_max
            ax[j].set_xlim(t[0], t_max)
            ax[j].set_xlim(t[0], t_max)

            ax[j].set_ylabel('Joint torque')
            ax[j].set_ylabel('Joint torque')
            ax[j].set_ylabel('Joint torque')
        
        ax[2].set_xlabel('Time')

        plt.savefig(os.path.join(save_dir,f"joint_torques_{joint_type}_ep={ep}.png"), bbox_inches='tight')
        plt.close(fig1)


    # Make a bar plot of mean joint torques
    print("actions:", actions.shape)
    
    actions_reshaped = actions.reshape(-1, 6, 4) # Reshape to (T, 6 legs, 4 joints)

    # Define contralateral leg pairs: (L1, R1), (L2, R2), (L3, R3)
    leg_pairs = [(0, 1), (2, 3), (4, 5)]
    mean_actions = []
    for left_idx, right_idx in leg_pairs:
        # mean across the two legs for each joint, then time-average
        joint_pair_mean = np.abs(actions_reshaped[:, [left_idx, right_idx], :]).mean(axis=1)
        mean_over_time = joint_pair_mean.mean(axis=0)
        mean_actions.append(mean_over_time)
    mean_actions = np.array(mean_actions) # shape (3, 4)

    # Plot
    leg_pair_labels = ['Fore-legs', 'Middle-legs', 'Hind-legs']
    joint_labels = [f'Joint {i}' for i in range(4)]

    fig2, ax2 = plt.subplots()
    bottom = np.zeros(len(leg_pair_labels))
    for j in range(mean_actions.shape[1]):
        ax2.bar(leg_pair_labels, mean_actions[:, j], bottom=bottom, label=joint_labels[j])
        bottom += mean_actions[:, j]

    # ax2.set_xlabel('Leg Pair')
    ax2.set_ylabel('Mean Joint Torque (normalized)')
    ax2.legend(title='Joint Index')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir,f"mean_joint_torques_ep={ep}.png"), bbox_inches='tight')
    plt.close(fig2)