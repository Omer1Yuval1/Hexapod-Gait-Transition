import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_joint_torque_stats(P, all_actions, save_dir, n):
    
    '''
        Make a stacked bar plot of mean joint torques
    '''
    
    print("all_actions:", all_actions.shape)
    leg_pairs = [(0, 1), (2, 3), (4, 5)]

    episode_means  = []  # list to collect each episode’s (3,4) array.
    for actions in all_actions:
        
        actions_reshaped = actions.reshape(-1, 6, 4) # Reshape to (T, 6 legs, 4 joints).

        time_mean = np.abs(actions_reshaped).mean(axis=0)

        # Define contralateral leg pairs: (L1, R1), (L2, R2), (L3, R3)
        mean_actions = []
        for left_idx, right_idx in leg_pairs:
            joint_pair_mean = np.abs(time_mean[[left_idx, right_idx], :]).mean(axis=0) # mean across the two legs for each joint.
            mean_actions.append(joint_pair_mean)
        
        episode_means.append(mean_actions)

    mean_all_episodes = np.mean(episode_means , axis=0)
    std_all_episodes = np.std(episode_means , axis=0)


    # Plot
    leg_pair_labels = ['Fore-legs', 'Middle-legs', 'Hind-legs']
    joint_labels = [f'Joint {i}' for i in range(4)]

    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout()
    plt.gcf().set_size_inches(12, 9)
    plt.rcParams.update({'font.size': 14})

    bottom = np.zeros(mean_all_episodes.shape[0])

    for j in range(mean_all_episodes.shape[1]): # For each joint.
        ax.bar(
            leg_pair_labels,
            mean_all_episodes[:, j],
            bottom=bottom,
            label=joint_labels[j],
            yerr=std_all_episodes[:, j],
            capsize=5
        )
        bottom += mean_all_episodes[:, j]

    ax.set_ylabel('Mean Joint Torque (normalized)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.legend(title='Joint Index')
    plt.tight_layout()
    
    abs_all_actions = np.abs(all_actions)

    episode_means = abs_all_actions.mean(axis=(1, 2))

    mean = episode_means.mean()
    std = episode_means.std()

    plt.savefig(os.path.join(save_dir,f"mean_joint_torques_n={n}_mean={round(mean, 3)}_std={round(std, 3)}.png"), bbox_inches='tight')
    plt.close(fig)


    col_name = "torque"
    new_col = pd.Series(episode_means, name=col_name)

    csv_path = os.path.join(save_dir, "stats.csv")

    if os.path.exists(csv_path):
        # Load existing CSV
        df = pd.read_csv(csv_path)
        # Add or overwrite the column
        df[col_name] = new_col
    else:
        # Create new DataFrame with just this column
        df = pd.DataFrame({col_name: new_col})

    # Write back (overwrites or creates)
    df.to_csv(csv_path, index=False)