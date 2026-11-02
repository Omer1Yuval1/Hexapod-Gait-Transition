import numpy as np
from scipy.stats import circmean # , circstd
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_leg_angles_stats(leg_angles_means, leg_angles_SDs, save_path):
        
    cmap = get_cmap('tab10')  # Replace 'tab10' with any other colormap (e.g., 'viridis', 'plasma')
    # colors = [cmap(i / leg_angles.shape[1]) for i in range(leg_angles.shape[1])]
    color_indices = [0, 4, 1, 2, 8, 9]
    colors = [cmap(i / 9) for i in color_indices]
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for i in range(leg_angles_means.shape[1]): # For each leg.
        
        mean_all_episodes = circmean(leg_angles_means[:,i], high=np.pi*2, low=0)
        SD_all_episodes = np.mean(leg_angles_SDs[:,i])
        
        # ax.scatter(leg_angles_means[:,i], np.ones_like(leg_angles_means[:,i]), color=colors[i], s=1, label='Angles') # Plot the data points.
        ax.plot([mean_all_episodes, mean_all_episodes], [0, 1], color=colors[i], label='Circular Mean') # Plot the circular mean.
        theta = np.linspace(mean_all_episodes - SD_all_episodes, mean_all_episodes + SD_all_episodes, 100) # Highlight the region of circular standard deviation.
        ax.fill_between(theta, 0, 1, color=colors[i], alpha=0.2, label='±1 Circular Std Dev')
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(['0', '90', '180', '270'])
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=20)  # Set font size for angle ticks.
        ax.tick_params(axis='y', labelsize=20)  # Set font size for radius ticks.


    fig.savefig(os.path.join(save_path, f"leg_angles_n={leg_angles_means.shape[0]}.png"), bbox_inches='tight')
    plt.close(fig)