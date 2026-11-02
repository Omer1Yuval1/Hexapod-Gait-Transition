import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def apply_pca_and_plot_limit_cycle(x, dt, save_path, name, index_range=None):
    
    """ x must be either a 2d numpy array in the form of [n_samples, n_features] or a list of such 2d arrays (containing multiple time series) """

    print(f"Apply PCA and plotting a limit cycle for {name}")

    font_size = 34

    save_path = os.path.join(save_path, "limit_cycle")
    os.makedirs(save_path, exist_ok=True)

    # Standardize input to always be a list of datasets for seamless looping
    if not isinstance(x, list):
        multi_dataset = False
        datasets = [x]
    else:
        multi_dataset = True
        datasets = x

    global_data = np.concatenate(datasets, axis=0) # Concatenate purely to calculate the global space parameters.

    # print("global_data = ", global_data.shape)
    
    # Fit the global Scaler and PCA models
    scaler = StandardScaler()
    scaler.fit(global_data)

    pca = PCA(n_components=min(5, global_data.shape[1]))
    pca.fit(scaler.transform(global_data))

    # Print the variance captured by each individual component
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC {i+1}: {var*100:.2f}% variance captured")

    plt.rcParams.update({'font.size': font_size})

    # Initialize 2D Figure
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw={'aspect': 'equal'})
    fig1.tight_layout()

    # Initialize 3D Figure
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw={'projection': '3d'})
    fig2.tight_layout()

    # Loop through each individual dataset, transform, and plot

    if index_range is not None:
        idx_gait1_end = index_range[0]
        idx_gait2_start = index_range[1]

        # Lists to accumulate the 3D points across all datasets
        gait1_points_list = []
        gait2_points_list = []
    
    all_pc1_list = [] # Accumulator for the PC1 heatmap.
    all_pc2_list = [] # Accumulator for the PC2 heatmap.
    all_pc3_list = [] # Accumulator for the PC3 heatmap.


    for dataset in datasets:
        # Transform this specific dataset into the shared global space
        scaled_dataset = scaler.transform(dataset)
        principal_components = pca.transform(scaled_dataset)

        if index_range is not None:
            # Accumulate points for ellipse fitting (using all 3 components)
            gait1_points_list.append(principal_components[:idx_gait1_end, :3])
            gait2_points_list.append(principal_components[idx_gait2_start:, :3])

        """ if index_range is not None:
            start_idx, end_idx = index_range
            pc1 = principal_components[start_idx:end_idx, 0]
            pc2 = principal_components[start_idx:end_idx, 1]
            pc3 = principal_components[:, 2]
            t = np.arange(start_idx, start_idx + len(pc1)) 
        else:
            pc1 = principal_components[:, 0]
            pc2 = principal_components[:, 1]
            pc3 = principal_components[:, 2]
            t = np.arange(len(pc1)) """
        pc1 = principal_components[:, 0]
        pc2 = principal_components[:, 1]
        pc3 = principal_components[:, 2]
        t = np.arange(len(pc1))

        all_pc1_list.append(pc1)
        all_pc2_list.append(pc2)
        all_pc3_list.append(pc3)

        # --- 2D Plotting (ax1) ---
        points_2d = np.array([pc1, pc2]).T.reshape(-1, 1, 2)
        segments_2d = np.concatenate([points_2d[:-1], points_2d[1:]], axis=1)
        lc_2d = LineCollection(segments_2d, cmap='turbo') 
        lc_2d.set_array(t[:-1])
        lc_2d.set_linewidth(2)
        lc_2d.set_alpha(0.7) 
        ax1.add_collection(lc_2d)

        # --- 3D Plotting (ax2) ---
        points_3d = np.array([pc1, pc2, pc3]).T.reshape(-1, 1, 3)
        segments_3d = np.concatenate([points_3d[:-1], points_3d[1:]], axis=1)
        lc_3d = Line3DCollection(segments_3d, cmap='turbo') 
        lc_3d.set_array(t[:-1])
        lc_3d.set_linewidth(2)
        lc_3d.set_alpha(0.7) 
        ax2.add_collection(lc_3d)

    ax1.autoscale() # Auto-scale the 2D axis.

    # Calculate global limits for the 3D axis
    global_pca = pca.transform(scaler.transform(global_data))
    ax2.set_xlim(global_pca[:, 0].min(), global_pca[:, 0].max())
    ax2.set_ylim(global_pca[:, 1].min(), global_pca[:, 1].max())
    ax2.set_zlim(global_pca[:, 2].min(), global_pca[:, 2].max())
    
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_zlabel('PC 3')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    elevation, azimuth, roll = ax2.elev, ax2.azim, ax2.roll # Get the current orientation from ax2, then use it to apply the same orientation to the ellipse plot.
    
    # Save both figures with distinct names
    fig1.savefig(os.path.join(save_path, f"{name}_2d.png"), bbox_inches='tight')
    fig2.savefig(os.path.join(save_path, f"{name}_3d.png"), bbox_inches='tight')
    
    """ if isinstance(x, list):
        plt.show() """

    # Close figures to free memory
    plt.close(fig1)
    plt.close(fig2)

    if multi_dataset:

        for index, all_pc_list in enumerate([all_pc1_list, all_pc2_list, all_pc3_list]):
            all_pc_matrix = np.array(all_pc_list)
            num_episodes, num_frames = all_pc_matrix.shape
        
            time_1d = np.arange(num_frames) * dt
            time_matrix = np.tile(time_1d, (num_episodes, 1))

            # Flatten matrices to feed into the 2D histogram
            time_flat = time_matrix.flatten()
            pc_flat = all_pc_matrix.flatten()

            # Initialize Heatmap Figure
            fig_heat, ax_heat = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
            fig_heat.tight_layout(pad=3.0)
            plt.rcParams.update({'font.size': font_size})

            # Plot 2D Histogram
            # Using num_frames for X-bins prevents temporal blurring across discrete time steps
            # Using 50 bins for Y provides high vertical resolution for PC1 values
            h = ax_heat.hist2d(time_flat, pc_flat, bins=[num_frames//4, 25], cmap='turbo', density=True)
            cbar = fig_heat.colorbar(h[3], ax=ax_heat, label='Density')
            cbar.set_ticks([])

            ax_heat.set_xlabel('Time (s)')
            ax_heat.set_ylabel(f'PC {index+1}')
            ax_heat.set_xticks([])
            ax_heat.set_yticks([])
            # ax_heat.set_title('Density of PC 1 Trajectories Over Time')

            fig_heat.savefig(os.path.join(save_path, f"{name}_pc{index+1}_heatmap.png"), bbox_inches='tight')
            plt.close(fig_heat)

    if index_range is not None:
        # --- Fit and Plot the 3D Ellipses ---
        # Combine the points from all episodes into massive point clouds
        all_gait1_points = np.vstack(gait1_points_list)
        all_gait2_points = np.vstack(gait2_points_list)

        # Calculate the parametric 3D coordinates for both ellipses
        ellipse1 = get_3d_ellipse(all_gait1_points, num_std=1.5)
        ellipse2 = get_3d_ellipse(all_gait2_points, num_std=1.5)

        fig_ellipse, ax_ellipse = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw={'projection': '3d'})
        fig_ellipse.tight_layout()
        plt.rcParams.update({'font.size': font_size})

        ax_ellipse.plot(ellipse1[:, 0], ellipse1[:, 1], ellipse1[:, 2], color='cyan', linewidth=4) # , label='Gait 1 Limit Cycle'
        ax_ellipse.plot(ellipse2[:, 0], ellipse2[:, 1], ellipse2[:, 2], color='magenta', linewidth=4) # , label='Gait 2 Limit Cycle'
        
        ax_ellipse.legend()

        # Apply identical global limits to the new ellipse figure
        ax_ellipse.set_xlim(global_pca[:, 0].min(), global_pca[:, 0].max())
        ax_ellipse.set_ylim(global_pca[:, 1].min(), global_pca[:, 1].max())
        ax_ellipse.set_zlim(global_pca[:, 2].min(), global_pca[:, 2].max())
        ax_ellipse.set_xlabel('PC 1')
        ax_ellipse.set_ylabel('PC 2')
        ax_ellipse.set_zlabel('PC 3')
        ax_ellipse.set_xticklabels([])
        ax_ellipse.set_yticklabels([])
        ax_ellipse.set_zticklabels([])

        # ax_ellipse.view_init(elev=20, azim=38)
        ax_ellipse.view_init(elev=elevation, azim=azimuth, roll=roll) # Apply the orientation of the 3d PCA plot of all datasets.

        fig_ellipse.savefig(os.path.join(save_path, f"{name}_3d_ellipse.png"), bbox_inches='tight')
        # plt.show()
        plt.close(fig_ellipse)

def get_3d_ellipse(points, num_std=2.0, num_points=100):
    """ Fits a 3D ellipse to a cloud of points using SVD. """
    # 1. Find the center of the limit cycle
    center = np.mean(points, axis=0)
    centered_points = points - center

    # 2. Extract the principal axes of the spatial cloud
    U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
    
    # The first two rows of Vt define the 2D plane of the limit cycle
    major_axis = Vt[0]
    minor_axis = Vt[1]

    # 3. Calculate the lengths of the semi-axes
    std_devs = S / np.sqrt(len(points))
    a = num_std * std_devs[0]
    b = num_std * std_devs[1]

    # 4. Generate the parametric 3D coordinates of the ellipse
    theta = np.linspace(0, 2 * np.pi, num_points)
    ellipse_3d = center + np.outer(np.cos(theta), a * major_axis) + np.outer(np.sin(theta), b * minor_axis)

    return ellipse_3d