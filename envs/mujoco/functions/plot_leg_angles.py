import numpy as np
from scipy.stats import circmean, circstd
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert
from scipy.signal import correlate
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

from envs.mujoco.functions.apply_pca_and_plot_limit_cycle import apply_pca_and_plot_limit_cycle


def plot_leg_angles(dt_opt, t, multi_gait, save_path, plot, name, leg_angles, leg_tip_x_positions, qpos_horizontal, qpos_vertical, leg_angles_mean, leg_angles_std, qpos_horizontal_mean, qpos_horizontal_std, qpos_vertical_mean, qpos_vertical_std):

    win_size = int(0.05 / dt_opt) # seconds -> frames.

    leg_phases_tip_position = []
    leg_tip_x_positions_normalized = []

    if(plot):
        cmap = plt.get_cmap('tab10')  # Replace 'tab10' with any other colormap (e.g., 'viridis', 'plasma')
        # colors = [cmap(i / leg_angles.shape[1]) for i in range(leg_angles.shape[1])]
        color_indices = [0, 4, 1, 2, 8, 9]
        colors = [cmap(i / 9) for i in color_indices]
        colors_pairs = [cmap(0), cmap(3), cmap(2)]
    
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        fig1, ax1 = plt.subplots(2, 6, subplot_kw={'projection': 'polar'}, figsize=(26, 12))
        fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'}) # Leg angles over time (one episode, polar).
        
        fig3, ax3 = plt.subplots(3, 1, figsize=(26, 12)) # Leg tip positions over time (one episode).
        fig4, ax4 = plt.subplots(1, 1, figsize=(16, 4)) # Phase lag of leg tip positions over time - cartesian plot.
        fig5, ax5 = plt.subplots(subplot_kw={'projection': 'polar'}) # Phase lag of leg tip positions over time - cartesian plot.

        # fig4, ax4 = plt.subplots(3, 1, figsize=(26, 12)) # Leg angles (cartesian).
        # fig4, ax4 = plt.subplots(3, 1, figsize=(26, 12)) # Angular velocity as a function of joint angle.

    for i in range(leg_angles.shape[1]): # For each leg.

        if(i % 2 == 0):
            side = 0
        else:
            side = 1

        """ Normalize and smooth leg_tip_x_positions """
        x = leg_tip_x_positions[:, i] - np.mean(leg_tip_x_positions[:, i])
        x = uniform_filter1d(x, size=win_size, axis=0, mode='nearest')
        leg_tip_x_positions_normalized.append(x) # Centered and smoothed.
        
        if(plot):
            ax.scatter(leg_angles[:,i], np.ones_like(leg_angles[:,i]), color=colors[i], s=1, label='Angles') # Plot the data points.
            ax.plot([leg_angles_mean[i], leg_angles_mean[i]], [0, 1], color=colors[i], label='Circular Mean') # Plot the circular mean.
            theta = np.linspace(leg_angles_mean[i] - leg_angles_std[i], leg_angles_mean[i] + leg_angles_std[i], 100) # Highlight the region of circular standard deviation.
            ax.fill_between(theta, 0, 1, color=colors[i], alpha=0.2, label='±1 Circular Std Dev')
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax.set_xticklabels(['0', '90', '180', '270'])
            # ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=20) # Set font size for angle ticks.
            ax.tick_params(axis='y', labelsize=20) # Set font size for radius ticks.
            
            ax2.plot(leg_angles[:,i], t, color=colors[i]) # Plot leg angles over time.
            if(not multi_gait):
                ax2.set_ylim([0, 1])
                ax2.set_yticks([1])
            ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax2.set_xticklabels(['0', '90', '180', '270'])
            ax2.tick_params(axis='x', labelsize=20)  # Set font size for angle ticks.
            ax2.tick_params(axis='y', labelsize=20)  # Set font size for radius ticks.
            
            """ ax4.plot(leg_angles[:,i], t, color=colors[i]) # Plot leg angles over time.
            ax4.set_ylim([0, 1])
            ax4.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax4.set_xticklabels(['0', '90', '180', '270'])
            ax4.set_yticks([1])
            ax4.tick_params(axis='x', labelsize=20)  # Set font size for angle ticks.
            ax4.tick_params(axis='y', labelsize=20)  # Set font size for radius ticks. """

            ax1[0,i].scatter(qpos_horizontal[:,i], np.ones_like(qpos_horizontal[:,i]), color=colors[i], s=1, label='Angles') # Plot the data points.
            ax1[0,i].plot([qpos_horizontal_mean[i], qpos_horizontal_mean[i]], [0, 1], color=colors[i], label='Circular Mean') # Plot the circular mean.
            theta = np.linspace(qpos_horizontal_mean[i] - qpos_horizontal_std[i], qpos_horizontal_mean[i] + qpos_horizontal_std[i], 100) # Highlight the region of circular standard deviation.
            ax1[0,i].fill_between(theta, 0, 1, color=colors[i], alpha=0.2, label='±1 Circular Std Dev')
            ax1[0,i].set_xticks([])
            # ax1[0,i].set_xticklabels(['0', '90', '180', '270'])
            ax1[0,i].set_yticks([])
            ax1[0,i].tick_params(axis='x', labelsize=20)  # Set font size for angle ticks.
            ax1[0,i].tick_params(axis='y', labelsize=20)  # Set font size for radius ticks.

            ax1[1,i].scatter(qpos_vertical[:,i], np.ones_like(qpos_vertical[:,i]), color=colors[i], s=1, label='Angles') # Plot the data points.
            ax1[1,i].plot([qpos_vertical_mean[i], qpos_vertical_mean[i]], [0, 1], color=colors[i], label='Circular Mean') # Plot the circular mean.
            theta = np.linspace(qpos_vertical_mean[i] - qpos_vertical_std[i], qpos_vertical_mean[i] + qpos_vertical_std[i], 100) # Highlight the region of circular standard deviation.
            ax1[1,i].fill_between(theta, 0, 1, color=colors[i], alpha=0.2, label='±1 Circular Std Dev')
            ax1[1,i].set_xticks([])
            # ax1[1,i].set_xticklabels(['0', '90', '180', '270'])
            ax1[1,i].set_yticks([])
            ax1[1,i].tick_params(axis='x', labelsize=20)  # Set font size for angle ticks.
            ax1[1,i].tick_params(axis='y', labelsize=20)  # Set font size for radius ticks.

            """ Plot leg_tip_x_positions """
            ii = int(np.floor((i)/2))

            ax3[ii].plot(t, leg_tip_x_positions_normalized[-1], '-'*(1+side), c=colors_pairs[ii], label=f'Leg {i+1}')
            ax3[ii].set_xlim([0, t[-1]+dt_opt])
            ax3[ii].set_yticks([])
            ax3[ii].tick_params(axis='both', labelsize=40)
            if(ii < 2):
                ax3[ii].set_xticks([])

        """ phase_lag_of_leg_tip_x_positions """
        if(i % 2 == 0):
            sig1 = leg_tip_x_positions[:, i] - np.mean(leg_tip_x_positions[:, i])
            sig2 = leg_tip_x_positions[:, i+1] - np.mean(leg_tip_x_positions[:, i+1])

            # Smooth
            sig1 = uniform_filter1d(sig1, size=win_size, axis=0, mode='nearest')
            sig2 = uniform_filter1d(sig2, size=win_size, axis=0, mode='nearest')
            
            period = 0.25 # [s]. todo: get from input argument.
            window_sec = period
            fs = int(1 / dt_opt)
            # phase_diff = compute_hilbert_phase_difference(sig1, sig2)
            phase_diff = compute_sliding_phase_simple(sig1, sig2, fs=fs, window_sec=window_sec, period_sec=period)
            
            # Smooth phases
            phase_diff = uniform_filter1d(phase_diff, size=4*win_size, axis=0, mode='nearest')

            leg_phases_tip_position.append(phase_diff)

            if(plot):
                win_samples = int(window_sec * fs)
                n_windows = len(t) - win_samples
                t_centers = (t[:n_windows] + t[win_samples - 1 : win_samples - 1 + n_windows]) / 2 # t_centers = (t[:n_windows] + t[win_samples - 1:]) / 2
                
                ax4.plot(t_centers, np.abs(phase_diff), c=colors_pairs[ii])
                
                ax5.plot(np.abs(phase_diff), t_centers, c=colors_pairs[ii])

    if(plot):
        ax4.set_xlim([0, t[-1]+dt_opt])
        ax4.set_yticks([0, np.pi/2, np.pi])
        ax4.set_xlabel('Time [s]', fontsize=32)
        ax4.set_ylabel('Interleg phases [°]', fontsize=32)
        ax4.set_yticklabels(['0', '90', '180'])
        ax4.tick_params(axis='both', labelsize=32)

        ax5.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax5.set_xticklabels(['0', '90', '180', '270'])

        fig.savefig(os.path.join(save_path, f"leg_angles_{name}.png"), bbox_inches='tight')
        fig2.savefig(os.path.join(save_path, f"leg_angles_over_time_{name}.png"), bbox_inches='tight')
        fig1.savefig(os.path.join(save_path, f"horizontal+vertical_joint_angles_{name}.png"), bbox_inches='tight')
        
        fig3.savefig(os.path.join(save_path, f"leg_tip_x_positions_{name}_win={win_size}.png"), bbox_inches='tight')
        fig4.savefig(os.path.join(save_path, f"phase_lag_of_leg_tip_x_positions_{name}_win={win_size}.png"), bbox_inches='tight')
        fig5.savefig(os.path.join(save_path, f"phase_lag_of_leg_tip_x_positions_polar_{name}_win={win_size}.png"), bbox_inches='tight')

        plt.close(fig)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)
    
    """ leg_angles_pairwise """
    if(plot):
        colormap = plt.get_cmap('Set1')    
        # Swap the order of the first two colors
        modified_colors = list(colormap.colors)
        modified_colors[0], modified_colors[1] = modified_colors[1], modified_colors[0]
        colormap = ListedColormap(modified_colors, name='modified_Set1')

        fig1 = plt.figure(figsize=(260,195))
        
        fig1.tight_layout()
        plt.gcf().set_size_inches(12, 9)
        plt.rcParams.update({'font.size': 14})
        
        axes_1 = [fig1.add_subplot(311 + j) for j in range(3)]
        
        plt.subplots_adjust(left=0.09, right=0.91, top=0.98, bottom=0.06)
        
        for j, ax in enumerate(axes_1):

            x1 = leg_angles[:, 2*j]
            x2 = leg_angles[:, 2*j+1]
            
            x1 = np.unwrap((x1 - circmean(x1)) % (2 * np.pi))
            x2 = np.unwrap((x2 - circmean(x2)) % (2 * np.pi))

            if circmean(x1) > np.pi:
                x1 -= 2 * np.pi
            if circmean(x2) > np.pi:
                x2 -= 2 * np.pi

            ax.plot(t, x1*180/np.pi, '-', c=colormap(j))
            # ax.scatter(t, leg_angles[:,2 * j], color='k', s=5, alpha=0.7)
            ax.plot(t, x2*180/np.pi, '-', c=colormap(j))
            
            # ax.set_xlim(t[0], t_max)
            # ax.set_yticks(np.arange(-A[2 * j] + offsets[2 * j], A[2 * j] + offsets[2 * j] + A[2 * j] / 2, A[2 * j] / 2))
            # ax.set_ylim(-A[2 * j] + offsets[2 * j], A[2 * j] + offsets[2 * j])
            ax.set_ylabel('Leg angle [°]')
            # print(-A[2 * j] + offsets[2 * j] , A[2 * j] + offsets[2 * j], np.arange(-A[2 * j] + offsets[2 * j], A[2 * j] + offsets[2 * j] + A[2 * j] / 2, A[2 * j] / 2))
        
        axes_1[-1].set_xlabel('Time')

        plt.savefig(os.path.join(save_path, f"leg_angles_pairwise_{name}.png"), bbox_inches='tight')
        plt.close(fig1)


        # Plot limit cycle of leg tip x poisitions for each leg pair (centered and smoothed)
        # ----------------------------------------------------------------------------------
        fig2, axes_2 = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), subplot_kw={'aspect': 'equal'})
        fig2.tight_layout()
        plt.gcf().set_size_inches(12, 9)
        plt.rcParams.update({'font.size': 14})

        for i, ax in enumerate(axes_2): # For each leg pair.
            x = leg_tip_x_positions_normalized[2 * i]
            y = leg_tip_x_positions_normalized[2 * i+1]
            
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
        
        plt.savefig(os.path.join(save_path, f"limit_cycle_of_leg_tip_x_position_pairwise_{name}.png"), bbox_inches='tight')
        plt.close(fig2)
        
        # Plot limit cycle of leg tip x poisitions for all legs (centered and smoothed)
        apply_pca_and_plot_limit_cycle(np.array(leg_tip_x_positions_normalized).T, dt_opt, save_path, f"limit_cycle_of_PCA_of_leg_tip_x_position_{name}")
        
        # Plot limit cycle of leg phases (based on leg tip x poisitions)
        apply_pca_and_plot_limit_cycle(np.array(leg_phases_tip_position).T, dt_opt, save_path, f"limit_cycle_of_PCA_of_phases_of_leg_tip_x_position_{name}")
        

    return leg_phases_tip_position, np.array(leg_tip_x_positions_normalized).T



def compute_hilbert_phase_difference(sig1, sig2):
    """
    Computes the instantaneous phase difference between two signals.
    
    Args:
        sig1 (np.array): Reference signal.
        sig2 (np.array): Comparison signal.
        
    Returns:
        np.array: Phase difference in radians over time [-pi, pi].
    """
    # 1. Compute analytic signals
    analytic1 = hilbert(sig1)
    analytic2 = hilbert(sig2)
    
    # 2. Extract instantaneous phases
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    
    # 3. Compute difference
    # We use exp(i*(phi1 - phi2)) to compute the difference on the complex 
    # plane, then take the angle. This handles the wrapping automatically.
    phase_diff = np.angle(np.exp(1j * (phase1 - phase2)))
    
    return phase_diff

def compute_sliding_phase_simple(sig1, sig2, fs, window_sec, period_sec):
    """
    Computes time-dependent phase shift [0, pi] using a sliding window cross-correlation.
    
    Args:
        sig1, sig2: Input signal arrays.
        fs: Sampling frequency (Hz).
        window_sec: Duration of the sliding window (seconds).
        period_sec: The period of the oscillation (seconds) used to convert time lag to phase.
        
    Returns:
        np.array: Phase difference curve in radians [0, pi].
    """
    win_samples = int(window_sec * fs)
    phases = []
    
    # Iterate with a sliding window (stride=1 for max resolution)
    for i in range(len(sig1) - win_samples):
        # 1. Extract window
        x = sig1[i : i + win_samples]
        y = sig2[i : i + win_samples]
        
        # 2. Compute Cross-Correlation
        # 'full' mode allows us to find the lag even if it's large
        cc = correlate(x, y, mode='full')
        
        # 3. Find Lag
        # The index of the max value corresponds to the optimal shift
        shift_idx = np.argmax(cc)
        
        # Convert index to centered lag (0 is center)
        lag_samples = shift_idx - (len(x) - 1)
        
        # 4. Convert Lag to Phase (Radians)
        # Ratio of lag to period * 2pi
        lag_sec = lag_samples / fs
        phi = (lag_sec / period_sec) * 2 * np.pi
        
        phases.append(phi)

    phases = np.array(phases)

    # 5. Wrap to [0, pi]
    # Use complex exponential to wrap to [-pi, pi], then take abs for [0, pi]
    return np.abs(np.angle(np.exp(1j * phases)))