import numpy as np
# import math
import pandas as pd
# from scipy import interpolate
import os
# import pickle
import time
from collections import defaultdict

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
from matplotlib.colors import ListedColormap

# from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy import stats

import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
import mujoco
# from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from output.insect.plot_joint_angles import plot_joint_angles
from output.insect.plot_leg_angles_stats import plot_leg_angles_stats
from output.insect.plot_joint_torques import plot_joint_torques
from output.insect.plot_joint_torque_stats import plot_joint_torque_stats
from output.insect.plot_contact_forces import plot_contact_forces
from output.insect.plot_leg_trajectories import plot_leg_trajectories
from output.insect.plot_rewards import plot_rewards
from output.get_com_speed import get_com_speed
from envs.mujoco.functions.get_leg_angles_all import get_leg_angles_all

from envs.mujoco.functions.apply_pca_and_plot_limit_cycle import apply_pca_and_plot_limit_cycle

from functions.generate_mujoco_xml_insect import generate_mujoco_xml_insect

def save_agent_video(model=None, animal=None, P=None, model_path=None, model_file=None, ep=0):
    
    save_stats = True
    test = False
    reset_test = False
    
    if(save_stats):
        n = 20 # 200.
    else:
        n = 1
    
    if(P is None):
        from parameters.parameters_main import parameters_main
        P = parameters_main(animal)
    else: # Using this temporarily to include sites.
        from parameters.parameters_main import parameters_main
        P_temp = parameters_main(P["animal"])
        P["model"]["joints"] = P_temp["model"]["joints"]

        P["gaits"][0]["params"]["gait_transition"]["offset"] = 0
        P["gaits"][1]["params"]["gait_transition"]["offset"] = 0

    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    mujoco_model_path = os.path.join(main_dir, 'envs', 'mujoco', 'assets', P["physical_model"]["file_name"] + ".xml")
    
    dt = P["dt_opt"]

    if(test):
        t_max = 16 * P["gaits"][0]["period"] # [s].
        n = 1
    else:
        t_max = 16 * P["gaits"][0]["period"] # [s].
    
    steps_per_episode = int(t_max / dt - 1)
    # print("Steps per episode for video recording:", steps_per_episode)
    
    # t = np.arange(0, t_max, dt) # t = np.arange(0,qpos.shape[0]*dt,dt)
    t = np.arange(0, t_max, dt) # t = np.arange(0,qpos.shape[0]*dt,dt)
    
    # print(steps_per_episode, t.shape)
    
    speed = np.zeros((steps_per_episode, n))

    win_samples = int(P["gaits"][0]["period"] * int(1 / P["dt_opt"]))
    n_windows = len(t) - win_samples
    t_window_centers = (t[:n_windows] + t[win_samples - 1 : win_samples - 1 + n_windows]) / 2 # t_centers = (t[:n_windows] + t[win_samples - 1:]) / 2

    leg_phases_tip_position_all = np.zeros((n, 3, len(t) - win_samples)) # One column for each leg pair.
    # print(win_size, leg_phases_tip_position_all.shape)
    
    leg_angles_means = np.zeros((n, 6))
    leg_angles_SDs = np.zeros((n, 6))
    leg_tip_x_positions_normalized_all = []
    optimized_joint_angles_all = []
    all_actions = []
    # initial_global_joint_angles = np.zeros((n, 6))

    stance_log = defaultdict(list) # Structure: { leg_id: [[t1, d1], [t2, d2], ...], ... }, where t is mid-stance time and d is stance duration.
    
    model_path += '/data/'
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, "limit_cycle"), exist_ok=True)
    
    gaits = ("forward" if P["gaits"][0]["direction"] == 1 else "backward") + "_" +  str(P['gaits'][0]['phase'])
    if(len(P["gaits"]) == 2):
        gaits += "-" + ("forward" if P["gaits"][1]["direction"] == 1 else "backward") + "_" + str(P['gaits'][1]['phase'])
    
    if(P["gaits"][0]["direction"] == 1):
        cameras = ["worldview_0", "robot_view_0", "robot_view_1"]
    else:
        cameras = ["worldview_bwd", "robot_view_0", "robot_view_1"]

    if(test or reset_test or save_stats):
        cameras = ["robot_view_1"]
    
    # save_dir = os.path.join(main_dir, 'resources', 'results', P['animal'], model_file)
    
    # Evaluate trained agent and save data
    for i in range(n):
        print(f"Evaluation episode {i}")

        if(i > 0): # This is used to make sure the inner loop runs only once when doing statistics.
            cameras = cameras[0]

        for j in range(len(cameras)):
            
            np.random.seed(i)
            
            generate_mujoco_xml_insect(P=P, camera_mode=j) # This is done here in order to be able to generate videos from different perspectives.
            time.sleep(1)
            
            if(i == 0 and not save_stats and ((not test) or reset_test)):
                print("Recording video...")
                env = gym.make(P["env"], P=P, record=True, xml_file=mujoco_model_path, max_episode_steps=steps_per_episode, render_mode="rgb_array", camera_name=cameras[j]) # P["rendering"]["camera_names"][2]
                
                video_name = f"{P['animal']}_cam{j}"
                # video_name = f"{P['animal']}_{gaits}_{P['gaits'][0]['phase']}_ep={ep}_{i}_cam{j}"
                env = gym.wrappers.RecordVideo(env, video_folder=model_path, name_prefix=video_name)
                obs, info = env.reset() # obs, info = env.reset(seed=i)
                env.render()
                
                """ if(not P["object_mode"]):
                    env.unwrapped.mujoco_renderer.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True """
                env.unwrapped.mujoco_renderer.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            else:
                env = gym.make(P["env"], P=P, record=True, xml_file=mujoco_model_path, max_episode_steps=steps_per_episode)
                obs, info = env.reset() # obs, info = env.reset(seed=i)
            
            # terminated = 0
            # truncated = 0
            actions = np.array([])
            # qvel = np.array([])
            rewards = defaultdict(list)
            # tarsi_collisions = np.array([])
            # geom_xpos = None # np.empty((0, 30, 3))
            # geom_xpos_matrix = np.array([])
            cfrc_ext = [] # np.array([]) # Contact forces.
            # leg_pos = np.array([])
            # qpos_ref = np.array([])
            if(reset_test):
                gait_phase = 0
                ω = 2 * np.pi / P["gaits"][0]["period"]

            # s = 0
            # while(not terminated and not truncated):
            for ii in range(steps_per_episode):
                # print(s)
                # s += 1
                if(reset_test):
                    gait_phase = (gait_phase + ω * P["dt_opt"]) % (2 * np.pi)
                    obs = env.reset_model(gait_phase=gait_phase)
                    action, _ = model.predict(obs)
                    _, _, _, _, _ = env.step(0*action)
                else:
                    action, _states = model.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                
                    actions = np.append(actions, action)
                    cfrc_ext.append(info["cfrc_ext"].copy()) # cfrc_ext = np.append(cfrc_ext, info["cfrc_ext"])

                    for key, value in info["reward"].items():
                        rewards[key].append(value)

            env.close()
            
            if(not reset_test):
                # print(info["history"]["qpos"].shape, info["history"]["tarsi_collisions"].shape)
                qpos = np.array(info["history"]["qpos"])
                qvel = np.array(info["history"]["qvel"])
                # tarsi_collisions = info["history"]["tarsi_collisions"].reshape(-1, 6)
                
                cfrc_ext = np.array(cfrc_ext)
                
                mean_angle, std_angle, _, leg_phases_tip_position, leg_tip_x_positions_normalized = get_leg_angles_all(info["history"]["geom_xpos"], qpos, len(P["objects"]), P=P, t=t, save_path=model_path, plot=True if i==0 else False, name=j)
                # print(leg_tip_x_positions_normalized)
                if(i == 0 and not save_stats):
                    print("Leg angles. Mean=", np.round(mean_angle*180/np.pi, 2), "Circular standard deviation=", np.round(std_angle*180/np.pi, 2))

                if(i == 0):
                    colormap = plt.get_cmap('Set1')    
                    # Swap the order of the first two colors
                    modified_colors = list(colormap.colors)
                    modified_colors[0], modified_colors[1] = modified_colors[1], modified_colors[0]
                    colormap = ListedColormap(modified_colors, name='modified_Set1')

                    """ if(not P["multi_gait"]):
                        plot_joint_angles(P["gaits"], t, t_max, qpos, qvel, actions.reshape(-1, action.shape[0]), colormap, model_path, j) # plot joint angles of leg pairs. """
                    
                    plot_joint_angles(P["gaits"], P["dt_opt"], t, t_max, qpos, qvel, actions.reshape(-1, action.shape[0]), colormap, model_path, j) # plot joint angles of leg pairs.
                    
                    for key in rewards:
                        rewards[key] = np.array(rewards[key])
                    plot_rewards(P, t, rewards, model_path, j)
                    
                    plot_joint_torques(P, t, t_max, actions.reshape(-1, action.shape[0]), colormap, model_path, j)

                
                stance_log_episode = plot_contact_forces(P, t, cfrc_ext, model_path, j, plot_flag=(not save_stats or i==0)) # print(cfrc_ext.shape)

                # Populate the stance_log by collecting (t,d) pairs per leg.
                for leg_idx, events in stance_log_episode.items():
                    stance_log[leg_idx].extend(events)

                if(save_stats):
                    if(j == 0):
                        speed[:,i] = get_com_speed(P["dt_opt"], qpos, t_max)
                        all_actions.append(actions.reshape(-1, action.shape[0]))

                        leg_angles_means[i,:] = mean_angle
                        leg_angles_SDs[i,:] = std_angle
                        leg_phases_tip_position_all[i] = np.array(leg_phases_tip_position)

                        leg_tip_x_positions_normalized_all.append(leg_tip_x_positions_normalized)

                        optimized_joint_angles = qpos[:, np.concatenate([P["gaits"][0]["params"]["horizontal_joints"]["qpos"], P["gaits"][0]["params"]["vertical_joints"]["qpos"]])]
                        optimized_joint_angles_all.append(optimized_joint_angles)
                        
                        plot_leg_trajectories(P, info["history"]["geom_xpos"], info["history"]["site_xpos"], len(P["objects"]), model_path, j, i, plot_flag=i==0)

                    elif(i == 0):
                        plot_leg_trajectories(P, info["history"]["geom_xpos"], info["history"]["site_xpos"], len(P["objects"]), model_path, j, i, plot_flag=True)

    if(save_stats):
        index_range = (int(1.8 / dt), int(2.2 / dt)) # None
        # index_range = None
        apply_pca_and_plot_limit_cycle(leg_tip_x_positions_normalized_all, P["dt_opt"], model_path, f"limit_cycle_of_PCA_of_leg_tip_x_position_n={n}", index_range)
        apply_pca_and_plot_limit_cycle(optimized_joint_angles_all, P["dt_opt"], model_path, f"limit_cycle_of_PCA_of_optimized_joint_angles_n={n}", index_range)
        print(len(leg_tip_x_positions_normalized_all), len(optimized_joint_angles_all))

        
        axis_label_font_size = 34
        axis_ticks_font_size = 30

        cmap = plt.get_cmap('tab10')
        colors_pairs = [cmap(0), cmap(3), cmap(2)]

        plot_leg_angles_stats(leg_angles_means, leg_angles_SDs, model_path)

        plot_joint_torque_stats(P, np.stack(all_actions, axis=0), model_path, n) # Save joint torque statistics.

        # Save speed statistics
        col_name = "speed"
        new_col = pd.Series(np.mean(speed, axis=0), name=col_name)

        csv_path = os.path.join(model_path, "stats.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path) # Load existing CSV.
            df[col_name] = new_col # Add or overwrite the column.
        else:
            df = pd.DataFrame({col_name: new_col}) # Create new DataFrame with just this column.

        # Write back (overwrites or creates)
        df.to_csv(csv_path, index=False)

        # np.savetxt(f'{model_path}/speed.csv', np.mean(speed), delimiter=',') # _{P["gaits"][0]["phase"]}_{direction}
        # np.savetxt(f'{model_path}/initial_global_joint_angles.csv', initial_global_joint_angles, delimiter=',')

        # Plot speed over time across episodes: speed[steps, n_episodes]
        # **************************************************************
        mean_speed = np.mean(speed, axis=1) # Mean across episodes.
        std_speed = np.std(speed, axis=1) # STD scross episodes.
        # time_steps = np.arange(speed.shape[0])

        plt.figure(figsize=(26, 4))

        # 1. Plot individual episodes (Background)
        # Use low alpha (transparency) to visualize density without clutter
        # plt.plot(t[:-1], speed, color='blue', alpha=0.1, linewidth=1)

        # 2. Plot the Mean (Signal)
        plt.plot(t[:-1], mean_speed, color='black', linewidth=2, label='Mean Speed')

        # 3. Add Shaded Region (Uncertainty/Variance)
        plt.fill_between(t[:-1], mean_speed - std_speed, mean_speed + std_speed, color='black', alpha=0.2, label='±1 Std Dev')

        plt.xlabel("Time [s]", fontsize=axis_label_font_size)
        plt.ylabel("Speed [cm/s]", fontsize=axis_label_font_size)
        # plt.title(f"Speed over time (n={n})", fontsize=26)
        plt.tick_params(axis='x', labelsize=axis_ticks_font_size)
        plt.tick_params(axis='y', labelsize=axis_ticks_font_size)
        plt.gca().xaxis.set_major_formatter('{x:g}')
        # plt.legend()
        # plt.grid(True, alpha=0.3)

        plt.xlim(0, t[-1]+P["dt_opt"])

        plt.savefig(f"{model_path}/speed_all_episodes_n={n}.png", bbox_inches='tight')
        plt.close()
        
        # Plot leg phases over time across episodes: speed[n_episodes, leg pair, steps]
        # *****************************************************************************
        leg_phases_tip_position_all = leg_phases_tip_position_all * 180 / np.pi

        mean_phases = np.mean(leg_phases_tip_position_all, axis=0) # Mean across episodes.
        std_phases = np.std(leg_phases_tip_position_all, axis=0) # STD scross episodes.

        # print(leg_phases_tip_position_all.shape)
        # print("mean_phases =", mean_phases)
        # print("leg_phases_tip_position_all =", leg_phases_tip_position_all)

        fig, ax = plt.subplots(3, 1, figsize=(26, 12)) # Leg tip positions over time (one episode).

        for i, axis in enumerate(ax):
            axis.plot(t_window_centers, mean_phases[i,:], color=colors_pairs[i], linewidth=2, label='Mean phase') # Plot the Mean (Signal).

            # Add Shaded Region (Uncertainty/Variance)
            axis.fill_between(t_window_centers, mean_phases[i,:] - std_phases[i,:], mean_phases[i,:] + std_phases[i,:], color=colors_pairs[i], alpha=0.2, label='±1 Std Dev')


            axis.set_ylabel("Phase [°]", fontsize=axis_label_font_size)
            axis.tick_params(axis='x', labelsize=axis_ticks_font_size)
            axis.tick_params(axis='y', labelsize=axis_ticks_font_size)

            axis.set_xlim([0, t[-1]+P["dt_opt"]])
            axis.xaxis.set_major_formatter('{x:g}')
            axis.set_ylim([0, 180])
            axis.set_yticks([0, 90, 180])
        
        ax[-1].set_xlabel("Time [s]", fontsize=axis_label_font_size)

        # ax[0].set_title(f"Leg phases over time (n={n})", fontsize=26)
        """ ax[0].set_title("Forelegs", fontsize=26)
        ax[0].set_title("Middlelegs", fontsize=26)
        ax[0].set_title("Hindlegs", fontsize=26) """
        
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        plt.savefig(f"{model_path}/leg_phases_all_episodes_n={n}.png", bbox_inches='tight')
        plt.close()

        # Plot histograms of Euclidean and DTW distances between the middle and hind legs (per episode)
        # *********************************************************************************************

        # --- Extract Middle and Hind Leg Phases ---
        num_episodes = leg_phases_tip_position_all.shape[0]
        middle_legs = leg_phases_tip_position_all[:, 1, :]
        hind_legs = leg_phases_tip_position_all[:, 2, :]

        # --- Calculate Phase and Time Shifts ---
        # 1. Signed Phase Difference (Preserved per episode)
        phase_shifts_episodes = middle_legs - hind_legs
        phase_shift_means = np.mean(phase_shifts_episodes, axis=1) # Extract 1 mean per episode

        # 2. Signed Time Shift (Preserved per episode)
        dtw_time_shifts_episodes = []
        dtw_time_shift_means = []
        dt = P["dt_opt"]

        for e in range(num_episodes):
            _, path = fastdtw(middle_legs[e], hind_legs[e])
            path_arr = np.array(path)
            shift = (path_arr[:, 0] - path_arr[:, 1]) * dt
            dtw_time_shifts_episodes.append(shift)
            dtw_time_shift_means.append(np.mean(shift)) # Extract 1 mean per episode
            
        dtw_time_shift_means = np.array(dtw_time_shift_means)

        # --- Plot Signed Phase Difference Histogram ---
        # Define common bins and calculate episode-level proportions
        min_p, max_p = np.min(phase_shifts_episodes), np.max(phase_shifts_episodes)
        phase_bins = np.linspace(min_p, max_p, 51)
        phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        
        # Calculate probabilities per episode, then average across episodes
        phase_hists = np.array([np.histogram(ep, bins=phase_bins, density=False)[0] / len(ep) for ep in phase_shifts_episodes])
        phase_hist_mean = np.mean(phase_hists, axis=0)

        fig_phase, ax_phase = plt.subplots(figsize=(10, 8))
        ax_phase.bar(phase_centers, phase_hist_mean, width=np.diff(phase_bins), color="#6F6F6FFF", edgecolor='black', alpha=0.6)

        ax_phase.set_xlabel("Phase shift [°]", fontsize=axis_label_font_size)
        ax_phase.set_ylabel("Normalized Proportion", fontsize=axis_label_font_size)
        ax_phase.tick_params(axis='both', labelsize=axis_ticks_font_size)
        ax_phase.grid(True, alpha=0.3)

        plt.savefig(f"{model_path}/phase_shift_hist_mid_hind_n={n}.png", bbox_inches='tight')
        plt.close(fig_phase)

        # --- Plot DTW Time Shift Histogram ---
        # Define common bins and calculate episode-level proportions
        all_time_shifts = np.concatenate(dtw_time_shifts_episodes)
        min_t, max_t = np.min(all_time_shifts), np.max(all_time_shifts)
        time_bins = np.linspace(min_t, max_t, 51)
        time_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        # Calculate probabilities per episode, then average across episodes
        time_hists = np.array([np.histogram(ep, bins=time_bins, density=False)[0] / len(ep) for ep in dtw_time_shifts_episodes])
        time_hist_mean = np.mean(time_hists, axis=0)

        fig_time, ax_time = plt.subplots(figsize=(10, 8))
        ax_time.bar(time_centers, time_hist_mean, width=np.diff(time_bins), color="#6F6F6FFF", edgecolor='black', alpha=0.6)

        ax_time.set_xlabel("Time shift [s]", fontsize=axis_label_font_size)
        ax_time.set_ylabel("Normalized Proportion", fontsize=axis_label_font_size)
        ax_time.tick_params(axis='both', labelsize=axis_ticks_font_size)
        ax_time.grid(True, alpha=0.3)

        plt.savefig(f"{model_path}/time_shift_hist_mid_hind_n={n}.png", bbox_inches='tight')
        plt.close(fig_time)
        
        # 1-sample t-test testing the 20 independent episode means against a null hypothesis of 0
        t_time, p_time = stats.ttest_1samp(dtw_time_shift_means, 0)
        t_phase, p_phase = stats.ttest_1samp(phase_shift_means, 0)

        print(f"DTW Time Shift Mean: {np.mean(dtw_time_shift_means):.4f} s (p = {p_time:.4f})")
        print(f"Phase Shift Mean: {np.mean(phase_shift_means):.4f} ° (p = {p_phase:.4f})")

        # Plot 2D histograms/heatmaps of stance mid-time and duration (for each leg)
        # **************************************************************************
        bin_size_time = 0.1 # [s] to [frames].
        bin_size_duration = 0.01 # [s] to [frames].

        x_bins = np.arange(0, 4 + bin_size_time, bin_size_time)
        max_duration = 0.3 
        y_bins = np.arange(0, max_duration + bin_size_duration, bin_size_duration)

        num_legs = len(stance_log)
        fig, axes = plt.subplots(1, num_legs, figsize=(5 * num_legs, 5), sharey=True, squeeze=False)

        # Iterate through each leg's stance data
        legs = ["L1", "R1", "L2", "R2", "L3", "R3"]
        for leg_idx in sorted(stance_log.keys()):
            ax = axes[0, leg_idx]
            
            # Convert list of [t, d] to a NumPy array for easy slicing
            data = np.array(stance_log[leg_idx]) * P["dt_opt"]
            
            if len(data) == 0:
                continue # Skip if no stance events were found
            
            # Extract columns: Column 0 is Time (t), Column 1 is Duration (d)
            x = data[:, 0] 
            y = data[:, 1]

            # print("x-size =", x.shape)
            
            # Create the 2D Histogram
            # bins: (Number of Time bins, Number of Duration bins)
            # cmap: 'inferno', 'viridis', 'Blues', etc.
            # cmin=1: Makes zero-count bins white/transparent instead of dark blue
            h = ax.hist2d(x, y, bins=[x_bins, y_bins], cmap='inferno', cmin=1) # range=[[0, 4], [y.min(), y.max()]].
            
            # Add labels and title
            ax.set_title(legs[leg_idx])
            ax.set_xlabel('Time (Step Midpoint)')
            if leg_idx == 0:
                ax.set_ylabel('Stance Duration (Steps)')
                
            # Add a colorbar for this subplot (optional, but helpful)
            plt.colorbar(h[3], ax=ax, label='Count')

        plt.tight_layout()
        plt.savefig(f"{model_path}/stance_(t,d)_heatmap_n={n}.png", bbox_inches='tight')
        plt.close()