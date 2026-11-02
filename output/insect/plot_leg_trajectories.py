import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import pandas as pd
import alphashape
from shapely.geometry import Polygon # , MultiPolygon, GeometryCollection
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_leg_trajectories(P, geom_xpos, site_xpos, objects_num, save_path, name, i, plot_flag=True):

    i0 = 1 + objects_num # The first object is the floor, followed by walls.
    
    # Set leg tip indices in geom_xpos
    leg_tip_indices = np.array([i0+6, i0+10, i0+14 ,i0+18, i0+22, i0+26])

    n_steps = len(geom_xpos)
    n_legs = len(leg_tip_indices)
    
    # pre-allocate storage: [time, leg, (x,y)] and [time, leg, (x,z)]
    traj_top  = np.zeros((n_steps, n_legs, 2))
    traj_side = np.zeros((n_steps, n_legs, 2))
    traj_side_unshifted = np.zeros((n_steps, n_legs, 2))

    xy_envelopes = []
    xy_area = []
    xy_spread = []
    xy_pc1_angle = []
    xy_thorax_dist = []

    for f, x in enumerate(geom_xpos): # For each frame/optimization step.
        # head = x[i0]
        thorax = x[i0+1]
        abdomen = x[i0+2]

        main_body_axis = thorax - abdomen

        yaw = -np.arctan2(main_body_axis[1], main_body_axis[0]) # xy angle of the main body axis.
        rot_xy = R.from_euler('z', yaw) # Rotation in the xy plane.

        # rot, _ = R.align_vectors([[1, 0, 0]], [main_body_axis])
        
        # rotate each leg-tip vector (tip − thorax)
        for j, idx in enumerate(leg_tip_indices): # For each leg.
            
            # site-based
            # tip_vec = site_xpos[f,j] - thorax # Subtract the thorax x,y,z position from the leg tip position.
            
            # geom-based
            tip_vec = x[idx] - thorax # Subtract the thorax x,y,z position from the leg tip position.
            
            tip_rot = rot_xy.apply(tip_vec)
            # tip_rot = rot.apply(tip_vec)
            
            traj_top[f, j]  = tip_rot[:2] # x,y.
            
            # site-based
            traj_side[f, j] = [tip_rot[0], site_xpos[f,j,2]] # x_body, z_world.
            # geom-based
            # traj_side[f, j] = [tip_rot[0], x[idx][2]] # x_body, z_world.
            
            # traj_side[f, j] = tip_rot[[0, 2]] # x_body, z_relative.
            
            # traj_side_unshifted[f, j] = tip_rot[[0, 2]] # (x, z).
            # traj_side_unshifted[f, j][1] += thorax[2] # Shift back to get the absolute z-coordinate (i.e., not relative to the thorax).

    # Remove the first N frames from the trajectories
    n_skip = int(np.ceil(0.25 / P["dt_opt"]))
    traj_top = traj_top[n_skip:]
    traj_side = traj_side[n_skip:]
    # traj_side_unshifted = traj_side_unshifted[n_skip:]
    
    # Compute trajectory envelope (xy plane)
    for j in range(n_legs): # For each leg.
        alpha = 0.5 # Shrink factor for the alpha shape. Smaller values lead to tighter envelopes.
        xy = np.stack([traj_top[:, j, 0], traj_top[:, j, 1]], axis=1)
        shape = alphashape.alphashape(xy, alpha)
        polys = []
        if isinstance(shape, Polygon):
            polys = [shape]
            for poly in polys:
                x_b, y_b = poly.exterior.xy
                xy_envelopes.append((x_b, y_b))
                xy_area.append(poly.area) # Compute area inside the polygon defined by (x_b, y_b).

        # Compute mean and first PCA component using SVD
        xy_points = np.stack([traj_top[:, j, 0], traj_top[:, j, 1]], axis=1)
        mean_xy = np.mean(xy_points, axis=0) # Mean trajectory coordinate for leg j.
        xy_centered = xy_points - mean_xy
        U, S, Vt = np.linalg.svd(xy_centered, full_matrices=False)
        pc1 = Vt[0] # First principal component direction
        angle = np.mod(np.arctan2(pc1[1], pc1[0]), np.pi) # Compute the angle of the first principal component (in radians), modulo pi.
        if angle > (np.pi / 2):
            angle = np.pi - angle # Map angles to [0, pi/2].
        xy_pc1_angle.append((angle))

        # Compute the mean distance of xy_points from mean_xy
        mean_dist = np.mean(np.linalg.norm(xy_centered, axis=1))
        xy_spread.append(mean_dist)

        # Compute the distance of the mean trajectory coordinate from the thorax point (in the top view). Note that xy_points are already shifted to thorax frame, hence mean_xy is also in thorax frame.
        dist = np.linalg.norm(mean_xy)
        xy_thorax_dist.append(dist)

    if(plot_flag):
        # Plot trajectories
        cmap = get_cmap('tab10')  # Replace 'tab10' with any other colormap (e.g., 'viridis', 'plasma')
        color_indices = [0, 4, 1, 2, 8, 9]
        colors = [cmap(i / 9) for i in color_indices]

        # Plot only top view trajectories
        fig1, ax1 = plt.subplots()

        for j in range(n_legs): # For each leg.
            ax1.plot(traj_top[:, j, 0], traj_top[:, j, 1], color=colors[j], lw=0.5, label=f'leg {j+1}')
            x_b, y_b = xy_envelopes[j]
            ax1.plot(x_b, y_b, linewidth=2, color='gray', alpha=0.7)

        ax1.set_xlabel('x (longitudinal)')
        ax1.set_ylabel('y (lateral)')
        
        ax1.axis('equal')
        ax1.set_aspect('equal', adjustable='box')
        ax1.margins(0)
        ax1.set_xlim([-5.5, 2])
        ax1.set_ylim([-2.5, 2.5])
        plt.tight_layout()

        fig1.savefig(os.path.join(save_path, f"leg_trajectories_xy_{name}.png"), bbox_inches='tight')

        plt.close(fig1)

        # Plot top and side view trajectories
        fig2 = plt.figure(figsize=(12, 5))
        gs = fig2.add_gridspec(2, 2)
        ax_top = fig2.add_subplot(gs[:, 0]) # Left: spans both rows.
        ax_side = fig2.add_subplot(gs[0, 1]) # Top-right.
        ax_side_bottom = fig2.add_subplot(gs[1, 1]) # bottom-right.
        
        for j in range(n_legs): # For each leg.
            ax_top.plot(traj_top[:, j, 0], traj_top[:, j, 1], color=colors[j], lw=0.5, label=f'leg {j+1}') # alpha=0.5
            if(j % 2 == 0): # X-Z view of the right side of the body.
                # print(traj_side.shape)
                ax_side.plot(traj_side[:, j, 0], traj_side[:, j, 1], color=colors[j], lw=0.5, label=f'leg {j+1}')
            else:
                ax_side_bottom.plot(traj_side[:, j, 0], traj_side[:, j, 1], color=colors[j], lw=0.5, label=f'leg {j+1}')
                # ax_side_bottom.plot(traj_side_unshifted[:, j, 0], traj_side_unshifted[:, j, 1], color=colors[j], lw=0.5, label=f'leg {j+1}')
            
            # Plot trajectory envelope using alpha shape
            # print("xy_envelopes:", len(xy_envelopes))
            x_b, y_b = xy_envelopes[j]
            ax_top.plot(x_b, y_b, linewidth=2, color='gray', alpha=0.7)
        
        # ax_top .set_title('Intrinsic Top View (x-y)')
        ax_top.set_xlabel('x (longitudinal)')
        ax_top.set_ylabel('y (lateral)')
        
        ax_top.axis('equal')
        ax_side.axis('equal')
        ax_side_bottom.axis('equal')

        for ax in (ax_top, ax_side, ax_side_bottom):
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.set_aspect('equal', adjustable='box')
            ax.margins(0)

        if(P["animal"] == "desert_locust"):
            ax_top.set_xlim([-5.5, 2])
            ax_top.set_ylim([-2.5, 2.5])

            ax_side.set_xlim([-5.5, 2])
            ax_side.set_ylim([0, 1.5])
            # ax_side.set_ylim([-1.5, 0.5])

            ax_side_bottom.set_xlim([-5.5, 2])
            ax_side_bottom.set_ylim([0, 1.5])
            # ax_side_bottom.set_ylim([-1.5, 0.5])
        elif(P["animal"] == "mole_cricket"):
            ax_top.set_xlim([-5.5, 2]) # ax_top.set_xlim([-3.5, 1.5])
            ax_top.set_ylim([-2.5, 2.5]) # ax_top.set_ylim([-2, 2])

            ax_side.set_xlim([-5.5, 2]) # ax_side.set_xlim([-3.5, 1.5])
            ax_side.set_ylim([0, 1.5])
            # ax_side.set_ylim([-1.5, 0.5])
            
            ax_side_bottom.set_xlim([-5.5, 2]) # ax_side_bottom.set_xlim([-3.5, 1.5])
            ax_side_bottom.set_ylim([0, 1.5])
            # ax_side_bottom.set_ylim([-1.5, 0.5])
        
        ax_side.set_ylabel('z (vertical)')
        
        ax_side_bottom.set_xlabel('x (longitudinal)')
        ax_side_bottom.set_ylabel('z (vertical)')
        
        # ax_top .legend(loc='upper right')
        # ax_side.legend(loc='upper right')
        
        ax_side.set_xticks([])
        
        plt.tight_layout()

        fig2.savefig(os.path.join(save_path, f"leg_trajectories_{name}.png"), bbox_inches='tight')

        plt.close(fig2)

    # Save data
    csv_path = os.path.join(save_path, "stats.csv")

    stats = {
        "trajectory_area":  xy_area,
        "trajectory_spread": xy_spread,
        "pc1_angle_rad":   xy_pc1_angle,
        "thorax_dist":     xy_thorax_dist,
    }

    row_idx = i

    # --- build feature-major columns and the row dict ---
    features = list(stats.keys())
    stat_cols = [f"{feat}_leg{j+1}" for feat in features for j in range(n_legs)]
    row = {f"{feat}_leg{j+1}": stats[feat][j] for feat in features for j in range(n_legs)}

    # --- load or init file ---
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    # ensure needed columns exist
    for c in stat_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # --- select target row (append or specific index) ---
    if row_idx is None:
        target_idx = len(df)
        df.loc[target_idx] = pd.NA
    else:
        target_idx = int(row_idx)
        if target_idx >= len(df):
            # extend with NaN rows up to target_idx
            df = pd.concat(
                [df, pd.DataFrame(index=range(len(df), target_idx + 1))],
                ignore_index=True
            )

    # --- assign only the desired columns in that row ---
    for k, v in row.items():
        df.loc[target_idx, k] = v

    # --- keep clean column order: any non-stat columns first, then feature-major stats ---
    other_cols = [c for c in df.columns if c not in stat_cols]
    df = df[other_cols + stat_cols]

    df.to_csv(csv_path, index=False)