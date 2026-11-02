import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from collections import defaultdict

from output.insect.plot_swing_stance import plot_swing_stance

def plot_contact_forces(P, t, contact_forces, save_dir, ep, plot_flag):
    '''
    The shape of the contact forces arrays is [nbodies x 6].
    Bodies (29=1+1+3+24): worldbody [0,:], torso [1,:], head [2,:], thorax[3,:], abdoment [4,:], 24 leg parts (6 legs, where each consists of 4 bodies: coxa, femur, tibia, tarsus).
    Indices of terminal contacts
        forelegs tibia: [7,:], [11,:]
        forelegs tarsus: [8,:], [12,:]
        middlelegs tarsus: [16,:], [20,:]
        hindlegs tarsus: [24,:], [28,:]
    '''
    
    win_size = int(0.05 / P["dt_opt"]) # seconds -> frames.
    threshold = 10
    # print(win_size, threshold)

    contact_forces = uniform_filter1d(contact_forces, size=win_size, axis=0, mode='nearest')

    # contact_forces = np.clip(contact_forces, 0, 1000)


    L2x = contact_forces[:, 16, 0]
    R2x = contact_forces[:, 20, 0]
    L2y = contact_forces[:, 16, 1]
    R2y = contact_forces[:, 20, 1]

    ind = 2
    if(P["animal"] == "mole_cricket"):
        L1z = contact_forces[:, 7, ind]
        R1z = contact_forces[:, 11, ind]
    else:
        L1z = contact_forces[:, 8, ind]
        R1z = contact_forces[:, 12, ind]
    L2z = contact_forces[:, 16, ind]
    R2z = contact_forces[:, 20, ind]
    L3z = contact_forces[:, 24, ind]
    R3z = contact_forces[:, 28, ind]
    
    L2tx = contact_forces[:, 16, 3]
    R2tx = contact_forces[:, 20, 3]
    L2ty = contact_forces[:, 16, 4]
    R2ty = contact_forces[:, 20, 4]
    L2tz = contact_forces[:, 16, 5]
    R2tz = contact_forces[:, 20, 5]

    if(plot_flag):
        fig, ax = plt.subplots(2, 3, figsize=(36, 18))
        plt.rcParams.update({'font.size': 24})
        for axis in ax.flat:
            axis.tick_params(axis='both', which='major', labelsize=20)
            axis.tick_params(axis='both', which='minor', labelsize=20)
            axis.xaxis.label.set_size(24)
            axis.yaxis.label.set_size(24)
            axis.title.set_size(24)



    force_labels = ['x', 'y', 'z', 'tx', 'ty', 'tz']
    force_titles = [
        'Translational force in the x direction',
        'Translational force in the y direction',
        'Translational force in the z direction',
        'Torsional force about the x-axis',
        'Torsional force about the y-axis',
        'Torsional force about the z-axis'
    ]
    force_units = [
        'Force [g × cm / s²]',
        'Force [g × cm / s²]',
        'Force [g × cm / s²]',
        'Force [g × cm² / s²]',
        'Force [g × cm² / s²]',
        'Force [g × cm² / s²]'
    ]
    forces = [
        (L2x, R2x), (L2y, R2y), (L2z, R2z),
        (L2tx, R2tx), (L2ty, R2ty), (L2tz, R2tz)
    ]

    if(plot_flag):
        for i, (label, title, unit, (L_force, R_force)) in enumerate(zip(force_labels, force_titles, force_units, forces)):
            row, col = divmod(i, 3)
            ax[row, col].plot(t[:-1], L_force, label=f'L2{label}')
            ax[row, col].plot(t[:-1], R_force, label=f'R2{label}')
            ax[row, col].set_xlabel('Time [s]')
            ax[row, col].set_ylabel(unit)
            ax[row, col].set_title(title)
            ax[row, col].set_xlim(t[0], t[-1] + P["dt_opt"])

        plt.savefig(os.path.join(save_dir, f"cotact_forces_win={win_size}_ep={ep}.png"), bbox_inches='tight')

        plt.close(fig)
    
    if(P["animal"] == "mole_cricket"):
        contact_forces_terminals = contact_forces[:,[7,11,16,20,24,28],:]
    else:
        contact_forces_terminals = contact_forces[:,[8,12,16,20,24,28],:]
    
    # contact_forces_sum = np.mean(np.abs(contact_forces_terminals), axis=2)
    
    contact_forces_sum = contact_forces_terminals[:,:, 5]
    # contact_forces_sum = contact_forces_terminals[:,:, 0]
    contact_forces_sum[contact_forces_sum < 0] = 0

    # print(contact_forces_sum)
    contact_forces_binary = contact_forces_sum.copy()
    
    max = 4*np.mean(np.abs(contact_forces_sum)) # 350

    contact_forces_binary[contact_forces_sum > max] = max
    contact_forces_binary /= max

    th = 0.3
    contact_forces_binary[contact_forces_binary > th] = 1
    contact_forces_binary[contact_forces_binary <= th] = 0

    """ contact_forces_binary[(contact_forces_binary >= -threshold) & (contact_forces_binary <= threshold)] = 0
    contact_forces_binary[contact_forces_binary > threshold] = 1
    contact_forces_binary[contact_forces_binary < -threshold] = 1 """

    contact_forces_binary = 1 - contact_forces_binary
    contact_forces_binary[contact_forces_binary == 0] = 0.3

    if(plot_flag):
        plot_swing_stance(contact_forces_binary, save_dir, f"win={win_size}_th={threshold}_ep={ep}")

    contact_forces_continuous = contact_forces_sum.copy()

    contact_forces_continuous[contact_forces_sum > max] = max
    contact_forces_continuous /= max
    # contact_forces_continuous[contact_forces_sum <= -threshold] = -threshold
    # print(contact_forces_continuous.max(), contact_forces_continuous.min())
    # contact_forces_continuous = contact_forces_continuous / threshold # Normalize to [0,1].
    # contact_forces_continuous /= np.max(np.abs(contact_forces_continuous))
    contact_forces_continuous = 1 - contact_forces_continuous
    # print(contact_forces_continuous.max(), contact_forces_continuous.min())
    
    if(plot_flag):
        plot_swing_stance(contact_forces_continuous, save_dir, f"contact_forces_continuous_win={win_size}_ep={ep}", 'bone')
    
    # Binarize the contact force swing-stance diagram (black indicates contact)
    contact_forces_binarized = contact_forces_continuous
    contact_forces_binarized[contact_forces_binarized < 0.9] = 0
    contact_forces_binarized[contact_forces_binarized >= 0.9] = 1

    stance_log = defaultdict(list)
    for leg_idx, leg_binary_trace in enumerate(contact_forces_binarized.T):
        events = extract_stance_events(1- leg_binary_trace) # Make stance 1 and swing 0.
        stance_log[leg_idx].extend(events)

        # print(leg_idx, len(events))

    if(plot_flag):
        contact_forces_binarized[contact_forces_binarized == 0] = 0.3 # For coloring purposes only.
        plot_swing_stance(contact_forces_binarized, save_dir, f"contact_forces_binarized_win={win_size}_ep={ep}", 'bone', plot_flag)

    return stance_log



def extract_stance_events(binary_row):
    """
    Input: 1D binary array (0=swing, 1=stance)
    Output: List of [midpoint, duration]
    """
    # Find changes: 1 (start), -1 (end)
    # Pad with 0 to detect starts/ends at boundaries
    diffs = np.diff(binary_row, prepend=0, append=0)
    
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    events = []
    
    # Zip matches starts with ends
    for start, end in zip(starts, ends):
        duration = end - start
        
        # Check for boundary cuts (optional but recommended)
        # If a stance starts exactly at 0 or ends exactly at len, 
        # we might want to exclude it as we don't know true duration.
        is_boundary = (start == 0) or (end == len(binary_row))
        
        if not is_boundary and duration > 0:
            t_mid = (start + end) / 2.0
            events.append([t_mid, duration])
            
    return events