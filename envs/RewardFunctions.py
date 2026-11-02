import numpy as np
from scipy.spatial.transform import Rotation as R
# from astropy.stats import circmean # Circular mean with weights.

from envs.mujoco.functions.get_leg_angles_all import get_leg_angles_all

class RewardFunctions:
    def __init__(self):
        pass
    
    def get_body_state_reward(self, phase_diffs):
        return np.mean(phase_diffs)
    
    def get_new_contact_time_reward(self, leg_sets, last_contact_time, gait_phase, T=None, set_target_period=True):
        # Compute the contact-based reward using an adaptive global clock.
        

        # 1. Get the current phase within the *reference* cycle.
        # gait_phase

        # 2. Compute the reference contact state (boolean) and swing/stance duration within the current cycle. When a new cycle starts, the reference durations reset (all legs should change state).
        stance_phaes = np.pi/2 # gait.phase
        if stance_phaes < gait_phase: # This means the first leg set should be in stance.
            print(1)
        else: # This means the second leg set should be in stance.
            print(1)

        # 3. Compute a penalty that captures the deviations from the reference state.
        # For each leg, compare its curernt and reference contact states. A match is +1, a mismatch is -1 (or 0?).
        # Avergae scores across legs.

        # 4. Compute an actual phase based on the actual state (swing/stance state and their durations within the cycle).

        # 5. Nudge the global phase based on the actual phase.

        # * It is important to avoid mixing swing and stance durations, as the agent, for example, might try to compensate for a shorter stance by being in stance during the ref swing phase.


    def get_contact_time_reward(self, leg_sets, last_contact_time, gait_phase, T=None, set_target_period=True):
        
        # Minimize the std of contact time within each leg set, and minimize the diff of the means diff and half period.
        contact_time_std = np.empty(len(leg_sets))
        contact_time_mean = np.empty(len(leg_sets))
        for i in range(len(leg_sets)): # Compute mean and std for each leg set.
            contact_time_mean[i] = np.mean(last_contact_time[leg_sets[i]])
            contact_time_std[i] = np.std(last_contact_time[leg_sets[i]])

            # print("gait phase =", gait_phase, f"leg set {i} =", leg_sets[i], "contact times =", last_contact_time[leg_sets[i]], "std =", np.round(contact_time_std[i], 3))
        
        # intra_set_contact_std = np.mean(contact_time_std) # Minimize the std within each leg set separately.
        intra_set_contact_std = np.sqrt(np.mean(contact_time_std**2))
        
        
        # Minimize the difference between inter-set time lags and the reference time lag (half a period). 
        """ if(gait_phase == "double_tripod"):
            inter_set_contact_diff = np.abs(contact_time_mean[0] - contact_time_mean[1])
            if(set_target_period):
                inter_set_contact_diff = np.abs( inter_set_contact_diff - (T/2) ) # Minimize the difference between the difference of the means and half a period.
            else:
                # low = np.abs( inter_set_contact_diff - 0.1)
                # high = np.abs( inter_set_contact_diff - 0.5)
                inter_set_contact_diff *= (-1) # Maximize the difference between the means.
        elif(gait_phase == "bound"):
            if(set_target_period):
                # inter_set_contact_diff = np.abs( np.abs(contact_time_mean[0] - contact_time_mean[1]) - (T/2) ) # Fore+hind and middle.
                inter_set_contact_diff = np.abs( np.abs(contact_time_mean[1] - contact_time_mean[2]) - (T/2) ) # Middle and hind legs are in anti-phase.
                # inter_set_contact_diff = np.abs( np.abs(contact_time_mean[0] - contact_time_mean[2]) - 0 ) # Front and hind legs are in-phase. """


        # Inter-Set Error (Pairwise Difference) - We compare every leg in Set 0 against every leg in Set 1.
        times_A = last_contact_time[leg_sets[0]]
        times_B = last_contact_time[leg_sets[1]]

        # BROADCASTING MAGIC: Reshape times_A to (N, 1) and subtract times_B (M,)
        diff_matrix = np.abs(times_A[:, np.newaxis] - times_B) # This creates a matrix (N, M) containing the difference of every pair. Matrix[i, j] = times_A[i] - times_B[j].

        if set_target_period:
            pairwise_errors = np.abs(diff_matrix - (T/2)) # We want the difference between sets to be exactly T/2. Calculate how far each pair is from T/2.
            # inter_set_contact_diff_new = np.mean(pairwise_errors) # The final error is the mean error of all interactions.
            inter_set_contact_diff_new = np.sqrt(np.mean(pairwise_errors**2))
        else: # If no specific period is enforced, we just want them "far apart".
            # inter_set_contact_diff_new = -np.mean(diff_matrix) # We maximize the distance (negate the mean difference).
            inter_set_contact_diff_new = -np.sqrt(np.mean(diff_matrix**2))

        # print("gait phase =", gait_phase, "inter_set_contact_diff =", np.round(inter_set_contact_diff, 3), "inter_set_contact_diff_new =", np.round(inter_set_contact_diff_new, 3))
        
        return intra_set_contact_std, inter_set_contact_diff_new
        # return intra_set_contact_std, inter_set_contact_diff
    
    def get_sine_velocity_reward(self, angle, velocity, amplitude, ω, target_velocity_sign):
        
        '''
        Note that this calculation assumes that angle=0 is the equilibrium position.

        This function computes an angular velocity reward by calculating a target velocity and comparing it to the current (measured) velocity.
        The target velocity is computed using the following equation: |target_velocity| = ω·√(A² - θ²).

        angle is the current angle with respect to the equilibrium position (i.g., angle-offset).
        velocity is the current angular velocity from self.data.qvel.
        amplitude is the maximum angle allowed (in radians).
        ω is the target frequency in radians. It is defined by 2*pi*frequency, where frequency is in Hz.
        angular_velocity_sign is the sign of the target angular velocity (1 or -1).
        '''
        
        if abs(angle) <= amplitude: # Check if the angle is within the amplitude range
            v_target = ω * np.sqrt(max(amplitude**2 - angle**2, 0.0)) * target_velocity_sign
        else: # An out-of-range penalty.
            v_target = -(ω * (angle - (np.sign(angle)*amplitude)))

        return (velocity - v_target)

    def get_interjoint_phase_reward(self):
        
        reward = np.zeros(1) # np.zeros(len(self.P["gait_params"]["horizontal_joints"]["qpos"])//2)
        
        # L1 = self.data.qvel[self.P["gait_params"]["horizontal_joints"]["qvel"][0]]
        # R1 = self.data.qvel[self.P["gait_params"]["horizontal_joints"]["qvel"][1]]
        L2 = self.data.qpos[self.P["gait_params"]["horizontal_joints"]["qpos"][2]] #  - self.P["gait_params"]["offsets"]["horizontal"][2]
        R2 = self.data.qpos[self.P["gait_params"]["horizontal_joints"]["qpos"][3]] #  - self.P["gait_params"]["offsets"]["horizontal"][3]

        if(self.gaits[0]["phase"] == "double_tripod"): # Reward legs within each contralateral pair for having equal magnitude and opposite sign.
            reward[0] = abs(L2 + R2) / max(abs(L2), abs(R2))
        else: # Reward legs within each contralateral pair for having equal magnitude and sign.
            reward[0] = abs(L2 - R2) / max(abs(L2), abs(R2))

        return np.mean(reward)

    def get_action_smoothness_reward(self, action, previous_action):
        if(previous_action is None):
            return 0
        else:
            return np.mean(np.abs(action - previous_action))
    
    def get_contact_forces_smoothness_reward(self, contact_forces, previous_contact_forces):
        return np.mean(np.abs(contact_forces.flatten() - previous_contact_forces.flatten()))
    
    def leg_angles_reward(self):
        # TODO: based on the global phase, we can evaluate the mean and std at every time step.
        # This is used to constrain the leg angles to specific mean and std values.
        # This reward is evaluated every full period of the gait cycle.
        
        a = round(self.clock.t, self.P["time_rounding"]) % round(self.gaits[0]["period"], self.P["time_rounding"])
        b = round(a, self.P["time_rounding"])
        c = b % self.gaits[0]["period"]
        
        if(c == 0): # Evaluate the mean and std of leg angles every full period.
            mean_angle, std_angle, _ = get_leg_angles_all(self.history["geom_xpos"], None, len(self.P["objects"]))
            
            leg_angles_mean_diffs = np.abs(mean_angle - self.P["gait_params"]["leg_angles_mean"]) % (2 * np.pi)
            leg_angles_mean_diffs = np.where(leg_angles_mean_diffs > np.pi, (2 * np.pi) - leg_angles_mean_diffs, leg_angles_mean_diffs)

            leg_angles_mean_reward = np.mean(leg_angles_mean_diffs)
            leg_angles_std_reward = np.mean(np.abs(std_angle - self.P["gait_params"]["leg_angles_std"]))
            
            W = np.array([1,1])
            reward_stats = (W[0]*leg_angles_mean_reward + W[1]*leg_angles_std_reward) / sum(np.abs(W))
            # reward_stats = leg_angles_mean_reward
            
        else:
            reward_stats = leg_angles_mean_reward = leg_angles_std_reward = 0
    
        return reward_stats, leg_angles_mean_reward, leg_angles_std_reward
    
    def get_left_right_symmetry_reward(self, qpos, qpos_spring, joint_indices, gait_phase):
        # Minimize the (absolute) difference between matching left and right joints.

        # TODO: consider normalizing by the size of the range (to make small ranges equally important).
        # TODO: consider specifying the signs (depending on the gait).

        qpos_shifted = qpos - qpos_spring # Make joint angles symmetric around the equilibrium position of the spring

        qpos_left_ind = joint_indices[::2]  # Left joints.
        qpos_right_ind = joint_indices[1::2]  # Right joints.
        
        if(gait_phase == "double_tripod"):
            sign = -1 # Each left-right pair of corresponding joints should have the same magnitude, opposite sign angle.
        elif(gait_phase == "bound"):
            sign = 1 # Each left-right pair of corresponding joints should have equal angles.

        qpos_diff = []
        for left_idx, right_idx in zip(qpos_left_ind, qpos_right_ind):
            left_angle = qpos_shifted[left_idx]
            right_angle = qpos_shifted[right_idx]

            diff = np.abs(left_angle - (sign*right_angle))
            # diff = abs(abs(left_angle) - abs(right_angle)) # Smallest circular difference of absolute angles.

            qpos_diff.append(diff)
        
        reward = np.mean(qpos_diff)
        return reward