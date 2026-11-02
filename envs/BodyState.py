import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.stats import circmean
import copy

class BodyState:
    def __init__(self):
        pass

    def get_body_velocity(self, dt, position_before, position_after):
        velocity = (position_after - position_before) / dt
        x_velocity, y_velocity, z_velocity = velocity
        return x_velocity, y_velocity, z_velocity
    
    def get_body_orientation(self, body_quaternion):
        
        qw, qx, qy, qz = body_quaternion

        r = R.from_quat([qx, qy, qz, qw])
        pitch, roll, yaw = r.as_euler('xyz', degrees=False)

        return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
    
    def get_reference_joint_state(self, gait_phase, ω, A, φ, Δ):
        """
        This function calculates the reference angle and angular velocity for a given set of joints.

        Args:
            gait_phase (float): The target phase of the gait cycle.
            ω (float): The target angular velocity of the gait cycle.
            A (float): The target amplitude of the joint's movement.
            φ (float): The phase offset for the joint.
            Δ (float): The angular offset for the joint.

        Returns:
            The reference angles (qpos) and angular velocities (qvel) for the joints.
        """
        
        qpos = A*np.sin(gait_phase + φ) + Δ
        qvel = A*ω*np.cos(gait_phase + φ)

        return qpos, qvel
    
    def joint_state_to_phase_arcsin(self, angle, offset, A, phase_shift, global_phase):
        """
        Converts the joint angle into a phase value using the arcsin function.
        
        Args:
            angle (float): The angular position of the joint [rad].
            A (float): The amplitude of the joint's motion [rad].

        Returns:
            float: The phase that best matches the global gait phase out of the two candidates.
            Note that the returned phase is in the local joint's coordinates (i.e., without substracting the phase shift).
        """

        y = (angle - offset) / A
        y = np.clip(y, -1.0, 1.0)

        x = np.arcsin(y) # This returns a value in the range [-π/2, π/2].
        phase_candidate_1 = x % (2*np.pi)
        phase_candidate_2 = (np.pi - phase_candidate_1) % (2*np.pi)
        
        arcsin_phases = np.array([phase_candidate_1, phase_candidate_2])

        φ_candidates = (arcsin_phases - phase_shift) % (2*np.pi)
        φ_global = global_phase % (2*np.pi)
        d = (φ_candidates - φ_global + np.pi) % (2*np.pi) - np.pi
        
        idx = np.argmin(np.abs(d))
        best_phase = arcsin_phases[idx]
        
        return best_phase # np.array([phase_candidate_1, phase_candidate_2])
    
    def joint_state_to_phase_arctan2(self, angle, velocity, offset, A, ω):
        """
        Converts the joint state into a phase value using the arctangent function (based on angle and angular velocity).
        
        Args:
            angle (float): The angular position of the joint [rad].
            velocity (float): The angular velocity of the joint [rad/s].
            A (float): The amplitude of the joint's motion [rad].
            ω (float): The angular frequency of the joint's motion [rad/s].

        Returns:
            float: The phase value in radians, normalized to the range [0, 2π).
        """
        
        y = (angle - offset) / A
        x = velocity/(ω * A)
        """
        Normalize the vector (x, y) to unit length. Clipping/normalizing each coordinate separately can lead to incorrect phase values.
        norm = np.hypot(y, x)
        y /= norm
        x /= norm
        """
        phase = np.arctan2(y, x) # This returns a value in the range [-π, π].
        phase = phase % (2 * np.pi)
        return phase

    """ def get_best_phase(self, phase, global_phase):
        
        # phase [rad] is the joint phase after subtracting the phase shift.
        # global_phase [rad] is the global clock of the gait cycle.

        phase_error = (phase - global_phase) % (2*np.pi)
        phase_error = min(phase_error, (2*np.pi) - phase_error) # [0, π].
        phase_error /= np.pi # [0, 1].

        return phase_error """
    
    def get_body_state(self, gait_params, qpos, qvel, gait_phase, ω):

        """
        Compute the mean phase of the gait based on joint angles and velocities.
        Compute the phase difference of each joint from the predicted gait phase.
        """
        
        # ii = [0,1] # J0 forelegs.
        # ii = [0,1,6,7] # J0+J1 forelegs.
        # ii = [2,3] # J0 middle.
        # ii = [8,9] # J1 middle.
        # ii = [2,3,8,9] # J0+J1 middle.
        # ii = [4,5,10,11] # J2+J1 hind.

        # ii = [2,3,8,9, 4,5,10,11] # middle & hind.
        # ii = [0,1,6,7, 2,3,8,9, 4,5,10,11] # J0+J1 of all legs (fore, middle & hind).

        qpos_ind = np.concatenate([gait_params["horizontal_joints"]["qpos"], gait_params["vertical_joints"]["qpos"]]) # [ii]
        qvel_ind = np.concatenate([gait_params["horizontal_joints"]["qvel"], gait_params["vertical_joints"]["qvel"]]) # [ii]
        φ = np.concatenate([gait_params["leg_phases"]["horizontal"], gait_params["leg_phases"]["vertical"]]) # [ii]
        A = np.concatenate([gait_params["amplitudes"]["horizontal"], gait_params["amplitudes"]["vertical"]]) # [ii]
        Δ = np.concatenate([gait_params["offsets"]["horizontal"], gait_params["offsets"]["vertical"]]) # [ii]

        joint_phases = self.joint_state_to_phase_arctan2(qpos[qpos_ind], qvel[qvel_ind], Δ, A, ω)
        self.joint_phases_current = (joint_phases - φ) % (2*np.pi) # This is the phase of the joint in global coordinates (i.e., after substracting the phase shift).

        phase_diffs = (self.joint_phases_current - gait_phase) % (2 * np.pi)
        phase_diffs = np.minimum(phase_diffs, (2 * np.pi) - phase_diffs) # Element-wise, circular smallest difference.

        mean_phase = circmean(self.joint_phases_current, high=2*np.pi, low=0)

        qpos_ref, qvel_ref = self.get_reference_joint_state(gait_phase, ω, A, φ, Δ)

        qpos_diffs = (qpos[qpos_ind] - qpos_ref) % (2 * np.pi)
        qpos_diffs = np.minimum(qpos_diffs, (2 * np.pi) - qpos_diffs) # Element-wise, circular smallest difference.
        qvel_diffs = np.abs(qvel[qvel_ind] - qvel_ref)

        qpos_errors = qpos_diffs / A
        
        # Apply adaptive weights to the qpos_errors based on the target joint phases.
        # adaptive_qpos_weight = np.abs(np.sin(gait_phase + φ)**1) # Adaptive weight based on the target joint phase. It is highest at the extremes of the target range.
        # qpos_errors *= adaptive_qpos_weight
        
        return phase_diffs, qpos_errors, qvel_diffs, mean_phase, self.joint_phases_current
    
    """ def set_transition_params(self, y_pos, gait_flags, transition_position, marker_crossed):

        dist = y_pos - transition_position
        
        if(dist > 0 or marker_crossed): # Once the marker is crossed once, set the flag to the second gait.
            crossed = True
        else:
            crossed = False

        if(marker_crossed or dist > 0):
            self.gait_flag = gait_flags[1]
        else:
            self.gait_flag = gait_flags[0]
        
        return dist, crossed """
    
    def randomize_gait_order(self, gaits):
        new_gaits = copy.deepcopy(gaits) # Avoid modifying the original list.
        
        # Create a list of indices and randomize their order
        indices = list(range(len(new_gaits)))
        random.shuffle(indices)

        # Create a list of gait flags
        gait_flags = [-1, 1]

        # Reorder the gaits and flags lists in the same way based on the above indices
        new_gaits = [new_gaits[i] for i in indices]
        gait_flags = [gait_flags[i] for i in indices]

        return new_gaits, gait_flags
    
    """ def transition_reset(self, transition_position, transition_position_noise, direction):
        self.transition_position = transition_position * direction
        self.transition_position += random.uniform(-transition_position_noise, transition_position_noise) """