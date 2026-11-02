import numpy as np
import random
# from scipy.stats import circmean # , circstd

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from gymnasium.envs.registration import register

from envs.Clock import Clock
from envs.BodyState import BodyState
from envs.HistoryLogger import HistoryLogger
from envs.RewardFunctions import RewardFunctions

register(id='insect_env', entry_point='envs.mujoco.insectEnv:InsectEnv')

class InsectEnv(MujocoEnv, utils.EzPickle):
    
    def __init__(
        self,
        P=None,
        xml_file=None,
        record=False,
        **kwargs,
    ):
        
        self.P = P

        # TODO: note that in multi-gait mode, the env is initialized with the phase of the first gait.
        self.clocks = []
        for i in range(len(self.P["gaits"])):
            self.clocks.append(Clock(dt_opt=self.P["dt_opt"]))
        
        self.body_state = BodyState()
        self.history = HistoryLogger()
        self.rewards = RewardFunctions()
        
        self.record = record
        
        render_fps = round(1 / self.P["dt_opt"])
        
        utils.EzPickle.__init__(
            self,
            xml_file,
            **kwargs,
        )
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(self.P["observation"]["size"],), dtype=np.float64)
        
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=self.P["frame_skip"],
            observation_space=observation_space,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": render_fps,
        }

        """ if(self.record):
            body_masses = np.array([self.model.body(i).mass[0] for i in range(self.model.nbody)])
            body_names = [self.model.body(i).name for i in range(self.model.nbody)]
            for name, mass in zip(body_names, body_masses):
                print(f"Body: {name}, Mass: {mass}") """

    @property
    def contact_forces(self):
        contact_forces = np.clip(self.data.cfrc_ext, -self.gaits[0]["params"]["contact_force_range"], self.gaits[0]["params"]["contact_force_range"]) / self.gaits[0]["params"]["contact_force_range"]
        return contact_forces
    
    def step(self, action):
        
        terminated = False
        
        for i in range(len(self.gaits)):
            self.clocks[i].tick()
        
        position_before = self.get_body_com("torso")[:3].copy()
        
        self.do_simulation(action, self.frame_skip)

        self.history.log_state(self.data.qpos, self.data.qvel, self.data.geom_xpos, self.data.site_xpos) # This must come before the call to self.get_reward().
        
        position_after = self.get_body_com("torso")[:3].copy()

        if(self.P["multi_gait"]):
            self.time_to_transition = self.clocks[0].t - self.transition_time # Negative before transition, positive after.

        reward, terminated, reward_info = self.get_reward(action, position_before, position_after, terminated)

        self.history.log_previous(action=action, contact_forces=self.contact_forces) # This must come after the call to self.get_reward().

        observation = self._get_obs() # This should come after nudging self.clock.gait_phase.

        info = {
            "history": self.history,
            "reward": reward_info,
            "x_position": position_after[0],
            "y_position": position_after[1],
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "geom_xpos": self.data.geom_xpos.copy(),
            "qpos_ref": 0,
            "cfrc_ext": self.data.cfrc_ext.copy(), # com-based external force on body (nbody x 6).
            "contact_forces": self.contact_forces.copy(), # This is the clipped version of `cfrc_ext`.
        }
        
        if(self.record):
            terminated = False
        
        if self.render_mode == "rgb_array": # if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info
    
    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        
        if self.P["observation"]["exclude_xy_from_obs"]:
            position = position[2:]
        
        if self.P["observation"]["use_gait_phase"]:
            gait_phase = []
            for i in range(len(self.gaits)):
                gait_phase += [np.sin(self.clocks[i].gait_phase), np.cos(self.clocks[i].gait_phase)]
        else:
            gait_phase = []
        
        if self.P["observation"]["use_joint_phases"]:
            joint_phases = []
            for i in range(len(self.gaits)):
                joint_phases += [np.sin(self.clocks[i].joint_phases_current), np.cos(self.clocks[i].joint_phases_current)] # [sin(phase), cos(phase)].
        else:
            joint_phases = []
        
        if self.P["observation"]["use_contact_forces"]:
            contact_force = self.contact_forces.flat.copy()
        else:
            contact_force = []

        if self.P["observation"]["use_contact_time"]:
            last_contact_times = self.clocks[0].last_contact_times.copy() # This is gait-independent, so we simply use the first one.
        else:
            last_contact_times = []
        
        if(self.P["multi_gait"]):
            gait_flag = self.gait_flags[1] if self.time_to_transition > 0 else self.gait_flags[0]
            
            transition_obs = np.array([abs(self.time_to_transition), gait_flag]) # Unsigned time_to_transition.
            # transition_obs = np.array([self.time_to_transition, gait_flag]) # Signed time_to_transition.
            ### transition_obs = np.array([abs(self.time_to_transition)*gait_flag, gait_flag])
            if(self.test):
                print("transition_obs =", transition_obs)
        else:
            transition_obs = []
        
        """ if self.P["observation"]["use_angular_velocity_sign"]:
            angular_velocity_sign = self.angular_velocity_sign.copy()
        else:
            angular_velocity_sign = [] """
        
        # print(self.P["observation"]["size"])
        
        return np.concatenate((position, velocity, gait_phase, joint_phases, contact_force, last_contact_times, transition_obs))

    def get_reward(self, action, position_before, position_after, terminated):
        
        reward_weights = self.gaits[0]["params"]["reward_weights"]
        thresholds = self.gaits[0]["params"]["thresholds"]
        
        if(self.gaits[0]["phase"] in ["double_tripod", "bound"]):
            
            _, y_velocity, z_velocity = self.body_state.get_body_velocity(self.P["dt_opt"], position_before, position_after)
            
            if(self.P["multi_gait"]):
                neg_sigmoid = self.gaits[0]["params"]["gait_transition"]["sigmoid_function"](-self.time_to_transition, self.gaits[0]["params"]["gait_transition"]["slope"], self.gaits[0]["params"]["gait_transition"]["offset"]) # Starts at 1 and goes to 0.
                pos_sigmoid = self.gaits[1]["params"]["gait_transition"]["sigmoid_function"](self.time_to_transition, self.gaits[1]["params"]["gait_transition"]["slope"], self.gaits[1]["params"]["gait_transition"]["offset"]) # Starts at 0 and goes to 1.

                """ if(self.gaits[0]["phase"] == "bound"):
                    neg_sigmoid *= 2
                elif(self.gaits[1]["phase"] == "bound"):
                    pos_sigmoid *= 2 """
                
                progress_reward_gait_0 = self.gaits[0]["direction"] * y_velocity
                progress_reward_gait_1 = self.gaits[1]["direction"] * y_velocity
                progress_reward = (neg_sigmoid * progress_reward_gait_0) + (pos_sigmoid * progress_reward_gait_1)

                # print("neg_sigmoid, pos_sigmoid =", np.round(neg_sigmoid, 2), np.round(pos_sigmoid, 2))
            else: # Single gait.
                progress_reward = self.gaits[0]["direction"] * y_velocity

            if(self.data.qpos[2] > thresholds["max_z_pos"]):
                terminated = True
                # print("Terminated: max_z_pos exceeded.", self.data.qpos[2])
            
            if(not reward_weights["contact_time"] == 0):
                if(self.P["multi_gait"]):
                    
                    # Give preference to the bound gait in the contact time penalty. If both are bound, both get amplified.
                    if(self.gaits[0]["phase"] == "bound"):
                        gait_0_factor = 1
                    else:
                        gait_0_factor = 1
                    
                    if(self.gaits[1]["phase"] == "bound"):
                        gait_1_factor = 1
                    else:
                        gait_1_factor = 1
                    
                    # This updates `self.clock.last_contact_times`, and the order of values actually does not depend on `leg_sets`. The only requirement is that values in `terminal_geoms` are the same for all gaits (should be anyway for the same animal), 
                    self.clocks[0].update_last_contact_time(self.model.geom, self.data.contact, self.P["dt_opt"], self.gaits[0]["params"]["terminal_geoms"], self.gaits[0]["params"]["leg_sets"])
                    self.clocks[1].update_last_contact_time(self.model.geom, self.data.contact, self.P["dt_opt"], self.gaits[1]["params"]["terminal_geoms"], self.gaits[1]["params"]["leg_sets"])
                    
                    intra_set_contact_std_0, inter_set_contact_std_0 = self.rewards.get_contact_time_reward(self.gaits[0]["params"]["leg_sets"], self.clocks[0].last_contact_times, self.gaits[0]["phase"], self.gaits[0]["period"])
                    intra_set_contact_std_1, inter_set_contact_std_1 = self.rewards.get_contact_time_reward(self.gaits[1]["params"]["leg_sets"], self.clocks[1].last_contact_times, self.gaits[1]["phase"], self.gaits[1]["period"])
                    
                    W = reward_weights["contact_time_intra_inter"]
                    contact_time_reward_0 = (W[0]*intra_set_contact_std_0 + W[1]*inter_set_contact_std_0) / sum(np.abs(W))
                    contact_time_reward_1 = (W[0]*intra_set_contact_std_1 + W[1]*inter_set_contact_std_1) / sum(np.abs(W))
                    
                    contact_time_reward = (gait_0_factor * neg_sigmoid * contact_time_reward_0) + (gait_1_factor * pos_sigmoid * contact_time_reward_1)

                    intra_set_contact_std = (gait_0_factor * neg_sigmoid * intra_set_contact_std_0) + (gait_1_factor * pos_sigmoid * intra_set_contact_std_1)
                    inter_set_contact_std = (gait_0_factor * neg_sigmoid * inter_set_contact_std_0) + (gait_1_factor * pos_sigmoid * inter_set_contact_std_1)

                    # print(np.round(intra_set_contact_std_0, 2), np.round(inter_set_contact_std_0, 2), np.round(intra_set_contact_std_1, 2), np.round(inter_set_contact_std_1, 2))
                    # print("t=", np.round(self.clocks[0].t, 2), np.round(contact_time_reward_0, 4), np.round(contact_time_reward_1, 4))
                    # print(np.round(self.clocks[0].last_contact_times, 2))
                    # print("t=", self.clocks[0].t, "; self.time_to_transition=", self.time_to_transition, "sigmoids=", neg_sigmoid, pos_sigmoid)

                    if(self.test):
                        print("intra_set_contact_std_0 =", np.round(intra_set_contact_std_0, 3), "intra_set_contact_std_1 =", np.round(intra_set_contact_std_1, 3))
                        print("inter_set_contact_std_0 =", np.round(inter_set_contact_std_0, 3), "inter_set_contact_std_1 =", np.round(inter_set_contact_std_1, 3))
                else: # Single gait.
                    self.clocks[0].update_last_contact_time(self.model.geom, self.data.contact, self.P["dt_opt"], self.gaits[0]["params"]["terminal_geoms"], self.gaits[0]["params"]["leg_sets"])
                    intra_set_contact_std, inter_set_contact_std = self.rewards.get_contact_time_reward(self.gaits[0]["params"]["leg_sets"], self.clocks[0].last_contact_times, self.gaits[0]["phase"], self.gaits[0]["period"])

                    W = reward_weights["contact_time_intra_inter"]
                    contact_time_reward = (W[0]*intra_set_contact_std + W[1]*inter_set_contact_std) / sum(np.abs(W))
                    
                    # TODO: this should be time-dependent (otherwise the inital pose will terminate) - terminate only if violated for a certain amount of time (dependent on the gait period).
                    """ if(contact_time_reward < -1.5 and contact_time_violation_time > 0.5 * self.gaits[0]["period"]):
                        terminated = True """
            else:
                contact_time_reward = contact_time_reward_0 = contact_time_reward_1 = 0
                intra_set_contact_std = inter_set_contact_std = 0
                intra_set_contact_std_0 = intra_set_contact_std_1 = inter_set_contact_std_0 = inter_set_contact_std_1 = 0
            
            if(not reward_weights["angular_velocity"] == 0):
                ii = [2,3] # Middle horizontal joints.
                # ii = [2,3, 4,5] # Middle & hind horizontal joints.
                
                if(self.P["multi_gait"]):

                    """ qpos_0 = self.data.qpos[np.array(self.gaits[0]["params"]["horizontal_joints"]["qpos"])[ii]]
                    qvel_0 = self.data.qvel[np.array(self.gaits[0]["params"]["horizontal_joints"]["qvel"])[ii]]
                    A0 = np.array(self.gaits[0]["params"]["amplitudes"]["horizontal"])[ii]
                    φ0 = np.array(self.gaits[0]["params"]["leg_phases"]["horizontal"])[ii]
                    Δ0 = np.array(self.gaits[0]["params"]["offsets"]["horizontal"])[ii]

                    qpos_1 = self.data.qpos[np.array(self.gaits[1]["params"]["horizontal_joints"]["qpos"])[ii]]
                    qvel_1 = self.data.qvel[np.array(self.gaits[1]["params"]["horizontal_joints"]["qvel"])[ii]]
                    A1 = np.array(self.gaits[1]["params"]["amplitudes"]["horizontal"])[ii]
                    φ1 = np.array(self.gaits[1]["params"]["leg_phases"]["horizontal"])[ii]
                    Δ1 = np.array(self.gaits[1]["params"]["offsets"]["horizontal"])[ii]

                    qpos_ref_0, qvel_ref_0 = self.body_state.get_reference_joint_state(self.clocks[0].gait_phase, self.clocks[0].ω, A0, φ0, Δ0)
                    qpos_ref_1, qvel_ref_1 = self.body_state.get_reference_joint_state(self.clocks[1].gait_phase, self.clocks[1].ω, A1, φ1, Δ1)

                    angular_velocity_errors_0 = np.zeros(len(ii))
                    for i in range(len(angular_velocity_errors_0)):
                        angular_velocity_errors_0[i] = self.rewards.get_sine_velocity_reward(qpos_0[i] - Δ0[i], qvel_0[i], A0[i], self.clocks[0].ω, np.sign(qvel_ref_0[i]))

                    angular_velocity_error_0 = np.mean((angular_velocity_errors_0**2) * (np.abs(qpos_0-Δ0) / A0)) # Mean squared error, scaled by the ratio between the absolute current joint angle and the amplitude.
                    
                    angular_velocity_errors_1 = np.zeros(len(ii))
                    for i in range(len(angular_velocity_errors_1)):
                        angular_velocity_errors_1[i] = self.rewards.get_sine_velocity_reward(qpos_1[i] - Δ1[i], qvel_1[i], A1[i], self.clocks[1].ω, np.sign(qvel_ref_1[i]))

                    angular_velocity_error_1 = np.mean((angular_velocity_errors_1**2) * (np.abs(qpos_1-Δ1) / A1)) # Mean squared error, scaled by the ratio between the absolute current joint angle and the amplitude.
                    
                    angular_velocity_error = (neg_sigmoid * angular_velocity_error_0) + (pos_sigmoid * angular_velocity_error_1) """

                    # Temporary: checking if the AV reward prevents convegence in multi_gait mode.
                    angular_velocity_error = qpos_ref_0 = qvel_ref_0 = qpos_ref_1 = qvel_ref_1 = 0
                else:
                    qpos = self.data.qpos[np.array(self.gaits[0]["params"]["horizontal_joints"]["qpos"])[ii]]
                    qvel = self.data.qvel[np.array(self.gaits[0]["params"]["horizontal_joints"]["qvel"])[ii]]
                    A = np.array(self.gaits[0]["params"]["amplitudes"]["horizontal"])[ii]
                    φ = np.array(self.gaits[0]["params"]["leg_phases"]["horizontal"])[ii]
                    Δ = np.array(self.gaits[0]["params"]["offsets"]["horizontal"])[ii]

                    qpos_ref, qvel_ref = self.body_state.get_reference_joint_state(self.clocks[0].gait_phase, self.clocks[0].ω, A, φ, Δ)

                    angular_velocity_errors = np.zeros(len(ii))
                    for i in range(len(angular_velocity_errors)):
                        angular_velocity_errors[i] = self.rewards.get_sine_velocity_reward(qpos[i] - Δ[i], qvel[i], A[i], self.clocks[0].ω, np.sign(qvel_ref[i]))

                    angular_velocity_error = np.mean((angular_velocity_errors**2) * (np.abs(qpos-Δ) / A)) # Mean squared error, scaled by the ratio between the absolute current joint angle and the amplitude.
                    # angular_velocity_error = np.mean(angular_velocity_errors**2) # Mean squared error.
            else:
                angular_velocity_error = 0

                if(self.P["multi_gait"]):
                    qpos_ref_0 = qpos_ref_1 = qvel_ref_0 = qvel_ref_1 = 0
                else:
                    qpos_ref = qvel_ref = 0

            if(not reward_weights["qpos"] == 0):
                
                if(self.P["multi_gait"]):
                    phase_diffs_0, qpos_errors_0, qvel_diffs_0, mean_phase_0, self.clocks[0].joint_phases_current = self.body_state.get_body_state(self.gaits[0]["params"], self.data.qpos, self.data.qvel, self.clocks[0].gait_phase, self.clocks[0].ω)
                    phase_diffs_1, qpos_errors_1, qvel_diffs_1, mean_phase_1, self.clocks[1].joint_phases_current = self.body_state.get_body_state(self.gaits[1]["params"], self.data.qpos, self.data.qvel, self.clocks[1].gait_phase, self.clocks[1].ω)

                    phase_diffs_mean_0 = self.rewards.get_body_state_reward(phase_diffs_0)
                    phase_diffs_mean_1 = self.rewards.get_body_state_reward(phase_diffs_1)
                    
                    self.clocks[0].update_gait_phase(mean_phase_0, self.gaits[0]["params"]["adaptive_gait_phase_circular_mean_weights"])
                    self.clocks[1].update_gait_phase(mean_phase_1, self.gaits[1]["params"]["adaptive_gait_phase_circular_mean_weights"])

                    # Scale the error of each gait for early termination
                    """ if((self.time_to_transition < -0.5 and np.mean(qpos_errors_0) > 0.5) or (self.time_to_transition > 0.5 and np.mean(qpos_errors_1) > 0.5)):
                        terminated = True """
                    """ if((neg_sigmoid * np.mean(qpos_errors_0)) > 0.5 or (pos_sigmoid * np.mean(qpos_errors_1)) > 0.5):
                        terminated = True """

                    W = [reward_weights["qpos_subweights"]["phase"], reward_weights["qpos_subweights"]["qpos"], reward_weights["qpos_subweights"]["qvel"]] # [0.25, 5, 0]
                    body_state_reward_0 = (W[0]*phase_diffs_mean_0 + W[1]*np.mean(qpos_errors_0**2) + W[2]*np.mean(qvel_diffs_0)) / 3
                    body_state_reward_1 = (W[0]*phase_diffs_mean_1 + W[1]*np.mean(qpos_errors_1**2) + W[2]*np.mean(qvel_diffs_1)) / 3

                    body_state_reward = (neg_sigmoid * body_state_reward_0) + (pos_sigmoid * body_state_reward_1)
                    
                else:
                    phase_diffs, qpos_errors, qvel_diffs, mean_phase, self.clocks[0].joint_phases_current = self.body_state.get_body_state(self.gaits[0]["params"], self.data.qpos, self.data.qvel, self.clocks[0].gait_phase, self.clocks[0].ω)
                    # print(phase_diffs, qpos_errors, qvel_diffs, mean_phase, self.clocks[0].joint_phases_current)
                    
                    phase_diffs_mean = self.rewards.get_body_state_reward(phase_diffs)
                    self.clocks[0].update_gait_phase(mean_phase, self.gaits[0]["params"]["adaptive_gait_phase_circular_mean_weights"])
                    
                    # if np.any(phase_diffs > (np.pi / 4)):
                    # if(phase_diffs_mean > self.gaits[0]["params"]["thresholds"]["mean_joint_phase_error"]):
                    # if(np.any(qpos_errors > 0.5)):
                    # [0,1,6,7, 2,3,8,9, 4,5,10,11]
                    # if(np.mean(qpos_errors[[0,1]]) > 0.5 or np.mean(qpos_errors[[2,3]]) > 0.5 or np.mean(qpos_errors[[4,5]]) > 0.5):
                    # if(np.mean(qpos_errors) > 0.5 or np.mean(qpos_errors[[12,13]])  > 0.5): # [*0.5*].
                    if(np.mean(qpos_errors) > 0.5): # [*0.5*].
                        terminated = True
                        print("Terminated: np.mean(qpos_errors) > 0.5.")
                        """ if(self.record):
                            print("Terminated due to body state reward. mean_phase =", np.round(mean_phase*180/np.pi, 2), "phase_diffs, qpos_errors, qvel_diffs =", np.round(np.array([phase_diffs, qpos_errors, qvel_diffs])*180/np.pi, 2)) """
                    # print("mean_phase =", np.round(mean_phase*180/np.pi, 2), "phase_diffs, qpos_diffs, qvel_diffs =", np.round(np.array([phase_diffs, qpos_errors, qvel_diffs])*180/np.pi, 2))

                    W = [reward_weights["qpos_subweights"]["phase"], reward_weights["qpos_subweights"]["qpos"], reward_weights["qpos_subweights"]["qvel"]] # [0.25, 5, 0]
                    body_state_reward = (W[0]*phase_diffs_mean + W[1]*np.mean(qpos_errors**2) + W[2]*np.mean(qvel_diffs)) / 3
                    
                    # Testing: a reward for larger absolute middle J0 angles (this is later multiplied by a negative weight).
                    # body_state_reward -= 0.2*np.mean(np.abs(self.data.qpos[np.array(self.gaits[0]["params"]["horizontal_joints"]["qpos"])[[2,3]]]))
                    # body_state_reward = phase_diffs_mean # (qpos_error + qvel_error)
            else:
                body_state_reward = 0

                if(self.P["multi_gait"]):
                    phase_diffs_mean_0 = phase_diffs_mean_1 = 0
                else:
                    phase_diffs_mean = 0
            
            if(not reward_weights["leg_angles_stats"] == 0):
                leg_angles_reward_stats, leg_angles_reward_mean, leg_angles_reward_std = self.leg_angles_reward()
            else:
                leg_angles_reward_stats = 0
                leg_angles_reward_mean, leg_angles_reward_std = 0, 0
            
            action_smoothness_reward = self.rewards.get_action_smoothness_reward(action, self.history["previous_action"])
            
            if(not reward_weights["contact_force_smoothness"] == 0):
                contact_forces_smoothness_reward = self.rewards.get_contact_forces_smoothness_reward(self.contact_forces.flatten(), self.history["previous_contact_forces"])
            else:
                contact_forces_smoothness_reward = 0
            
            roll, pitch, yaw = self.body_state.get_body_orientation(self.data.qpos[3:7]) # Computing this anyway for logging purposes.
            if(not reward_weights["orientation"] == 0):
                # Compute orientation errors using circular smallest difference
                
                roll_error = abs(roll - 0) % (2 * np.pi)
                pitch_error = abs(pitch - 0) % (2 * np.pi)
                yaw_error = abs(yaw - thresholds["target_yaw_angle"]) % (2 * np.pi)
                
                roll_error = np.minimum(roll_error, (2 * np.pi) - roll_error) # Target roll is 0°.
                pitch_error = np.minimum(pitch_error, (2 * np.pi) - pitch_error) # Target pitch is 0°.
                yaw_error = np.minimum(yaw_error, (2 * np.pi) - yaw_error)

                orientation_reward = roll_error + pitch_error + yaw_error
                
                if(roll_error > thresholds["max_roll_angle"] or pitch_error > thresholds["max_pitch_angle"] or yaw_error > thresholds["max_yaw_angle"]):
                    terminated = True
                    # print("Terminated: orientation thresholds exceeded.", np.round(np.array([roll_error, pitch_error, yaw_error])*180/np.pi, 2), "t=", np.round(self.clocks[0].t,2))
                    # if(self.record):
                        # print("Terminated due to orientation condition. roll, pitch, yaw =", np.round(np.array([roll, pitch, yaw])*180/np.pi, 2))
            
            else:
                orientation_reward = 0
            # print(np.round(np.array([roll, pitch, yaw])*180/np.pi, 2))

            if(not reward_weights["left_right_symmetry"] == 0):
                if(self.P["multi_gait"]):
                    joint_indices_0 = np.concatenate((self.gaits[0]["params"]["horizontal_joints"]["qpos"], self.gaits[0]["params"]["vertical_joints"]["qpos"])) # All horizontal and vertical joints (+ the hind J0).
                    joint_indices_1 = np.concatenate((self.gaits[1]["params"]["horizontal_joints"]["qpos"], self.gaits[1]["params"]["vertical_joints"]["qpos"])) # All horizontal and vertical joints (+ the hind J0).

                    left_right_symmetry_reward_0 = self.rewards.get_left_right_symmetry_reward(self.data.qpos, self.model.qpos_spring, joint_indices_0, self.gaits[0]["phase"])
                    left_right_symmetry_reward_1 = self.rewards.get_left_right_symmetry_reward(self.data.qpos, self.model.qpos_spring, joint_indices_1, self.gaits[1]["phase"])

                    left_right_symmetry_reward = (neg_sigmoid * left_right_symmetry_reward_0) + (pos_sigmoid * left_right_symmetry_reward_1)
                else:
                    joint_indices = np.concatenate((self.gaits[0]["params"]["horizontal_joints"]["qpos"], self.gaits[0]["params"]["vertical_joints"]["qpos"])) # All horizontal and vertical joints (+ the hind J0).
                    # joint_indices = np.concatenate((self.gaits[0]["params"]["horizontal_joints"]["qpos"], self.gaits[0]["params"]["vertical_joints"]["qpos"][-2:]))
                    # joint_indices = self.gaits[0]["params"]["vertical_joints"]["qpos"][-2:]
                    
                    left_right_symmetry_reward = self.rewards.get_left_right_symmetry_reward(self.data.qpos, self.model.qpos_spring, joint_indices, self.gaits[0]["phase"])
            else:
                left_right_symmetry_reward = 0
            
            reward = 0
            reward += reward_weights["progress"] * progress_reward
            reward += reward_weights["qpos"] * body_state_reward
            reward += reward_weights["angular_velocity"] * angular_velocity_error
            
            reward += reward_weights["action_smoothness"] * action_smoothness_reward
            reward += reward_weights["orientation"] * orientation_reward

            reward += reward_weights["left_right_symmetry"] * left_right_symmetry_reward

            reward += reward_weights["contact_time"] * contact_time_reward
            reward += reward_weights["leg_angles_stats"] * leg_angles_reward_stats
            # reward += reward_weights["contact_interleg_phase"] * contact_reward
            reward += reward_weights["contact_force_smoothness"] * contact_forces_smoothness_reward
            
            # reward += reward_weights["angular_velocity"] * angular_velocity_reward
            
            W = np.array([
                reward_weights["progress"],
                reward_weights["contact_time"],
                reward_weights["left_right_symmetry"],
                # reward_weights["angular_velocity"],
                reward_weights["action_smoothness"],
                reward_weights["orientation"],
                reward_weights["leg_angles_stats"],
                reward_weights["qpos"],
                reward_weights["contact_force_smoothness"]
            ])

            reward /= sum(np.abs(W))
            
            reward += reward_weights["step_reward"]

            reward_info = {
                "progress": progress_reward if not self.P["multi_gait"] else [progress_reward_gait_0, progress_reward_gait_1] if self.gait_flags[0] == -1 else [progress_reward_gait_1, progress_reward_gait_0],
                "y_pos": self.data.qpos[1], 
                "z_pos": self.data.qpos[2],
                "side_motion": self.data.qpos[0],
                "contact_time": contact_time_reward if not self.P["multi_gait"] else [contact_time_reward_0, contact_time_reward_1] if self.gait_flags[0] == -1 else [contact_time_reward_1, contact_time_reward_0],
                "intra_set_contact_std": intra_set_contact_std if not self.P["multi_gait"] else [intra_set_contact_std_0, intra_set_contact_std_1] if self.gait_flags[0] == -1 else [intra_set_contact_std_1, intra_set_contact_std_0],
                "inter_set_contact_std": inter_set_contact_std if not self.P["multi_gait"] else [inter_set_contact_std_0, inter_set_contact_std_1] if self.gait_flags[0] == -1 else [inter_set_contact_std_1, inter_set_contact_std_0],
                # "angular_velocity": angular_velocity_reward,
                "leg_angles": leg_angles_reward_stats,
                "leg_angles_reward_mean": leg_angles_reward_mean,
                "leg_angles_reward_std": leg_angles_reward_std,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "phase_diffs_mean": phase_diffs_mean if not self.P["multi_gait"] else [phase_diffs_mean_0, phase_diffs_mean_1],
                "gait_phase": [self.clocks[i].gait_phase for i in range(len(self.gaits))],
                # "gait_phase": self.clocks[0].gait_phase if not self.P["multi_gait"] else [self.clocks[0].gait_phase, self.clocks[1].gait_phase],
                "joint_phases": [self.clocks[i].joint_phases_current for i in range(len(self.gaits))],
                # "joint_phases": self.clocks[0].joint_phases_current if not self.P["multi_gait"] else [self.clocks[0].joint_phases_current, self.clocks[1].joint_phases_current],
                "qpos_ref": qpos_ref if not self.P["multi_gait"] else [qpos_ref_0, qpos_ref_1],
                "qvel_ref": qvel_ref if not self.P["multi_gait"] else [qvel_ref_0, qvel_ref_1],
            }
        elif(self.gaits[0]["type"] == "jump"):
            
            reward = 0

            zpos = self.get_body_com("torso")[2].copy()

            _, y_velocity, z_velocity = self.body_state.get_body_velocity(self.P["dt_opt"], position_before, position_after)
            roll, pitch, yaw = self.body_state.get_body_orientation(self.data.qpos[3:7])
            self.clocks[0].update_last_contact_time(self.model.geom, self.data.contact, self.P["dt_opt"], self.gaits[0]["params"]["terminal_geoms"], self.gaits[0]["params"]["leg_sets"])
            
            target_roll = 0 # 0°.
            target_pitch = 340 * (np.pi/180) # -20°.
            target_yaw = 180 * (np.pi/180) # 180° yaw is the target for the jump.
            max_roll_error = 30 * (np.pi/180)
            max_pitch_error = 30 * (np.pi/180)
            max_yaw_error = 45 * (np.pi/180)

            # Compute orientation errors using circular smallest difference
            roll_error = abs(roll - target_roll) % (2 * np.pi)
            roll_error = np.minimum(roll_error, (2 * np.pi) - roll_error) 
            pitch_error = abs(pitch - target_pitch) % (2 * np.pi)
            pitch_error = np.minimum(pitch_error, (2 * np.pi) - pitch_error)
            yaw_error = abs(yaw - target_yaw) % (2 * np.pi)
            yaw_error = np.minimum(yaw_error, (2 * np.pi) - yaw_error)

            orientation_reward = roll_error + pitch_error + 3*yaw_error
            orientation_weight = -1

            # if(self.clocks[0].gait_phase < (np.pi / 2)): # Use quarter of the cycle for pose preparation.
            if(False): # Use quarter of the cycle for pose preparation.
                # print(self.clocks[0].update_last_contact_time)
                reward += reward_weights["pre_jump_contact"] * np.mean(self.clocks[0].last_contact_times)
                reward += orientation_weight * orientation_reward
                W = np.array([reward_weights["pre_jump_contact"], orientation_weight])

                # Early termination conditions
                if(roll_error > max_roll_error or pitch_error > max_pitch_error or yaw_error > max_yaw_error):
                    terminated = True

                # print(orientation_reward, reward)
                # print(np.round(np.array([roll, pitch, yaw])*180/np.pi, 2), "roll_error, pitch_error, yaw_error =", np.round(np.array([roll_error, pitch_error, yaw_error])*180/np.pi, 2))
            else:
                # reward += reward_weights["progress"] * self.gaits[0]["direction"] * y_velocity
                reward += reward_weights["jump"] * (zpos**2) # z_velocity
                # W = np.array([reward_weights["progress"], reward_weights["jump"]])
                W = np.array([reward_weights["jump"]])
            
            # reward /= sum(np.abs(W))
            
            reward += reward_weights["step_reward"]

            # print("total reward = ", reward)

            reward_info = {
                "y_velocity": y_velocity,
                "z_velocity": z_velocity,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "gait_phase": self.clocks[0].gait_phase,
            }

        return reward, terminated, reward_info

    def weighted_circular_mean(self, angles, weights):
        
        # Angles must be given in radians.
        # Weight should sum up to 1.
        
        x = y = 0
        for angle, weight in zip(angles, weights):
            x += np.cos(angle) * weight
            y += np.sin(angle) * weight
        
        return np.arctan2(y, x) % (2 * np.pi)

    def reset_model(self, seed=None, gait_phase=None):
        
        self.test = False

        if(seed is not None):
            np.random.seed(seed)
        
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        if(self.P["multi_gait"]): # Set the gait phase and randomize gait order transition time.

            self.transition_time = self.P["gaits"][0]["params"]["gait_transition"]["time_marker"]
            
            if(not self.record): # If training, randomize the transition marker.
                self.gaits, self.gait_flags = self.body_state.randomize_gait_order(list(self.P["gaits"])) # list() is used to create a shallow copy and prevent changes to self.P["gaits"] inside the function.
                
                """ # Only use to testing
                self.gaits = self.P["gaits"].copy()
                self.gait_flags = [-1, 1] """

                self.transition_time += random.uniform(-self.P["gaits"][0]["params"]["gait_transition"]["time_noise"], self.P["gaits"][0]["params"]["gait_transition"]["time_noise"])
            else:
                self.gaits = self.P["gaits"].copy()
                self.gait_flags = [-1, 1] # This is the default order. -1 is associated with self.P["gaits"][0], and +1 with self.P["gaits"][1].

                # Temporary: swap the order of the gaits during evaluation.
                """ self.gaits.reverse()
                self.gait_flags.reverse() """
        else:
            self.gaits = [self.P["gaits"][0]]
            self.gait_flags = [1]

        if(gait_phase is None):
            if(self.P["observation"]["use_body_state_randomizer"]):
                for i in range(len(self.gaits)):
                    self.clocks[i].reset(T=self.gaits[i]["period"]) # Randomize the gait phase uniformly between 0 and 2π and set the time to zero.
            else:
                for i in range(len(self.gaits)):
                    self.clocks[i].reset(T=self.gaits[i]["period"], gait_phase=0) # Set the gait phase to zero.
        else:
            for i in range(len(self.gaits)):
                self.clocks[i].reset(T=self.gaits[i]["period"], gait_phase=gait_phase) # Randomize the gait phase uniformly between 0 and 2π and set the time to zero.

        if(self.P["multi_gait"]): # Set the gait phase and randomize gait order transition time.
            self.time_to_transition = self.clocks[0].t - self.transition_time # Negative before transition, positive after.
        
        if(self.P["observation"]["use_body_state_randomizer"]):
            # Randomize the body state according to the first gait (after gait order randomization)
            qpos_ind = np.concatenate([self.gaits[0]["params"]["horizontal_joints"]["qpos"], self.gaits[0]["params"]["vertical_joints"]["qpos"]])
            qvel_ind = np.concatenate([self.gaits[0]["params"]["horizontal_joints"]["qvel"], self.gaits[0]["params"]["vertical_joints"]["qvel"]])
            A = np.concatenate([self.gaits[0]["params"]["amplitudes"]["horizontal"], self.gaits[0]["params"]["amplitudes"]["vertical"]])
            φ = np.concatenate([self.gaits[0]["params"]["leg_phases"]["horizontal"], self.gaits[0]["params"]["leg_phases"]["vertical"]])
            Δ = np.concatenate([self.gaits[0]["params"]["offsets"]["horizontal"], self.gaits[0]["params"]["offsets"]["vertical"]])
            # print(qpos_ind, qvel_ind, A, φ, Δ)
            qpos_sub, qvel_sub = self.body_state.get_reference_joint_state(self.clocks[0].gait_phase, self.clocks[0].ω, A, φ, Δ)
            
            qpos[qpos_ind] = qpos_sub
            qvel[qvel_ind] = qvel_sub

        else: # Add noise.
            noise_low = -self.gaits[0]["params"]["qpos_noise_scale"]
            noise_high = self.gaits[0]["params"]["qpos_noise_scale"]
            qpos = qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
            
            """ qpos_noise = 0.5 # 0.5 rad = ~28 deg.
            joint_indices = np.concatenate((self.gaits[0]["params"]["horizontal_joints"]["qpos"], self.gaits[0]["params"]["vertical_joints"]["qpos"]))
            qpos[joint_indices] += self.np_random.uniform(low=-qpos_noise, high=qpos_noise, size=len(joint_indices)) """
            
            
            qvel = qvel + (self.gaits[0]["params"]["qvel_noise_scale"] * self.np_random.standard_normal(self.model.nv))

        self.set_state(qpos, qvel)
        
        for i in range(len(self.gaits)):
            if(not self.gaits[i]["params"]["reward_weights"]["contact_time"] == 0):
                self.clocks[i].update_last_contact_time(self.model.geom, self.data.contact, self.P["dt_opt"], self.gaits[i]["params"]["terminal_geoms"], self.gaits[i]["params"]["leg_sets"]) # This isn't really dependent on the gait.

        self.history.reset(self.data.geom_xpos.shape, self.data.site_xpos.shape)
        self.history.log_state(qpos, qvel, self.data.geom_xpos, self.data.site_xpos)
        self.history.log_previous(action=None, contact_forces=self.contact_forces)

        # Set the current joint phases in self.clock
        for i in range(len(self.gaits)):
            if(self.gaits[i]["type"] == "walk" and self.gaits[i]["reward"] == "joint_based"):
                _, _, _, _, self.clocks[i].joint_phases_current = self.body_state.get_body_state(self.gaits[i]["params"], self.data.qpos, self.data.qvel, self.clocks[i].gait_phase, self.clocks[i].ω)
            else:
                self.clocks[i].joint_phases_current = None

        observation = self._get_obs()

        # print(observation.shape)
        
        """ if(not gait_phase is None):
            self.render() """
        
        if(self.test):
            print("Environment reset complete.")
            print("gaits =", self.gaits[0]["phase"], self.gaits[1]["phase"], self.gait_flags)
        
        return observation