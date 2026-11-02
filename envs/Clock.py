import numpy as np

class Clock:
    def __init__(
            self,
            T=None, # Period [s].
            dt_opt=None, # Optimisation time step [s].
    ):
        self.dt_opt = dt_opt

    def reset(self, T=None, gait_phase=None):

        self.ω = 2 * np.pi / T
        self.t = 0

        if(gait_phase is None): # Randomize the gait phase.
            self.gait_phase = np.random.uniform(0, 2*np.pi)
        else:
            self.gait_phase = gait_phase
        
        self.last_contact_times = np.zeros(6)

    def tick(self):
        """Advance the clock by one step"""
        self.gait_phase = (self.gait_phase + (self.ω * self.dt_opt)) % (2 * np.pi)
        self.t += self.dt_opt

    def update_gait_phase(self, mean_joint_phase, circular_mean_weights):
        # Nudge the global clock towards the mean phase of the horizontal joints.
        self.gait_phase = self.weighted_circular_mean(( self.gait_phase, mean_joint_phase ) , circular_mean_weights)
    
    def update_last_contact_time(self, geoms, contacts, dt_opt, terminal_geoms, leg_sets):
        # print(leg_sets)
        for i in range(len(leg_sets)): # For each leg set.
            for j in range(len(leg_sets[i])): # For each leg in set i.
                
                geom_id = geoms(terminal_geoms[leg_sets[i][j]]).id
                self.last_contact_times[leg_sets[i][j]] += dt_opt # Note that values in self.last_contact_times are non-negative.
                
                for contact in contacts: # For each contact.
                    # if((contact.geom1 == geom_id and contact.geom2 == floor_geom_id) or (contact.geom2 == geom_id and contact.geom1 == floor_geom_id)): # If the current geom is included in the current contact.
                    if(contact.geom1 == geom_id or contact.geom2 == geom_id): # If the current geom is included in the current contact.
                        self.last_contact_times[leg_sets[i][j]] = 0
                        break

    def weighted_circular_mean(self, angles, weights):
        
        # Angles must be given in radians.
        # Weight should sum up to 1.
        
        x = y = 0
        for angle, weight in zip(angles, weights):
            x += np.cos(angle) * weight
            y += np.sin(angle) * weight
        
        return np.arctan2(y, x) % (2 * np.pi)