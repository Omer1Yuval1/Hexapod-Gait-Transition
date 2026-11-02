import numpy as np

class HistoryLogger:
    def __init__(self):
        self._fields = {}

    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, value):
        self._fields[key] = value


    def log_state(self, qpos, qvel, geom_xpos, site_xpos):
        self["geom_xpos"] = np.vstack([self["geom_xpos"], [geom_xpos]])
        self["site_xpos"] = np.vstack([self["site_xpos"], [site_xpos]])
        self["qpos"].append(qpos.copy())
        self["qvel"].append(qvel.copy())

    def log_previous(self, action=None, contact_forces=None):
        if action is not None:
            self["previous_action"] = action.copy()
        if contact_forces is not None:
            self["previous_contact_forces"] = contact_forces.flatten()

    def reset(self, geom_xpos_shape, site_xpos_shape):
        self["geom_xpos"] = np.empty((0,) + geom_xpos_shape)
        self["site_xpos"] = np.empty((0,) + site_xpos_shape)
        self["qpos"] = []
        self["qvel"] = []
        self["previous_action"] = None
        self["previous_contact_forces"] = np.array([])
        