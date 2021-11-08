from giskardpy import identifier
from giskardpy.goals.goal import Goal


class SetPredictionHorizon(Goal):
    def __init__(self, prediction_horizon, joint_limits=None, joint_weights=None, **kwargs):
        super(SetPredictionHorizon, self).__init__(**kwargs)
        self.prediction_horizon = prediction_horizon
        if self.prediction_horizon == 1:
            del self.god_map.get_data(identifier.joint_weights)['acceleration']
            del self.god_map.get_data(identifier.joint_limits)['acceleration']
            del self.god_map.get_data(identifier.joint_weights)['jerk']
            del self.god_map.get_data(identifier.joint_limits)['jerk']
        self.god_map.set_data(identifier.prediction_horizon, prediction_horizon)
        self.world.sync_with_paramserver()
