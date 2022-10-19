from giskardpy import identifier
from giskardpy.goals.goal import Goal
from giskardpy.god_map import GodMap
from giskardpy.utils import logging


class SetPredictionHorizon(Goal):
    def __init__(self, prediction_horizon, **kwargs):
        super().__init__(**kwargs)
        self.new_prediction_horizon = prediction_horizon

    def make_constraints(self):
        self.prediction_horizon = self.new_prediction_horizon
        if 5 > self.prediction_horizon > 1:
            logging.logwarn('Prediction horizon should be 1 or greater equal 5.')
        if self.prediction_horizon == 1:
            if 'acceleration' in self.god_map.get_data(identifier.joint_weights):
                del self.god_map.get_data(identifier.joint_weights)['acceleration']
            if 'acceleration' in self.god_map.get_data(identifier.joint_limits):
                del self.god_map.get_data(identifier.joint_limits)['acceleration']
        if self.prediction_horizon <= 2:
            if 'jerk' in self.god_map.get_data(identifier.joint_weights):
                del self.god_map.get_data(identifier.joint_weights)['jerk']
            if 'jerk' in self.god_map.get_data(identifier.joint_limits):
                del self.god_map.get_data(identifier.joint_limits)['jerk']
        if self.prediction_horizon <= 3:
            if 'jerk' in self.god_map.get_data(identifier.joint_weights):
                del self.god_map.get_data(identifier.joint_weights)['snap']
            if 'jerk' in self.god_map.get_data(identifier.joint_limits):
                del self.god_map.get_data(identifier.joint_limits)['snap']
        self.god_map.set_data(identifier.prediction_horizon, self.prediction_horizon)
        self.world.sync_with_paramserver()


class SetMaxTrajLength(Goal):
    def __init__(self, new_length: int, **kwargs):
        super().__init__(**kwargs)
        self.god_map.set_data(identifier.MaxTrajectoryLength + ['length'], new_length)


class EnableVelocityTrajectoryTracking(Goal):
    def __init__(self, enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.god_map.set_data(identifier.fill_trajectory_velocity_values, enabled)