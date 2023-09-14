import numpy as np
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import LocalMinimumException
from giskardpy.my_types import Derivatives
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


# fast

class LocalMinimum(GiskardBehavior):
    @profile
    def __init__(self, name, window_size: int = 21, joint_convergence_threshold: float = 0.01, real_time: bool = False):
        super().__init__(name)
        self.joint_convergence_threshold = joint_convergence_threshold
        self.window_size = window_size
        self.real_time = real_time
        self.last_goal_id = -1
        if real_time:
            self.window_size *= self.sample_period

    @profile
    def initialise(self):
        if self.goal_id > self.last_goal_id:
            self.thresholds = self.make_velocity_threshold()
            self.number_of_controlled_joints = len(self.thresholds)
            self.endless_mode = self.god_map.get_data(identifier.endless_mode)
            self.last_goal_id = self.goal_id

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        if self.endless_mode:
            return Status.RUNNING
        traj_time = self.traj_time_in_sec
        if traj_time >= self.window_size:
            velocities = np.array(list(self.god_map.get_data(identifier.qp_solver_solution).xdot_velocity))
            below_threshold = np.all(np.abs(velocities) < self.thresholds)
            if below_threshold:
                run_time = self.get_runtime()
                logging.loginfo('Velocities went below threshold.')
                logging.loginfo(f'Found goal trajectory with length '
                                f'{traj_time:.3f}s in {run_time:.3f}s')
                raise LocalMinimumException('asdf')
        return Status.RUNNING

    def make_velocity_threshold(self, min_cut_off=0.01, max_cut_off=0.06):
        joint_convergence_threshold = self.joint_convergence_threshold
        free_variables = self.god_map.get_data(identifier.free_variables)
        thresholds = []
        for free_variable in free_variables:  # type: FreeVariable
            velocity_limit = self.god_map.evaluate_expr(free_variable.get_upper_limit(Derivatives.velocity))
            velocity_limit *= joint_convergence_threshold
            velocity_limit = min(max(min_cut_off, velocity_limit), max_cut_off)
            thresholds.append(velocity_limit)
        return np.array(thresholds)