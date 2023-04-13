import numpy as np
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.my_types import Derivatives
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time


# fast

class GoalReached(GiskardBehavior):
    @profile
    def __init__(self, name):
        super().__init__(name)
        self.window_size = self.get_god_map().get_data(identifier.GoalReached_window_size)
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)

    @profile
    def initialise(self):
        self.above_threshold_time = 0
        self.thresholds = self.make_velocity_threshold()
        self.number_of_controlled_joints = len(self.thresholds)

    @record_time
    @profile
    def update(self):
        planning_time = self.get_god_map().get_data(identifier.time)
        if planning_time - self.above_threshold_time >= self.window_size:
            velocities = np.array(list(self.get_god_map().get_data(identifier.qp_solver_solution).xdot_velocity))
            below_threshold = np.all(np.abs(velocities) < self.thresholds)
            if below_threshold:
                run_time = self.get_runtime()
                logging.loginfo('Velocities went below threshold.')
                logging.loginfo(f'Found goal trajectory with length '
                                f'{planning_time * self.sample_period:.3f}s in {run_time:.3f}s')
                return Status.SUCCESS
        return Status.RUNNING

    def make_velocity_threshold(self, min_cut_off=0.01, max_cut_off=0.06):
        joint_convergence_threshold = self.god_map.get_data(identifier.joint_convergence_threshold)
        free_variables = self.god_map.get_data(identifier.free_variables)
        thresholds = []
        for free_variable in free_variables:  # type: FreeVariable
            velocity_limit = self.god_map.evaluate_expr(free_variable.get_upper_limit(Derivatives.velocity))
            velocity_limit *= joint_convergence_threshold
            velocity_limit = min(max(min_cut_off, velocity_limit), max_cut_off)
            thresholds.append(velocity_limit)
        return np.array(thresholds)