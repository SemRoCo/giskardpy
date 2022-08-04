import numpy as np
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.utils import logging
from giskardpy.tree.behaviors.plugin import GiskardBehavior


# fast

class GoalReached(GiskardBehavior):
    def __init__(self, name):
        super(GoalReached, self).__init__(name)
        self.window_size = self.get_god_map().get_data(identifier.GoalReached_window_size)
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)

    @profile
    def initialise(self):
        self.above_threshold_time = 0
        self.thresholds = self.make_velocity_threshold()
        self.number_of_controlled_joints = len(self.thresholds)

    @profile
    def update(self):
        planning_time = self.get_god_map().get_data(identifier.time)
        if planning_time - self.above_threshold_time >= self.window_size:
            velocities = np.array(list(self.get_god_map().get_data(identifier.qp_solver_solution)[0].values()))
            below_threshold = np.all(np.abs(velocities) < self.thresholds)
            if below_threshold:
                run_time = self.get_runtime()
                logging.loginfo('Found goal trajectory with length {:.3f}s in {:.3f}s'.format(planning_time * self.sample_period,
                                                                                       run_time))
                self.time_collector.lengths.append(planning_time * self.sample_period)
                self.time_collector.times.append(run_time)
                return Status.SUCCESS
        return Status.RUNNING

    def make_velocity_threshold(self, min_cut_off=0.01, max_cut_off=0.06):
        joint_convergence_threshold = self.god_map.get_data(identifier.joint_convergence_threshold)
        free_variables = self.god_map.get_data(identifier.free_variables)
        thresholds = []
        for free_variable in free_variables:  # type: FreeVariable
            velocity_limit = self.god_map.evaluate_expr(free_variable.get_upper_limit(1))
            velocity_limit *= joint_convergence_threshold
            velocity_limit = min(max(min_cut_off, velocity_limit), max_cut_off)
            thresholds.append(velocity_limit)
        return np.array(thresholds)