import numpy as np
from time import time

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy import logging

#fast

class GoalReachedPlugin(GiskardBehavior):
    def __init__(self, name, window_size=None):
        super(GoalReachedPlugin, self).__init__(name)
        if window_size is None:
            self.window_size = self.get_god_map().get_data(identifier.sample_period) * 5
        else:
            self.window_size = window_size

        self.above_threshold_time = 0
        self.joint_convergence_threshold = self.get_god_map().get_data(identifier.joint_convergence_threshold)

    def update(self):
        current_js = self.get_god_map().get_data(identifier.joint_states)
        sample_period = self.get_god_map().get_data(identifier.sample_period)
        planning_time = self.get_god_map().get_data(identifier.time) * sample_period

        below_threshold = np.abs([v.velocity for v in current_js.values()]).max() < self.joint_convergence_threshold
        if below_threshold and planning_time - self.above_threshold_time >= self.window_size:
            logging.loginfo(u'found goal trajectory with length {}s in {}s'.format(planning_time,
                                                                         time() - self.get_blackboard().runtime))
            return Status.SUCCESS
        if not below_threshold:
            self.above_threshold_time = planning_time
        return Status.RUNNING


    def debug_print(self):
        import pandas as pd
        qp_data = self.get_god_map().get_data(identifier.qp_data)
        np_H = qp_data[identifier.H[-1]]
        np_A = qp_data[identifier.A[-1]]
        np_lb = qp_data[identifier.lb[-1]]
        np_ub = qp_data[identifier.ub[-1]]
        np_lbA = qp_data[identifier.lbA[-1]]
        np_ubA = qp_data[identifier.ubA[-1]]
        xdot_full = qp_data[identifier.xdot_full[-1]]
        A_dot_x = np_A.dot(xdot_full)

        lb = qp_data[identifier.b_keys[-1]]
        lbA = qp_data[identifier.bA_keys[-1]]
        weights = qp_data[identifier.weight_keys[-1]]
        xdot = qp_data[identifier.xdot_keys[-1]]

        num_j = len(lb)
        num_s = len(np_lb) - num_j
        num_h = len(lbA) - num_s

        p_lb = pd.DataFrame(np_lb[:num_j], lb).sort_index()
        p_ub = pd.DataFrame(np_ub[:num_j], lb).sort_index()
        p_lbA = pd.DataFrame(np_lbA, lbA).sort_index()
        p_A_dot_x = pd.DataFrame(A_dot_x, lbA).sort_index()
        p_ubA = pd.DataFrame(np_ubA, lbA).sort_index()
        p_weights = pd.DataFrame(np_H.dot(np.ones(np_H.shape[0])), weights).sort_index()
        p_xdot = pd.DataFrame(xdot_full, xdot).sort_index()
        p_A = pd.DataFrame(np_A, lbA, weights).sort_index(1).sort_index(0)
            # self.lbAs.T[[c for c in self.lbAs.T.columns if 'dist' in c]].plot()
        # arrays = [(p_weights, u'H'),
        #           (p_A, u'A'),
        #           (p_lbA, u'lbA'),
        #           (p_ubA, u'ubA'),
        #           (p_lb, u'lb'),
        #           (p_ub, u'ub')]
        # for a, name in arrays:
        #     self.check_for_nan(name, a)
        #     self.check_for_big_numbers(name, a)
        # print(p_A_dot_x[(p_lbA-1e-10 > p_A_dot_x).values].index)
        slack = p_xdot[num_j:]
        violated_constraints = ((slack * p_weights[num_j:]).abs() > 1e-5).values
        print(u'violated constraints:')
        print(slack[violated_constraints])
        pass

    def check_for_nan(self, name, p_array):
        p_filtered = p_array.apply(lambda x: zip(x.index[x.isnull()].tolist(), x[x.isnull()]), 1)
        p_filtered = p_filtered[p_filtered.apply(lambda x: len(x)) > 0]
        if len(p_filtered) > 0:
            self.print_pandas_array(p_filtered)
            logging.logwarn(u'{} has the following nans:'.format(name))

    def check_for_big_numbers(self, name, p_array, big=1e10):
        # FIXME fails if condition is true on first entry
        p_filtered = p_array.apply(lambda x: zip(x.index[abs(x) > big].tolist(), x[x > big]), 1)
        p_filtered = p_filtered[p_filtered.apply(lambda x: len(x)) > 0]
        if len(p_filtered) > 0:
            logging.logwarn(u'{} has the following big numbers:'.format(name))
            self.print_pandas_array(p_filtered)

    def print_pandas_array(self, array):
        import pandas as pd
        if len(array) > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(array)
