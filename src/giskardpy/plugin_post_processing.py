import numpy as np
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import InsolvableException, UnreachableException
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import make_filter_b_mask


class PostProcessing(GiskardBehavior):

    def __init__(self, name):
        self.threshold = 0.001
        super(PostProcessing, self).__init__(name)

    def setup(self, timeout):
        # return True
        return super(PostProcessing, self).setup(timeout)

    def initialise(self):
        super(PostProcessing, self).initialise()

    def update(self):
        e = self.get_blackboard_exception()
        if isinstance(e, InsolvableException):
            return Status.FAILURE

        if not self.check_reachability_xdot():
            return Status.FAILURE

        return Status.SUCCESS

    def check_reachability_xdot(self):
        num_joint_constraints = len(self.get_robot().controlled_joints)
        H = self.get_god_map().get_data(identifier.H)
        b_mask = make_filter_b_mask(H)[num_joint_constraints:]
        xdot_keys = np.array(self.get_god_map().get_data(identifier.xdot_keys)[num_joint_constraints:])
        xdot_keys_filtered = xdot_keys[b_mask]
        soft_constraints = self.get_god_map().get_data(identifier.soft_constraint_identifier)
        soft_constraints_filtered = [(i, soft_constraints[i]) for i in xdot_keys_filtered]

        xdotfull = self.get_god_map().get_data(identifier.xdot_full)
        if isinstance(xdotfull, int):
            return True
        controllable_joints = self.get_robot().controlled_joints
        xdot_soft_constraints = xdotfull[len(controllable_joints):]
        if (len(soft_constraints_filtered) != len(xdot_soft_constraints)):
            self.raise_to_blackboard(InsolvableException(u'this is a bug, please open an issue on github'))
            return False
        for i in range(len(soft_constraints_filtered)):
            constraint_name = soft_constraints_filtered[i][0]
            constraint = soft_constraints_filtered[i][1]
            if constraint.goal_constraint and abs(xdot_soft_constraints[i]) > self.threshold:
                self.raise_to_blackboard(
                    UnreachableException(u'soft constraint "{}" is not satisfied; |{}| > {}'.format(
                        constraint_name,
                        xdot_soft_constraints[i],
                        self.threshold
                    )))
                return False
        return True
