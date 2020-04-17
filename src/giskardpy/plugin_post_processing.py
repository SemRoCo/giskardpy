import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.exceptions import InsolvableException
from py_trees import Status
import numpy as np
from giskardpy.utils import make_filter_b_mask


class PostProcessing(GiskardBehavior):

    def __init__(self, name):
        super(PostProcessing, self).__init__(name)

    def setup(self, timeout):
        #return True
        return super(PostProcessing, self).setup(timeout)

    def initialise(self):
        super(PostProcessing, self).initialise()

    def update(self):
        e = self.get_blackboard_exception()
        if isinstance(e, InsolvableException):
            return Status.FAILURE

        if not self.check_reachability_xdot():
            self.raise_to_blackboard(InsolvableException(u'not all soft constraints were satisfied'))
            return Status.FAILURE

        return Status.SUCCESS


    def check_reachability_xdot(self):
        num_joint_constraints = len(self.get_robot().controlled_joints)
        H = self.get_god_map().safe_get_data(identifier.H)
        b_mask = make_filter_b_mask(H)[num_joint_constraints:]
        xdot_keys = np.array(self.get_god_map().safe_get_data(identifier.xdot_keys)[num_joint_constraints:])
        xdot_keys_filtered = xdot_keys[b_mask]
        soft_constraints = self.get_god_map().safe_get_data(identifier.soft_constraint_identifier)
        soft_constraints_filtered = [soft_constraints[i] for i in xdot_keys_filtered]

        xdotfull = self.get_god_map().safe_get_data(identifier.xdot_full)
        if isinstance(xdotfull, int):
            return True
        controllable_joints = self.get_robot().controlled_joints
        xdot_soft_constraints = xdotfull[len(controllable_joints):]
        if(len(soft_constraints_filtered) != len(xdot_soft_constraints)):
            return False
        for i in range(len(soft_constraints_filtered)):
            if soft_constraints_filtered[i].goal_constraint and abs(xdot_soft_constraints[i]) > 0.001:
                return False
        return True