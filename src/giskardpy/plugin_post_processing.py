import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.exceptions import InsolvableException
from py_trees import Status
import numpy as np
from giskardpy.plugin_interrupts import detect_wiggling


class PostProcessing(GiskardBehavior):

    def __init__(self, name):
        super(PostProcessing, self).__init__(name)
        #GiskardBehavior.__init__(self, name)

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

        if self.final_wiggle_detection():
            self.raise_to_blackboard(InsolvableException(u'endless wiggling detected'))
            return Status.FAILURE


        return Status.SUCCESS


    def check_reachability_xdot(self):
        num_joint_constraints = len(self.get_robot().controlled_joints)
        H = self.get_god_map().safe_get_data(identifier.H)
        b_mask = np.array(H.sum(axis=1) != 0)[num_joint_constraints:]
        #bA_keys = np.array(self.get_god_map().safe_get_data(identifier.bA_keys)[15:])
        xdot_keys = np.array(self.get_god_map().safe_get_data(identifier.xdot_keys)[num_joint_constraints:])
        xdot_keys_filtered = xdot_keys[b_mask]
        #bA_keys_filtered = bA_keys[b_mask]
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
            if soft_constraints_filtered[i][4] and abs(xdot_soft_constraints[i]) > 0.001:
                return False
        return True


    def final_wiggle_detection(self):
        js_samples = self.get_god_map().safe_get_data(identifier.wiggle_detection_samples)
        js_samples_array = np.array(js_samples.values())
        if len(js_samples_array[0]) < 4: #if there are less than 4 sample points it makes no sense to try to detect wiggling
            return False
        sample_period = self.get_god_map().safe_get_data(identifier.sample_period)
        wiggle_frequency_range = self.get_god_map().safe_get_data(identifier.wiggle_frequency_range)
        wiggle_detection_threshold = self.get_god_map().safe_get_data(identifier.wiggle_detection_threshold)
        max_detectable_freq = 1 / (2 * sample_period)
        min_wiggle_frequency = wiggle_frequency_range * max_detectable_freq
        return detect_wiggling(js_samples_array, sample_period, min_wiggle_frequency, wiggle_detection_threshold)