from py_trees import Status

from giskardpy.exceptions import PathCollisionException, InsolvableException
from giskardpy.identifier import time_identifier, closest_point_identifier, js_identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import closest_point_constraint_violated


class WiggleCancel(GiskardBehavior):
    def __init__(self, name, wiggle_precision_threshold):
        self.wiggle_precision = wiggle_precision_threshold
        super(WiggleCancel, self).__init__(name)

    def initialise(self):
        self.past_joint_states = set()
        super(WiggleCancel, self).initialise()

    def update(self):
        current_js = self.get_god_map().safe_get_data(js_identifier)
        time = self.get_god_map().safe_get_data(time_identifier)
        rounded_js = self.round_js(current_js)
        # TODO make 1 a parameter
        if time >= 1 and rounded_js in self.past_joint_states:
            # TODO raise to blackboard and return failure?
            raise InsolvableException(u'endless wiggling detected')
        self.past_joint_states.add(rounded_js)
        return Status.RUNNING

    def round_js(self, js):
        """
        :param js: joint_name -> SingleJointState
        :type js: dict
        :return: a sequence of all the rounded joint positions
        :rtype: tuple
        """
        return tuple(round(x.position, self.wiggle_precision) for x in js.values())


class MaxTrajLength(GiskardBehavior):
    pass


class CollisionCancel(GiskardBehavior):
    def __init__(self, name, collision_time_threshold):
        self.collision_time_threshold = collision_time_threshold
        super(CollisionCancel, self).__init__(name)

    def update(self):
        time = self.get_god_map().safe_get_data(time_identifier)
        if time >= self.collision_time_threshold:
            cp = self.get_god_map().safe_get_data(closest_point_identifier)
            if cp is not None and closest_point_constraint_violated(cp, tolerance=1):
                self.raise_to_blackboard(PathCollisionException(
                    u'robot is in collision after {} seconds'.format(self.collision_time_threshold)))
                return Status.SUCCESS
        return Status.FAILURE
