from py_trees import Status

from giskardpy.exceptions import PathCollisionException, InsolvableException
from giskardpy.plugin import GiskardBehavior, NewPluginBase
from giskardpy.utils import closest_point_constraint_violated


class WiggleCancel(NewPluginBase):
    def __init__(self, wiggle_precision_threshold, joint_state_identifier, time_identifier):
        self.joint_state_identifier = joint_state_identifier
        self.wiggle_precision = wiggle_precision_threshold
        self.time_identifier = time_identifier
        super(WiggleCancel, self).__init__()

    def initialize(self):
        self.past_joint_states = set()
        super(WiggleCancel, self).initialize()

    def update(self):
        current_js = self.god_map.safe_get_data([self.joint_state_identifier])
        time = self.get_god_map().safe_get_data([self.time_identifier])
        rounded_js = self.round_js(current_js)
        # TODO make 1 a parameter
        if time >= 1 and rounded_js in self.past_joint_states:
            raise InsolvableException(u'endless wiggling detected')
        self.past_joint_states.add(rounded_js)
        return super(WiggleCancel, self).update()

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
    def __init__(self, name, collision_time_threshold, time_identifier, closest_point_identifier):
        self.collision_time_threshold = collision_time_threshold
        self.time_identifier = time_identifier
        self.closest_point_identifier = closest_point_identifier
        super(CollisionCancel, self).__init__(name)

    def setup(self, timeout):
        return super(CollisionCancel, self).setup(timeout)

    def initialise(self):
        # TODO figure out a clean why of deciding how is responsible for godmap cleanup
        self.god_map.safe_set_data([self.closest_point_identifier], None)
        super(CollisionCancel, self).initialise()

    def update(self):
        time = self.get_god_map().safe_get_data([self.time_identifier])
        if time >= self.collision_time_threshold:
            cp = self.god_map.safe_get_data([self.closest_point_identifier])
            if cp is not None and closest_point_constraint_violated(cp, tolerance=1):
                self.raise_to_blackboard(PathCollisionException(
                    u'robot is in collision after {} seconds'.format(self.collision_time_threshold)))
                return Status.SUCCESS
        return Status.FAILURE
