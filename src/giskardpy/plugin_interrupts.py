from py_trees import Status

from giskardpy.exceptions import PathCollisionException
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import closest_point_constraint_violated


class WiggleCancel(GiskardBehavior):
    pass


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
        super(CollisionCancel, self).initialise()

    def update(self):
        time = self.get_god_map().safe_get_data([self.time_identifier])
        if time >= self.collision_time_threshold:
            cp = self.god_map.safe_get_data([self.closest_point_identifier])
            if closest_point_constraint_violated(cp, tolerance=1):
                self.raise_to_blackboard(PathCollisionException(
                    u'robot is in collision after {} seconds'.format(self.collision_time_threshold)))
                return Status.SUCCESS
        return Status.FAILURE
