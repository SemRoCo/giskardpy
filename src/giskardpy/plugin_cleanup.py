from py_trees import Status

from giskardpy.identifier import closest_point_identifier
from giskardpy.plugin import GiskardBehavior


class CleanUp(GiskardBehavior):
    def __init__(self, name):
        super(CleanUp, self).__init__(name)

    def initialise(self):
        self.get_god_map().safe_set_data(closest_point_identifier, None)

    def update(self):
        return Status.SUCCESS