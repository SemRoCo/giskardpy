from py_trees import Status

from giskardpy.plugin import GiskardBehavior


class CleanUp(GiskardBehavior):
    def __init__(self, name, closest_point_identifier):
        super(CleanUp, self).__init__(name)
        self.closest_point_identifier = closest_point_identifier

    def initialise(self):
        self.god_map.safe_set_data([self.closest_point_identifier], None)

    def update(self):
        return Status.SUCCESS