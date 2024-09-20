from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior


class GoalCleanUp(GiskardBehavior):
    def update(self):
        # fixme
        # for goal in god_map.motion_graph_manager.motion_goals.values():
        #     goal.clean_up()
        return Status.SUCCESS
