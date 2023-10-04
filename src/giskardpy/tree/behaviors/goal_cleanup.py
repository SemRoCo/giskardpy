import rospy
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class GoalCleanUp(GiskardBehavior):
    # @catch_and_raise_to_blackboard
    def update(self):
        for goal in god_map.motion_goal_manager.motion_goals.values():
            goal.clean_up()
        return Status.SUCCESS
