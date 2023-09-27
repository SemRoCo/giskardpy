import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class GoalCleanUp(GiskardBehavior):
    # @catch_and_raise_to_blackboard
    def update(self):
        for goal in GodMap.god_map.get_data(identifier.motion_goals).values():
            goal.clean_up()
        return Status.SUCCESS
