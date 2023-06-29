import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class GoalDone(GiskardBehavior):
    # @catch_and_raise_to_blackboard
    def update(self):
        all_goals_succeeded = None
        for goal in self.god_map.get_data(identifier.goals).values():
            is_done = goal.is_done()
            if is_done is not None:
                if all_goals_succeeded is None:
                    all_goals_succeeded = is_done
                else:
                    all_goals_succeeded = all_goals_succeeded and is_done
        if all_goals_succeeded:
            return Status.SUCCESS
        else:
            return Status.RUNNING
