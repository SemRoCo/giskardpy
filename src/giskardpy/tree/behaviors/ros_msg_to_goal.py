import traceback
import giskard_msgs.msg as giskard_msgs
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.configs.collision_avoidance_config import CollisionCheckerLib
from giskardpy.exceptions import InvalidGoalException
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.logging import loginfo
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class ParseActionGoal(GiskardBehavior):
    @record_time
    @profile
    def __init__(self, name):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        loginfo('Parsing goal message.')
        move_goal: giskard_msgs.MoveGoal = self.god_map.get_data(identifier.goal_msg)
        self.god_map.set_data(identifier.goal_id, self.goal_id + 1)
        try:
            self.monitor_manager.parse_monitors(move_goal.monitors)
            self.motion_goal_manager.parse_motion_goals(move_goal.goals)
        except AttributeError:
            traceback.print_exc()
            raise InvalidGoalException('Couldn\'t transform goal')
        except Exception as e:
            raise e
        if self.god_map.get_data(identifier.collision_checker) != CollisionCheckerLib.none:
            self.motion_goal_manager.parse_collision_entries(move_goal.collisions)
        loginfo('Done parsing goal message.')
        return Status.SUCCESS
