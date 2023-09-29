import traceback
import giskard_msgs.msg as giskard_msgs
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.configs.collision_avoidance_config import CollisionCheckerLib
from giskardpy.exceptions import InvalidGoalException
from giskardpy.god_map_interpreter import god_map
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
        move_goal = god_map.goal_msg
        god_map.goal_id += 1
        try:
            god_map.monitor_manager.parse_monitors(move_goal.monitors)
            god_map.motion_goal_manager.parse_motion_goals(move_goal.goals)
        except AttributeError:
            traceback.print_exc()
            raise InvalidGoalException('Couldn\'t transform goal')
        except Exception as e:
            raise e
        if god_map.is_collision_checking_enabled():
            god_map.motion_goal_manager.parse_collision_entries(move_goal.collisions)
        loginfo('Done parsing goal message.')
        return Status.SUCCESS
