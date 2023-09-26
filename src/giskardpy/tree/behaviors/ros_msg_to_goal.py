import json
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple
import giskard_msgs.msg as giskard_msgs
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.configs.collision_avoidance_config import CollisionCheckerLib
from giskardpy.exceptions import UnknownConstraintException, InvalidGoalException, \
    ConstraintInitalizationException, GiskardException
from giskardpy.goals.collision_avoidance import SelfCollisionAvoidance, ExternalCollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.get_goal import GetGoal
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import convert_dictionary_to_ros_message, get_all_classes_in_package, raise_to_blackboard, \
    json_to_kwargs
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
            raise_to_blackboard(InvalidGoalException('Couldn\'t transform goal'))
            traceback.print_exc()
            return Status.SUCCESS
        except Exception as e:
            raise_to_blackboard(e)
            # traceback.print_exc()
            return Status.SUCCESS
        if self.god_map.get_data(identifier.collision_checker) != CollisionCheckerLib.none:
            self.motion_goal_manager.parse_collision_entries(move_goal.collisions)
        loginfo('Done parsing goal message.')
        return Status.SUCCESS
