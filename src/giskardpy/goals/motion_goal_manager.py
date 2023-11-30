import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple

from giskardpy.exceptions import UnknownConstraintException, GiskardException, ConstraintInitalizationException, \
    DuplicateNameException
from giskardpy.goals.collision_avoidance import ExternalCollisionAvoidance, SelfCollisionAvoidance
from giskardpy.goals.goal import Goal
import giskard_msgs.msg as giskard_msgs
from giskardpy.god_map import god_map
from giskardpy.my_types import PrefixName
from giskardpy.utils import logging
from giskardpy.utils.utils import get_all_classes_in_package, json_to_kwargs, convert_dictionary_to_ros_message, \
    json_str_to_kwargs, ImmutableDict


class MotionGoalManager:
    motion_goals: Dict[str, Goal] = None

    def __init__(self):
        self.motion_goals = {}
        goal_package_paths = god_map.giskard.goal_package_paths
        self.allowed_motion_goal_types = {}
        for path in goal_package_paths:
            self.allowed_motion_goal_types.update(get_all_classes_in_package(path, Goal))

    @profile
    def parse_motion_goals(self, motion_goals: List[giskard_msgs.MotionGoal]):
        for motion_goal in motion_goals:
            try:
                logging.loginfo(f'Adding motion goal of type: \'{motion_goal.type}\' named: \'{motion_goal.name}\'')
                C = self.allowed_motion_goal_types[motion_goal.type]
            except KeyError:
                raise UnknownConstraintException(f'unknown constraint {motion_goal.type}.')
            try:
                params = json_str_to_kwargs(motion_goal.parameter_value_pair)
                if motion_goal.name == '':
                    motion_goal.name = None
                to_start = [god_map.monitor_manager.get_monitor(monitor_name) for monitor_name in motion_goal.to_start]
                to_hold = [god_map.monitor_manager.get_monitor(monitor_name) for monitor_name in motion_goal.to_hold]
                to_end = [god_map.monitor_manager.get_monitor(monitor_name) for monitor_name in motion_goal.to_end]
                c: Goal = C(name=motion_goal.name, to_start=to_start, to_hold=to_hold, to_end=to_end, **params)
                self.add_motion_goal(c)
            except Exception as e:
                traceback.print_exc()
                error_msg = f'Initialization of \'{C.__name__}\' constraint failed: \n {e} \n'
                if not isinstance(e, GiskardException):
                    raise ConstraintInitalizationException(error_msg)
                raise e

    def add_motion_goal(self, goal: Goal):
        name = goal.name
        if name in self.motion_goals:
            raise DuplicateNameException(f'Motion goal with name {name} already exists.')
        self.motion_goals[name] = goal

    @profile
    def get_constraints_from_goals(self):
        eq_constraints = ImmutableDict()
        neq_constraints = ImmutableDict()
        derivative_constraints = ImmutableDict()
        goals: Dict[str, Goal] = god_map.motion_goal_manager.motion_goals
        for goal_name, goal in list(goals.items()):
            try:
                new_eq_constraints, new_neq_constraints, new_derivative_constraints, _debug_expressions = goal.get_constraints()
            except Exception as e:
                raise ConstraintInitalizationException(str(e))
            eq_constraints.update(new_eq_constraints)
            neq_constraints.update(new_neq_constraints)
            derivative_constraints.update(new_derivative_constraints)
            # logging.loginfo(f'{goal_name} added {len(_constraints)+len(_vel_constraints)} constraints.')
        god_map.eq_constraints = eq_constraints
        god_map.neq_constraints = neq_constraints
        god_map.derivative_constraints = derivative_constraints
        return eq_constraints, neq_constraints, derivative_constraints

    def replace_jsons_with_ros_messages(self, d):
        if isinstance(d, list):
            for i, element in enumerate(d):
                d[i] = self.replace_jsons_with_ros_messages(element)

        if isinstance(d, dict):
            if 'message_type' in d:
                d = convert_dictionary_to_ros_message(d)
            else:
                for key, value in d.copy().items():
                    d[key] = self.replace_jsons_with_ros_messages(value)
        return d
