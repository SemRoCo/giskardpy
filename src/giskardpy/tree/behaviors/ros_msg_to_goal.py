import traceback
from typing import List

from py_trees import Status

from giskard_msgs.msg import MoveGoal
from giskardpy.exceptions import InvalidGoalException, UnknownGoalException, GiskardException, \
    GoalInitalizationException
from giskardpy.goals.base_traj_follower import BaseTrajFollower
from giskardpy.goals.goal import Goal
from giskardpy.middleware_interfaces.ros1.msg_converter import json_str_to_kwargs
from giskardpy.monitors.monitors import TimeAbove, LocalMinimumReached, EndMotion
from giskardpy.god_map import god_map
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time
from giskardpy.utils.utils import get_ros_msgs_constant_name_by_value
import giskardpy.casadi_wrapper as cas
import giskard_msgs.msg as giskard_msgs


def end_motion_in_move_goal(move_goal: MoveGoal) -> bool:
    for monitor in move_goal.monitors:
        if monitor.monitor_class == EndMotion.__name__:
            return True
    return False


class ParseActionGoal(GiskardBehavior):
    @record_time
    @profile
    def __init__(self, name):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        move_goal = god_map.move_action_server.goal_msg
        loginfo(f'Parsing goal #{god_map.move_action_server.goal_id} message.')
        self.sanity_check(move_goal)
        try:
            god_map.monitor_manager.parse_monitors(move_goal.monitors)
            self.parse_motion_goals(move_goal.goals)
        except AttributeError:
            traceback.print_exc()
            raise InvalidGoalException('Couldn\'t transform goal')
        except Exception as e:
            raise e
        # if god_map.is_collision_checking_enabled():
        #     god_map.motion_goal_manager.parse_collision_entries(move_goal.collisions)
        loginfo('Done parsing goal message.')
        return Status.SUCCESS

    @profile
    def parse_motion_goals(self, motion_goals: List[giskard_msgs.MotionGoal]):
        for motion_goal in motion_goals:
            try:
                logging.loginfo(
                    f'Adding motion goal of type: \'{motion_goal.motion_goal_class}\' named: \'{motion_goal.name}\'')
                C = god_map.motion_goal_manager.allowed_motion_goal_types[motion_goal.motion_goal_class]
            except KeyError:
                raise UnknownGoalException(f'unknown constraint {motion_goal.motion_goal_class}.')
            try:
                params = json_str_to_kwargs(motion_goal.kwargs, god_map.world)
                if motion_goal.name == '':
                    motion_goal.name = None
                start_condition = god_map.monitor_manager.logic_str_to_expr(motion_goal.start_condition,
                                                                            default=cas.TrueSymbol)
                hold_condition = god_map.monitor_manager.logic_str_to_expr(motion_goal.hold_condition,
                                                                           default=cas.FalseSymbol)
                end_condition = god_map.monitor_manager.logic_str_to_expr(motion_goal.end_condition,
                                                                          default=cas.TrueSymbol)
                c: Goal = C(name=motion_goal.name,
                            start_condition=start_condition,
                            hold_condition=hold_condition,
                            end_condition=end_condition,
                            **params)
                god_map.motion_goal_manager.add_motion_goal(c)
            except Exception as e:
                traceback.print_exc()
                error_msg = f'Initialization of \'{C.__name__}\' constraint failed: \n {e} \n'
                if not isinstance(e, GiskardException):
                    raise GoalInitalizationException(error_msg)
                raise e
        god_map.motion_goal_manager.init_task_state()

    def sanity_check(self, move_goal: MoveGoal):
        if not end_motion_in_move_goal(move_goal):
            logging.logwarn(f'No {EndMotion.__name__} monitor.')


class SetExecutionMode(GiskardBehavior):
    @record_time
    @profile
    def __init__(self, name: str = 'set execution mode'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        loginfo(
            f'Goal is of type {get_ros_msgs_constant_name_by_value(type(god_map.move_action_server.goal_msg), god_map.move_action_server.goal_msg.type)}')
        if god_map.is_goal_msg_type_projection():
            god_map.tree.switch_to_projection()
        elif god_map.is_goal_msg_type_execute():
            god_map.tree.switch_to_execution()
        else:
            raise InvalidGoalException(f'Goal of type {god_map.goal_msg.type} is not supported.')
        return Status.SUCCESS


class AddBaseTrajFollowerGoal(GiskardBehavior):
    def __init__(self, name: str = 'add base traj goal'):
        super().__init__(name)
        joints = god_map.world.search_for_joint_of_type((OmniDrive, DiffDrive))
        assert len(joints) == 1
        self.joint = joints[0]

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        local_min = LocalMinimumReached('local min')
        god_map.monitor_manager.add_expression_monitor(local_min)

        time_monitor = TimeAbove(threshold=god_map.trajectory.length_in_seconds)
        god_map.monitor_manager.add_expression_monitor(time_monitor)

        end_motion = EndMotion(start_condition=cas.logic_and(local_min.get_state_expression(),
                                                             time_monitor.get_state_expression()))
        god_map.monitor_manager.add_expression_monitor(end_motion)

        goal = BaseTrajFollower(self.joint.name, track_only_velocity=True,
                                end_condition=local_min.get_state_expression())
        goal.connect_end_condition_to_all_tasks(time_monitor.get_state_expression())
        god_map.motion_goal_manager.add_motion_goal(goal)
        god_map.motion_goal_manager.init_task_state()
        return Status.SUCCESS
