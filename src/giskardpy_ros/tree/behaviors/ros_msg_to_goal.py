import traceback
from typing import List, Union

import genpy
from line_profiler import profile
from py_trees import Status

import giskard_msgs.msg as giskard_msgs
import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import InvalidGoalException, UnknownGoalException, GiskardException, \
    GoalInitalizationException, UnknownMonitorException, MonitorInitalizationException
from giskardpy.goals.base_traj_follower import BaseTrajFollower
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.motion_graph.monitors.monitors import TimeAbove, LocalMinimumReached, EndMotion, CancelMotion
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.decorators import record_time
from giskardpy_ros.ros1.msg_converter import json_str_to_giskard_kwargs
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard, GiskardBlackboard
from line_profiler import profile


class ParseActionGoal(GiskardBehavior):
    @record_time
    @profile
    def __init__(self, name):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        move_goal = GiskardBlackboard().move_action_server.goal_msg
        get_middleware().loginfo(f'Parsing goal #{GiskardBlackboard().move_action_server.goal_id} message.')
        try:
            self.parse_monitors(move_goal.monitors)
            self.parse_motion_goals(move_goal.goals)
        except AttributeError:
            traceback.print_exc()
            raise InvalidGoalException('Couldn\'t transform goal')
        except Exception as e:
            raise e
        self.sanity_check()
        # if god_map.is_collision_checking_enabled():
        #     god_map.motion_goal_manager.parse_collision_entries(move_goal.collisions)
        get_middleware().loginfo('Done parsing goal message.')
        return Status.SUCCESS

    def sanity_check(self) -> None:
        if (not god_map.monitor_manager.has_end_motion_monitor()
                and not god_map.monitor_manager.has_cancel_motion_monitor()):
            get_middleware().logwarn(f'No {EndMotion.__name__} or {CancelMotion.__name__} monitor specified. '
                               f'Motion will not stop unless cancelled externally.')
            return
        if not god_map.monitor_manager.has_end_motion_monitor():
            get_middleware().logwarn(f'No {EndMotion.__name__} monitor specified. Motion can\'t end successfully.')

    @profile
    def parse_monitors(self, monitor_msgs: List[giskard_msgs.MotionGraphNode]):
        for monitor_msg in monitor_msgs:
            try:
                get_middleware().loginfo(f'Adding monitor of type: \'{monitor_msg.class_name}\'')
                C = god_map.monitor_manager.allowed_monitor_types[monitor_msg.class_name]
            except KeyError:
                raise UnknownMonitorException(f'unknown monitor type: \'{monitor_msg.class_name}\'.')
            try:
                kwargs = json_str_to_giskard_kwargs(monitor_msg.kwargs, god_map.world)
                pause_condition = monitor_msg.pause_condition
                end_condition = monitor_msg.end_condition
                monitor_name_to_state_expr = {str(key): value.get_state_expression() for key, value in
                                              god_map.monitor_manager.monitors.items()}
                monitor_name_to_state_expr[monitor_msg.name] = symbol_manager.get_symbol(
                    f'god_map.monitor_manager.state[{len(god_map.monitor_manager.monitors)}]')
                start_condition = god_map.monitor_manager.logic_str_to_expr(monitor_msg.start_condition,
                                                                            default=cas.TrueSymbol,
                                                                            monitor_name_to_state_expr=monitor_name_to_state_expr)
                pause_condition = god_map.monitor_manager.logic_str_to_expr(pause_condition, default=cas.FalseSymbol,
                                                                           monitor_name_to_state_expr=monitor_name_to_state_expr)
                end_condition = god_map.monitor_manager.logic_str_to_expr(end_condition, default=cas.FalseSymbol,
                                                                          monitor_name_to_state_expr=monitor_name_to_state_expr)
                monitor = C(name=monitor_msg.name,
                            start_condition=start_condition,
                            pause_condition=pause_condition,
                            end_condition=end_condition,
                            **kwargs)
                god_map.monitor_manager.add_monitor(monitor)
            except Exception as e:
                traceback.print_exc()
                error_msg = f'Initialization of \'{C.__name__}\' monitor failed: \n {e} \n'
                if not isinstance(e, GiskardException):
                    raise MonitorInitalizationException(error_msg)
                raise e

    @profile
    def parse_motion_goals(self, motion_goals: List[giskard_msgs.MotionGraphNode]):
        for motion_goal in motion_goals:
            try:
                get_middleware().loginfo(
                    f'Adding motion goal of type: \'{motion_goal.class_name}\' named: \'{motion_goal.name}\'')
                C = god_map.motion_goal_manager.allowed_motion_goal_types[motion_goal.class_name]
            except KeyError:
                raise UnknownGoalException(f'unknown constraint {motion_goal.class_name}.')
            try:
                params = json_str_to_giskard_kwargs(motion_goal.kwargs, god_map.world)
                if motion_goal.name == '':
                    motion_goal.name = None
                start_condition = god_map.monitor_manager.logic_str_to_expr(motion_goal.start_condition,
                                                                            default=cas.TrueSymbol)
                pause_condition = god_map.monitor_manager.logic_str_to_expr(motion_goal.pause_condition,
                                                                           default=cas.FalseSymbol)
                end_condition = god_map.monitor_manager.logic_str_to_expr(motion_goal.end_condition,
                                                                          default=cas.FalseSymbol)
                c: Goal = C(name=motion_goal.name,
                            start_condition=start_condition,
                            pause_condition=pause_condition,
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


def get_ros_msgs_constant_name_by_value(ros_msg_class: genpy.Message, value: Union[str, int, float]) -> str:
    for attr_name in dir(ros_msg_class):
        if not attr_name.startswith('_'):
            attr_value = getattr(ros_msg_class, attr_name)
            if attr_value == value:
                return attr_name
    raise AttributeError(f'Message type {ros_msg_class} has no constant that matches {value}.')


class SetExecutionMode(GiskardBehavior):
    @record_time
    @profile
    def __init__(self, name: str = 'set execution mode'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        get_middleware().loginfo(
            f'Goal is of type {get_ros_msgs_constant_name_by_value(type(GiskardBlackboard().move_action_server.goal_msg), GiskardBlackboard().move_action_server.goal_msg.type)}')
        if GiskardBlackboard().move_action_server.is_goal_msg_type_projection():
            GiskardBlackboard().tree.switch_to_projection()
        elif GiskardBlackboard().move_action_server.is_goal_msg_type_execute():
            GiskardBlackboard().tree.switch_to_execution()
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
        god_map.monitor_manager.add_monitor(local_min)

        time_monitor = TimeAbove(threshold=god_map.trajectory.length_in_seconds)
        god_map.monitor_manager.add_monitor(time_monitor)

        end_motion = EndMotion(start_condition=cas.logic_and(local_min.get_state_expression(),
                                                             time_monitor.get_state_expression()))
        god_map.monitor_manager.add_monitor(end_motion)

        goal = BaseTrajFollower(self.joint.name, track_only_velocity=True,
                                end_condition=local_min.get_state_expression())
        goal.connect_end_condition_to_all_tasks(time_monitor.get_state_expression())
        god_map.motion_goal_manager.add_motion_goal(goal)
        god_map.motion_goal_manager.init_task_state()
        return Status.SUCCESS
