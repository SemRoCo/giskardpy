import traceback
from typing import List, Union, Tuple

import genpy
from py_trees import Status

import giskard_msgs.msg as giskard_msgs
import giskardpy.casadi_wrapper as cas
from giskard_msgs.msg import MoveGoal
from giskardpy.data_types.exceptions import InvalidGoalException
from giskardpy.motion_statechart.goals.base_traj_follower import BaseTrajFollower
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.motion_statechart.monitors.monitors import TimeAbove, LocalMinimumReached, EndMotion, CancelMotion
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
        move_goal: MoveGoal = GiskardBlackboard().move_action_server.goal_msg
        get_middleware().loginfo(f'Parsing goal #{GiskardBlackboard().move_action_server.goal_id} message.')
        try:
            self.parse_motion_graph(move_goal)
        except AttributeError:
            traceback.print_exc()
            raise InvalidGoalException('Couldn\'t parse goal msg')
        except Exception as e:
            raise e
        self.sanity_check()
        # if god_map.is_collision_checking_enabled():
        #     god_map.motion_statechart_manager.parse_collision_entries(move_goal.collisions)
        get_middleware().loginfo('Done parsing goal message.')
        return Status.SUCCESS

    def parse_motion_graph(self, move_goal: MoveGoal):
        node_data = self.parse_motion_graph_component(motion_graph_nodes=move_goal.nodes)
        god_map.motion_statechart_manager.parse_motion_graph(nodes=node_data)

    def parse_motion_graph_component(self, motion_graph_nodes: List[giskard_msgs.MotionStatechartNode]) \
            -> List[Tuple[str, str, str, str, str, str, dict]]:
        parsed_nodes = []
        for node in motion_graph_nodes:
            parsed_nodes.append((node.class_name,
                                 node.name,
                                 node.start_condition,
                                 node.reset_condition,
                                 node.pause_condition,
                                 node.end_condition,
                                 json_str_to_giskard_kwargs(node.kwargs, god_map.world)))
        return parsed_nodes

    def sanity_check(self) -> None:
        if (not god_map.motion_statechart_manager.has_end_motion_monitor()
                and not god_map.motion_statechart_manager.has_cancel_motion_monitor()):
            get_middleware().logwarn(f'No {EndMotion.__name__} or {CancelMotion.__name__} monitor specified. '
                                     f'Motion will not stop unless cancelled externally.')
            return
        if not god_map.motion_statechart_manager.has_end_motion_monitor():
            get_middleware().logwarn(f'No {EndMotion.__name__} monitor specified. Motion can\'t end successfully.')


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
        god_map.motion_statechart_manager.add_monitor(local_min)

        time_monitor = TimeAbove(threshold=god_map.trajectory.length_in_seconds)
        god_map.motion_statechart_manager.add_monitor(time_monitor)

        end_motion = EndMotion(start_condition=cas.logic_and(local_min.get_observation_state_expression(),
                                                             time_monitor.get_observation_state_expression()))
        god_map.motion_statechart_manager.add_monitor(end_motion)

        goal = BaseTrajFollower(self.joint.name, track_only_velocity=True,
                                end_condition=local_min.get_observation_state_expression())
        goal.connect_end_condition_to_all_tasks(time_monitor.get_observation_state_expression())
        god_map.motion_statechart_manager.add_motion_goal(goal)
        god_map.motion_statechart_manager.init_task_state()
        return Status.SUCCESS
