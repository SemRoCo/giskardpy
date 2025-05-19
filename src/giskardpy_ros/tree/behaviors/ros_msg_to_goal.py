import traceback
from typing import List, Union, Tuple

import genpy
from py_trees import Status

import giskard_msgs.msg as giskard_msgs
import giskardpy.casadi_wrapper as cas
from giskard_msgs.msg import MoveGoal, MotionStatechartNode
from giskardpy.data_types.exceptions import InvalidGoalException, UnknownGoalException
from giskardpy.motion_statechart.goals.base_traj_follower import BaseTrajFollower
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.monitors.monitors import TimeAbove, LocalMinimumReached, EndMotion, CancelMotion, \
    Monitor
from giskardpy.motion_statechart.tasks.task import Task
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

    def create_and_add_node(self, msg_node: MotionStatechartNode, parsed_kwargs, add_node: bool=True):
        if msg_node.class_name in god_map.motion_statechart_manager.allowed_monitor_types:
            C = god_map.motion_statechart_manager.allowed_monitor_types[msg_node.class_name]
            node: Monitor = C(name=msg_node.name, **parsed_kwargs)
            if add_node:
                god_map.motion_statechart_manager.add_monitor(node)
        elif msg_node.class_name in god_map.motion_statechart_manager.allowed_task_types:
            C = god_map.motion_statechart_manager.allowed_task_types[msg_node.class_name]
            node: Task = C(name=msg_node.name, **parsed_kwargs)
            if add_node:
                god_map.motion_statechart_manager.add_task(node)
        elif msg_node.class_name in god_map.motion_statechart_manager.allowed_goal_types:
            C = god_map.motion_statechart_manager.allowed_goal_types[msg_node.class_name]
            node: Goal = C(name=msg_node.name, **parsed_kwargs)
            if add_node:
                god_map.motion_statechart_manager.add_goal(node)
        else:
            raise UnknownGoalException(f'unknown task type: \'{msg_node.class_name}\'.')
        return node

    def apply_transformation_to_leaf_dicts(self, obj):
        """
        Recursively apply transformation to any dict that is a leaf in a list.
        """
        if isinstance(obj, list):
            return [self.apply_transformation_to_leaf_dicts(item) for item in obj]
        elif isinstance(obj, dict) and 'class_name' in obj:
            msg_node = MotionStatechartNode()
            msg_node.class_name = obj.pop('class_name')
            msg_node.name = obj.pop('name')
            return self.create_and_add_node(msg_node, obj, add_node=False)
        else:
            return obj

    def process_kwargs(self, kwargs):
        if isinstance(kwargs, dict):
            processed = {}
            for key, value in kwargs.items():
                if isinstance(value, list):
                    # Recursively process lists with potential leaf dicts
                    processed[key] = self.apply_transformation_to_leaf_dicts(value)
                else:
                    processed[key] = value
            return processed
        return kwargs

    def parse_motion_graph(self, move_goal: MoveGoal) -> None:
        for msg_node in move_goal.nodes:
            parsed_kwargs = json_str_to_giskard_kwargs(msg_node.kwargs, god_map.world)
            get_middleware().loginfo(f'Adding node of type: \'{msg_node.class_name}\'')

            processed_kwargs = self.process_kwargs(parsed_kwargs)

            node = self.create_and_add_node(msg_node, processed_kwargs)

            node.start_condition = msg_node.start_condition
            node.pause_condition = msg_node.pause_condition
            node.end_condition = msg_node.end_condition
            node.reset_condition = msg_node.reset_condition

        god_map.motion_statechart_manager.parse_conditions()

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

        time_monitor = TimeAbove(threshold=god_map.trajectory.length_in_seconds, name='timeout')
        god_map.motion_statechart_manager.add_monitor(time_monitor)

        end_motion = EndMotion(name='end motion')
        end_motion.start_condition = f'{local_min.name} and {time_monitor.name}'
        god_map.motion_statechart_manager.add_monitor(end_motion)

        goal = BaseTrajFollower(self.joint.name, track_only_velocity=True)
        goal.end_condition = f'{local_min.name}'
        goal.connect_end_condition_to_all_tasks(time_monitor.name)
        god_map.motion_statechart_manager.add_goal(goal)
        god_map.motion_statechart_manager.parse_conditions()
        return Status.SUCCESS
