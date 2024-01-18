import traceback
from py_trees import Status

from giskard_msgs.msg import MoveGoal
from giskardpy.exceptions import InvalidGoalException
from giskardpy.goals.base_traj_follower import BaseTrajFollower
from giskardpy.monitors.monitors import TimeAbove, LocalMinimumReached
from giskardpy.monitors.payload_monitors import EndMotion
from giskardpy.god_map import god_map
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time
from giskardpy.utils.utils import get_ros_msgs_constant_name_by_value
import giskardpy.casadi_wrapper as cas

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
            god_map.motion_goal_manager.parse_motion_goals(move_goal.goals)
        except AttributeError:
            traceback.print_exc()
            raise InvalidGoalException('Couldn\'t transform goal')
        except Exception as e:
            raise e
        # if god_map.is_collision_checking_enabled():
        #     god_map.motion_goal_manager.parse_collision_entries(move_goal.collisions)
        loginfo('Done parsing goal message.')
        return Status.SUCCESS

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
        loginfo(f'Goal is of type {get_ros_msgs_constant_name_by_value(type(god_map.move_action_server.goal_msg), god_map.move_action_server.goal_msg.type)}')
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
