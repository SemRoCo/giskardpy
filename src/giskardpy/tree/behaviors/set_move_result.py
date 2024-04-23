from py_trees import Status

from giskard_msgs.msg import MoveResult
from giskardpy.data_types.exceptions import *
from giskardpy.goals.collision_avoidance import CollisionAvoidance
from giskardpy.goals.goal import NonMotionGoal
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.middleware import logging
from giskardpy.utils.decorators import record_time
import giskardpy.middleware.ros1.msg_converter as msg_converter


class SetMoveResult(GiskardBehavior):

    @profile
    def __init__(self, name, context, print=True):
        self.print = print
        self.context = context
        super().__init__(name)

    @record_time
    @profile
    def update(self):
        e = self.get_blackboard_exception()
        if e is None:
            move_result = MoveResult()
        else:
            move_result = MoveResult(error=msg_converter.exception_to_error_msg(e))

        if isinstance(e, EmptyProblemException) and god_map.tree.is_standalone():
            motion_goals = god_map.motion_goal_manager.motion_goals.values()
            non_motion_goals = len([x for x in motion_goals if isinstance(x, NonMotionGoal)])
            collision_avoidance = len([x for x in motion_goals if isinstance(x, CollisionAvoidance)])
            others = len([x for x in motion_goals if not isinstance(x, (NonMotionGoal, CollisionAvoidance))])
            if others == 0:
                if non_motion_goals != 0 and collision_avoidance != 0:
                    # ignore error
                    move_result = MoveResult()

        trajectory = god_map.trajectory
        joints = [god_map.world.joints[joint_name] for joint_name in god_map.world.movable_joint_names]
        sample_period = god_map.qp_controller_config.sample_period
        move_result.trajectory = msg_converter.trajectory_to_ros_trajectory(trajectory,
                                                                            sample_period=sample_period,
                                                                            start_time=0,
                                                                            joints=joints)
        if isinstance(e, PreemptedException):
            logging.logwarn(f'Goal preempted: \'{move_result.error.msg}\'.')
        else:
            if self.print:
                if e is None:
                    logging.loginfo(f'{self.context} succeeded.')
                else:
                    logging.logwarn(f'{self.context} failed: {move_result.error.msg}.')
        god_map.move_action_server.result_msg = move_result
        return Status.SUCCESS
