from py_trees import Status

from giskard_msgs.msg import MoveResult
from giskardpy.exceptions import *
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time


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
        elif isinstance(e, GiskardException):
            move_result = MoveResult(error=e.to_error_msg())
        else:
            move_result = MoveResult(error=GiskardException(str(e)).to_error_msg())

        trajectory = god_map.trajectory
        joints = [god_map.world.joints[joint_name] for joint_name in god_map.world.movable_joint_names]
        sample_period = god_map.qp_controller_config.sample_period
        move_result.trajectory = trajectory.to_msg(sample_period=sample_period, start_time=0, joints=joints)
        if move_result.error.code == GiskardError.PREEMPTED:
            logging.logwarn(f'Goal preempted: \'{move_result.error.msg}\'.')
        else:
            if self.print:
                if move_result.error.code == GiskardError.SUCCESS:
                    logging.loginfo(f'{self.context} succeeded.')
                else:
                    logging.logwarn(f'{self.context} failed: {move_result.error.msg}.')
        god_map.move_action_server.result_msg = move_result
        return Status.SUCCESS
