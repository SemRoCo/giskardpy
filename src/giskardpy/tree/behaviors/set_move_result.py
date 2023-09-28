from py_trees import Status

import giskardpy.identifier as identifier
from giskard_msgs.msg import MoveResult
from giskardpy.exceptions import *
from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.control_modes import ControlModes
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
            move_result = e.to_move_result()
        else:
            move_result = GiskardException(str(e)).to_move_result()

        if isinstance(e, EmptyProblemException) and god_map.is_standalone():
            move_result = MoveResult()

        trajectory = god_map.get_data(identifier.trajectory)
        joints = [god_map.world.joints[joint_name] for joint_name in god_map.world.movable_joint_names]
        sample_period = god_map.get_data(identifier.sample_period)
        move_result.trajectory = trajectory.to_msg(sample_period=sample_period, start_time=0, joints=joints)
        if move_result.error_code == MoveResult.PREEMPTED:
            logging.logwarn(f'Goal preempted: \'{move_result.error_message}\'.')
        else:
            if self.print:
                if move_result.error_code == MoveResult.SUCCESS:
                    logging.loginfo(f'{self.context} succeeded.')
                else:
                    logging.logwarn(f'{self.context} failed: {move_result.error_message}.')
        god_map.set_data(identifier.result_message, move_result)
        return Status.SUCCESS
