from py_trees import Blackboard, Status

from giskard_msgs.msg import MoveResult
from giskardpy import identifier
from giskardpy.tree.action_server import ActionServerBehavior
from giskardpy.utils import logging


class SendResult(ActionServerBehavior):
    def __init__(self, name, as_name, action_type=None):
        super(SendResult, self).__init__(name, as_name, action_type)

    def update(self):
        skip_failures = self.get_god_map().get_data(identifier.skip_failures)
        Blackboard().set('exception', None)  # FIXME move this to reset?
        result = self.get_god_map().get_data(identifier.result_message)

        # trajectory = self.get_god_map().get_data(identifier.trajectory)
        # sample_period = self.get_god_map().get_data(identifier.sample_period)
        # controlled_joints = self.get_god_map().get_data(identifier.controlled_joints)
        # result.trajectory = trajectory.to_msg(sample_period, controlled_joints, True)

        if result.error_codes[-1] == MoveResult.PREEMPTED:
            logging.logerr('Goal preempted')
            self.get_as().send_preempted(result)
            return Status.SUCCESS
        if skip_failures:
            if not self.any_goal_succeeded(result):
                self.get_as().send_aborted(result)
                return Status.SUCCESS

        else:
            if not self.all_goals_succeeded(result):
                logging.logwarn('Failed to execute goal.')
                self.get_as().send_aborted(result)
                return Status.SUCCESS
            else:
                logging.loginfo('----------------Successfully executed goal.----------------')
        self.get_as().send_result(result)
        return Status.SUCCESS

    def any_goal_succeeded(self, result):
        """
        :type result: MoveResult
        :rtype: bool
        """
        return MoveResult.SUCCESS in result.error_codes

    def all_goals_succeeded(self, result):
        """
        :type result: MoveResult
        :rtype: bool
        """
        return len([x for x in result.error_codes if x != MoveResult.SUCCESS]) == 0
