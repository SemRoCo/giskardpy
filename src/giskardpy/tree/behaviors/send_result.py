from py_trees import Blackboard, Status

from giskard_msgs.msg import MoveResult
from giskardpy import identifier
from giskardpy.god_map_user import GodMap
from giskardpy.tree.behaviors.action_server import ActionServerBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time


class SendResult(ActionServerBehavior):
    @record_time
    @profile
    def update(self):
        Blackboard().set('exception', None)  # FIXME move this to reset?
        result: MoveResult = GodMap.god_map.get_data(identifier.result_message)

        if result.error_code == MoveResult.PREEMPTED:
            logging.logerr('Goal preempted')
            self.get_as().send_preempted(result)
            return Status.SUCCESS
        if result.error_code != MoveResult.SUCCESS:
            logging.logwarn('Failed to execute goal.')
            self.get_as().send_aborted(result)
            return Status.SUCCESS
        else:
            logging.loginfo('----------------Successfully executed goal.----------------')
        self.get_as().send_result(result)
        return Status.SUCCESS
