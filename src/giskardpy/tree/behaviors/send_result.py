from py_trees import Status

from giskard_msgs.msg import MoveResult
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.action_server import ActionServerHandler
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time


class SendResult(GiskardBehavior):

    def __init__(self, action_server: ActionServerHandler):
        self.action_server = action_server
        name = f'send result to \'{action_server.name}\''
        super().__init__(name)

    @record_time
    @profile
    def update(self):
        # result: MoveResult = god_map.result_message

        # if result.error_code == MoveResult.PREEMPTED:
        #     logging.logerr('Goal preempted')
        #     self.action_server.send_preempted()
        #     return Status.SUCCESS
        # if result.error_code != MoveResult.SUCCESS:
        #     logging.logwarn('Failed to execute goal.')
        #     self.action_server.send_aborted()
        #     return Status.SUCCESS
        # else:
        #     logging.loginfo('----------------Successfully executed goal.----------------')
        self.action_server.send_result()
        return Status.SUCCESS
