from py_trees import Status

from giskardpy_ros.tree.behaviors.action_server import ActionServerHandler
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.middleware import middleware


class GoalReceived(GiskardBehavior):
    def __init__(self, action_server: ActionServerHandler):
        name = f'has \'{action_server.name}\' goal?'
        self.action_server = action_server
        super().__init__(name)

    @profile
    def update(self):
        if self.action_server.has_goal():
            middleware.loginfo(f'Received new goal.')
            self.action_server.accept_goal()
            return Status.SUCCESS
        return Status.FAILURE
