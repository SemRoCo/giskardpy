from py_trees import Sequence

from giskard_msgs.msg import MoveAction
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.action_server import ActionServerHandler
from giskardpy.tree.behaviors.goal_received import GoalReceived
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.synchronization import Synchronization
from giskardpy.tree.branches.update_world import UpdateWorld
from giskardpy.tree.decorators import failure_is_success


class WaitForGoal(Sequence):
    synchronization: Synchronization
    publish_state: PublishState
    goal_received: GoalReceived
    world_updater: UpdateWorld

    def __init__(self, name: str = 'wait for goal'):
        super().__init__(name)
        god_map.move_action_server = ActionServerHandler(action_name=god_map.giskard.action_server_name,
                                                         action_type=MoveAction)
        self.world_updater = failure_is_success(UpdateWorld)()
        self.synchronization = Synchronization()
        self.publish_state = PublishState()
        self.goal_received = GoalReceived(action_server=god_map.move_action_server)
        self.add_child(self.world_updater)
        self.add_child(self.synchronization)
        self.add_child(self.publish_state)
        self.add_child(self.goal_received)
