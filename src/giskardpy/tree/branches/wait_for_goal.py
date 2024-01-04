from py_trees import Sequence

from giskard_msgs.msg import MoveAction
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.goal_received import GoalReceived
from giskardpy.tree.behaviors.world_updater import WorldUpdater
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.synchronization import Synchronization


class WaitForGoal(Sequence):
    synchronization: Synchronization
    publish_state: PublishState
    goal_received: GoalReceived
    world_updater: WorldUpdater

    def __init__(self, name: str = 'wait for goal'):
        super().__init__(name)
        self.world_updater = WorldUpdater('update world')
        self.synchronization = Synchronization()
        self.publish_state = PublishState()
        self.goal_received = GoalReceived('has goal?',
                                          god_map.giskard.action_server_name,
                                          MoveAction)
        self.add_child(self.world_updater)
        self.add_child(self.synchronization)
        self.add_child(self.publish_state)
        self.add_child(self.goal_received)

