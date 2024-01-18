from py_trees import Sequence

from giskard_msgs.msg import MoveAction, WorldAction
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.action_server import ActionServerHandler
from giskardpy.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.behaviors.goal_received import GoalReceived
from giskardpy.tree.behaviors.notify_state_change import NotifyStateChange, NotifyModelChange
from giskardpy.tree.behaviors.send_result import SendResult
from giskardpy.tree.behaviors.world_updater import ProcessWorldUpdate
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.synchronization import Synchronization


class UpdateWorld(Sequence):
    synchronization: Synchronization
    publish_state: PublishState
    goal_received: GoalReceived
    process_goal: ProcessWorldUpdate

    def __init__(self):
        name = 'update world'
        super().__init__(name)
        god_map.world_action_server = ActionServerHandler(action_name='~update_world', action_type=WorldAction)
        self.goal_received = GoalReceived(god_map.world_action_server)
        self.send_result = SendResult(action_server=god_map.world_action_server)
        self.process_goal = ProcessWorldUpdate(action_server=god_map.world_action_server)

        self.add_child(self.goal_received)
        self.add_child(self.process_goal)
        self.add_child(NotifyModelChange())
        self.add_child(CollisionSceneUpdater())
        self.add_child(self.send_result)
