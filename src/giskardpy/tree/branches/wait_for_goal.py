from py_trees import Sequence

from giskard_msgs.msg import MoveFeedback, MoveAction
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy.tree.behaviors.cleanup import CleanUpPlanning
from giskardpy.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.behaviors.exception_to_execute import ClearBlackboardException
from giskardpy.tree.behaviors.goal_cleanup import GoalCleanUp
from giskardpy.tree.behaviors.goal_received import GoalReceived
from giskardpy.tree.behaviors.init_qp_controller import InitQPController
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.new_trajectory import NewTrajectory
from giskardpy.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy.tree.behaviors.ros_msg_to_goal import ParseActionGoal
from giskardpy.tree.behaviors.set_move_result import SetMoveResult
from giskardpy.tree.behaviors.sync_tf_frames import SyncTfFrames
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.behaviors.world_updater import WorldUpdater
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.synchronization import Synchronization
from giskardpy.tree.decorators import success_is_failure, running_is_success


class WaitForGoal(Sequence):
    synchronization: Synchronization
    publish_state: PublishState
    goal_received: GoalReceived
    world_updater: WorldUpdater

    def __init__(self, name: str = 'wait for goal'):
        super().__init__(name)
        self.world_updater = WorldUpdater('update world')
        self.synchronization = Synchronization('sync 1')
        self.publish_state = PublishState()
        self.goal_received = GoalReceived('has goal?',
                                          god_map.giskard.action_server_name,
                                          MoveAction)
        self.add_child(self.world_updater)
        self.add_child(self.synchronization)
        self.add_child(self.publish_state)
        self.add_child(self.goal_received)

