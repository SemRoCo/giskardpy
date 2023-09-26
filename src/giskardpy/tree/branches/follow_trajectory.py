from py_trees import Sequence

from giskard_msgs.msg import MoveFeedback
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy.tree.behaviors.cleanup import CleanUpPlanning
from giskardpy.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.behaviors.exception_to_execute import ClearBlackboardException
from giskardpy.tree.behaviors.goal_cleanup import GoalCleanUp
from giskardpy.tree.behaviors.init_qp_controller import InitQPController
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.new_trajectory import NewTrajectory
from giskardpy.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy.tree.behaviors.ros_msg_to_goal import ParseActionGoal
from giskardpy.tree.behaviors.set_move_result import SetMoveResult
from giskardpy.tree.behaviors.set_tracking_start_time import SetTrackingStartTime
from giskardpy.tree.behaviors.sync_configuration import SyncConfiguration
from giskardpy.tree.behaviors.sync_odometry import SyncOdometry
from giskardpy.tree.behaviors.sync_tf_frames import SyncTfFrames
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.behaviors.world_updater import WorldUpdater
from giskardpy.tree.decorators import success_is_failure, running_is_success


class FollowTrajectory(Sequence, GodMapWorshipper):
    world_updater: WorldUpdater
    sync_tf_frames: SyncTfFrames
    collision_scene_updater: CollisionSceneUpdater

    def __init__(self, name: str = 'follow trajectory'):
        super().__init__(name)
        self.add_child(IF('execute?', identifier.execute))
        self.add_child(SetTrackingStartTime('start start time'))
        self.add_child(self.grow_monitor_execution())
        self.add_child(SetZeroVelocity('set zero vel 2'))
