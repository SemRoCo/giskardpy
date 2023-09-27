from py_trees import Sequence

from giskard_msgs.msg import MoveFeedback
from giskardpy.god_map_user import GodMap
from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy.tree.behaviors.cleanup import CleanUpPlanning
from giskardpy.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.behaviors.exception_to_execute import ClearBlackboardException
from giskardpy.tree.behaviors.goal_cleanup import GoalCleanUp
from giskardpy.tree.behaviors.init_qp_controller import InitQPController
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.new_trajectory import NewTrajectory
from giskardpy.tree.behaviors.notify_state_change import NotifyStateChange
from giskardpy.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy.tree.behaviors.ros_msg_to_goal import ParseActionGoal
from giskardpy.tree.behaviors.set_move_result import SetMoveResult
from giskardpy.tree.behaviors.sync_configuration import SyncConfiguration
from giskardpy.tree.behaviors.sync_configuration2 import SyncConfiguration2
from giskardpy.tree.behaviors.sync_odometry import SyncOdometry
from giskardpy.tree.behaviors.sync_tf_frames import SyncTfFrames
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.behaviors.world_updater import WorldUpdater
from giskardpy.tree.decorators import success_is_failure, running_is_success


class Synchronization(Sequence):
    sync_tf_frames: SyncTfFrames
    collision_scene_updater: CollisionSceneUpdater

    def __init__(self, suffix: str = ''):
        super().__init__(f'synchronize{suffix}')
        self.sync_tf_frames = SyncTfFrames('sync tf frames1')
        GodMap.collision_scene_updater = CollisionSceneUpdater('update collision scene')
        self.add_child(self.sync_tf_frames)
        self.add_child(GodMap.collision_scene_updater)
        self.add_child(NotifyStateChange())

    def sync_6dof_joint_with_tf_frame(self, joint_name: PrefixName, tf_parent_frame: str, tf_child_frame: str):
        self.sync_tf_frames.sync_6dof_joint_with_tf_frame(joint_name, tf_parent_frame, tf_child_frame)

    def sync_joint_state_topic(self, group_name: str, topic_name: str):
        behavior = SyncConfiguration(group_name=group_name, joint_state_topic=topic_name)
        self.insert_child(child=behavior, index=1)

    def sync_joint_state2_topic(self, group_name: str, topic_name: str):
        behavior = SyncConfiguration2(group_name=group_name, joint_state_topic=topic_name)
        self.insert_child(child=behavior, index=1)

    def sync_odometry_topic(self, topic_name: str, joint_name: PrefixName):
        behavior = SyncOdometry(topic_name, joint_name)
        self.insert_child(child=behavior, index=1)
