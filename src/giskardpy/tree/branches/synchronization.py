from py_trees import Sequence

from giskardpy.god_map import god_map
from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.behaviors.notify_state_change import NotifyStateChange
from giskardpy.tree.behaviors.sync_joint_state import SyncJointState, SyncJointStatePosition
from giskardpy.tree.behaviors.sync_odometry import SyncOdometry, SyncOdometryNoLock
from giskardpy.tree.behaviors.sync_tf_frames import SyncTfFrames


class Synchronization(Sequence):
    sync_tf_frames: SyncTfFrames
    collision_scene_updater: CollisionSceneUpdater

    def __init__(self):
        super().__init__('synchronize')
        self.sync_tf_frames = None
        self.collision_scene_updater = CollisionSceneUpdater('update collision scene')
        self.add_child(self.collision_scene_updater)
        self.add_child(NotifyStateChange())

    def sync_6dof_joint_with_tf_frame(self, joint_name: PrefixName, tf_parent_frame: str, tf_child_frame: str):
        if self.sync_tf_frames is None:
            self.sync_tf_frames = SyncTfFrames('sync tf frames1')
            self.add_child(self.sync_tf_frames)
        self.sync_tf_frames.sync_6dof_joint_with_tf_frame(joint_name, tf_parent_frame, tf_child_frame)

    def sync_joint_state_topic(self, group_name: str, topic_name: str):
        behavior = SyncJointState(group_name=group_name, joint_state_topic=topic_name)
        self.insert_child(child=behavior, index=1)

    def sync_joint_state2_topic(self, group_name: str, topic_name: str):
        behavior = SyncJointStatePosition(group_name=group_name, joint_state_topic=topic_name)
        self.insert_child(child=behavior, index=1)

    def sync_odometry_topic(self, topic_name: str, joint_name: PrefixName):
        behavior = SyncOdometry(topic_name, joint_name)
        self.insert_child(child=behavior, index=1)

    def sync_odometry_topic_no_lock(self, topic_name: str, joint_name: PrefixName):
        behavior = SyncOdometryNoLock(topic_name, joint_name)
        self.insert_child(child=behavior, index=1)
