from typing import List

from py_trees import Sequence, Status

from giskardpy.god_map import god_map
from giskardpy.data_types.data_types import PrefixName
from giskardpy_ros.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy_ros.tree.behaviors.notify_state_change import NotifyStateChange
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.behaviors.sync_joint_state import SyncJointState, SyncJointStatePosition
from giskardpy_ros.tree.behaviors.sync_odometry import SyncOdometry, SyncOdometryNoLock
from giskardpy_ros.tree.behaviors.sync_tf_frames import SyncTfFrames
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard


class Synchronization(Sequence):
    sync_tf_frames: SyncTfFrames
    collision_scene_updater: CollisionSceneUpdater
    added_behaviors: List[GiskardBehavior]

    def __init__(self):
        super().__init__('synchronize')
        self.sync_tf_frames = None
        self.collision_scene_updater = CollisionSceneUpdater()
        self.add_child(NotifyStateChange())
        self.add_child(self.collision_scene_updater)
        self.added_behaviors = []

    def _number_of_synchronisation_behaviors(self) -> int:
        """
        this is only for testing
        """
        result = 0
        for child in self.children:
            if isinstance(child, SyncJointState):
                result += 1
        return result

    def sync_6dof_joint_with_tf_frame(self, joint_name: PrefixName, tf_parent_frame: str, tf_child_frame: str):
        if self.sync_tf_frames is None:
            self.sync_tf_frames = SyncTfFrames('sync tf frames1')
            self.insert_child(self.sync_tf_frames, index=0)
        self.sync_tf_frames.sync_6dof_joint_with_tf_frame(joint_name, tf_parent_frame, tf_child_frame)

    def sync_joint_state_topic(self, group_name: str, topic_name: str):
        behavior = SyncJointState(group_name=group_name, joint_state_topic=topic_name)
        if GiskardBlackboard().tree.has_started():
            behavior.setup()
            self.added_behaviors.append(behavior)
        self.insert_child(child=behavior, index=0)

    def sync_joint_state2_topic(self, group_name: str, topic_name: str):
        behavior = SyncJointStatePosition(group_name=group_name, joint_state_topic=topic_name)
        self.insert_child(child=behavior, index=0)

    def sync_odometry_topic(self, topic_name: str, joint_name: PrefixName):
        behavior = SyncOdometry(topic_name, joint_name)
        self.insert_child(child=behavior, index=0)

    def sync_odometry_topic_no_lock(self, topic_name: str, joint_name: PrefixName):
        behavior = SyncOdometryNoLock(topic_name, joint_name)
        self.insert_child(child=behavior, index=0)

    def remove_group_behaviors(self, group_name: str):
        # FIXME this only handles added sync joint state behaviors
        for behavior in list(self.added_behaviors):
            if isinstance(behavior, SyncJointState) and behavior.group_name == group_name:
                behavior.terminate(Status.SUCCESS)
                self.remove_child(behavior)
                self.added_behaviors.remove(behavior)

    def remove_added_behaviors(self):
        for behavior in self.added_behaviors:
            behavior.terminate(Status.SUCCESS)
            self.remove_child(behavior)
        self.added_behaviors = []
