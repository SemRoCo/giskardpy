from typing import Tuple, Dict, Optional

from line_profiler import profile
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.model.joints import Joint6DOF
from giskardpy.data_types.data_types import PrefixName
from giskardpy_ros.ros1 import msg_converter
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.ros1.tfwrapper import lookup_pose
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard


class SyncTfFrames(GiskardBehavior):
    joint_map: Dict[PrefixName, Tuple[str, str]]

    @profile
    def __init__(self, name, joint_map: Optional[Dict[PrefixName, Tuple[str, str]]] = None):
        super().__init__(name)
        if joint_map is None:
            self.joint_map = {}
        else:
            self.joint_map = joint_map

    def sync_6dof_joint_with_tf_frame(self, joint_name: PrefixName, tf_parent_frame: str, tf_child_frame: str):
        if joint_name in self.joint_map:
            raise AttributeError(f'Joint \'{joint_name}\' is already being tracking with a tf frame: '
                                 f'\'{self.joint_map[joint_name][0]}\'<-\'{self.joint_map[joint_name][1]}\'')
        joint = god_map.world.joints[joint_name]
        if not isinstance(joint, Joint6DOF):
            raise AttributeError(f'Can only sync Joint6DOF with tf but \'{joint_name}\' is of type \'{type(joint)}\'.')
        self.joint_map[joint_name] = (tf_parent_frame, tf_child_frame)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        for joint_name, (tf_parent_frame, tf_child_frame) in self.joint_map.items():
            joint: Joint6DOF = god_map.world.joints[joint_name]
            parent_T_child = lookup_pose(tf_parent_frame, tf_child_frame)
            pose = msg_converter.ros_msg_to_giskard_obj(parent_T_child, god_map.world)
            joint.update_transform(pose)

        return Status.SUCCESS
