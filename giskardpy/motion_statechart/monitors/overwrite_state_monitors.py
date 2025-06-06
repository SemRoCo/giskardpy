from __future__ import division

from typing import Dict, Optional

from giskardpy import casadi_wrapper as cas
from giskardpy.motion_statechart.monitors.monitors import PayloadMonitor
from giskardpy.god_map import god_map
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.model.joints import OmniDrive, DiffDrive, OmniDrivePR22
from giskardpy.data_types.data_types import PrefixName, ObservationState
from giskardpy.utils.math import axis_angle_from_quaternion


class SetSeedConfiguration(PayloadMonitor):
    def __init__(self,
                 seed_configuration: Dict[str, float],
                 group_name: Optional[str] = None,
                 name: Optional[str] = None):
        """
        Overwrite the configuration of the world to allow starting the planning from a different state.
        CAUTION! don't use this to overwrite the robot's state outside standalone mode!
        :param seed_configuration: maps joint name to float
        :param group_name: if joint names are not unique, it will search in this group for matches.
        """
        self.seed_configuration = seed_configuration
        if name is None:
            name = f'{str(self.__class__.__name__)}/{list(self.seed_configuration.keys())}'
        super().__init__(run_call_in_thread=False, name=name)
        self.group_name = group_name
        if group_name is not None:
            self.seed_configuration = {PrefixName(joint_name, group_name): v for joint_name, v in
                                       seed_configuration.items()}

    def __call__(self):
        for joint_name, initial_joint_value in self.seed_configuration.items():
            joint_name = god_map.world.search_for_joint_name(joint_name, self.group_name)
            if joint_name not in god_map.world.state:
                raise KeyError(f'World has no joint \'{joint_name}\'.')
            god_map.world.state[joint_name].position = initial_joint_value
        god_map.world.notify_state_change()
        self.state = ObservationState.true


class SetOdometry(PayloadMonitor):
    odom_joints = (OmniDrive, DiffDrive, OmniDrivePR22)

    def __init__(self,
                 base_pose: cas.TransMatrix,
                 group_name: Optional[str] = None,
                 name: Optional[str] = None):
        self.group_name = group_name
        if name is None:
            name = f'{self.__class__.__name__}/{self.group_name}'
        super().__init__(run_call_in_thread=False, name=name)
        self.base_pose = base_pose
        if self.group_name is None:
            drive_joints = god_map.world.search_for_joint_of_type(self.odom_joints)
            if len(drive_joints) == 0:
                raise GoalInitalizationException('No drive joints in world')
            elif len(drive_joints) == 1:
                self.brumbrum_joint = drive_joints[0]
            else:
                raise GoalInitalizationException('Multiple drive joint found in world, please set \'group_name\'')
        else:
            brumbrum_joint_name = god_map.world.groups[self.group_name].root_link.child_joint_names[0]
            self.brumbrum_joint = god_map.world.joints[brumbrum_joint_name]
            if not isinstance(self.brumbrum_joint, self.odom_joints):
                raise GoalInitalizationException(f'Group {self.group_name} has no odometry joint.')

    def __call__(self):
        base_pose = god_map.world.transform(self.brumbrum_joint.parent_link_name, self.base_pose)
        position = base_pose.to_position().to_np()
        orientation = base_pose.to_rotation().to_quaternion().to_np()
        god_map.world.state[self.brumbrum_joint.x.name].position = position[0]
        god_map.world.state[self.brumbrum_joint.y.name].position = position[1]
        axis, angle = axis_angle_from_quaternion(orientation[0],
                                                 orientation[1],
                                                 orientation[2],
                                                 orientation[3])
        if axis[-1] < 0:
            angle = -angle
        if isinstance(self.brumbrum_joint, OmniDrivePR22):
            god_map.world.state[self.brumbrum_joint.yaw1_vel.name].position = 0
            god_map.world.state[self.brumbrum_joint.yaw.name].position = angle
        else:
            god_map.world.state[self.brumbrum_joint.yaw.name].position = angle
        god_map.world.notify_state_change()
        self.state = ObservationState.true
