import numpy as np
import rospy

from giskardpy.data_types.data_types import Derivatives, PrefixName
from giskardpy.model.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.model.joints import JustinTorso, RevoluteJoint
from giskardpy.model.world_config import WorldConfig
from giskardpy_ros.configs.robot_interface_config import StandAloneRobotInterfaceConfig


class WorldWithJustinConfig(WorldConfig):
    map_name: PrefixName
    localization_joint_name: PrefixName
    odom_link_name: PrefixName
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom',
                 drive_joint_name: str = 'brumbrum',
                 description_name: str = 'robot_description'):
        super().__init__()
        self.map_name = PrefixName(map_name)
        self.localization_joint_name = PrefixName(localization_joint_name)
        self.odom_link_name = PrefixName(odom_link_name)
        self.drive_joint_name = drive_joint_name
        self.urdf = rospy.get_param('robot_description')

    def setup(self):
        self.set_default_color(1, 1, 1, 1)
        self.set_default_limits({Derivatives.velocity: 1,
                                 Derivatives.acceleration: np.inf,
                                 Derivatives.jerk: None})
        self.add_empty_link(self.map_name)
        self.add_6dof_joint(parent_link=self.map_name, child_link=self.odom_link_name,
                            joint_name=self.localization_joint_name)
        self.add_empty_link(self.odom_link_name)
        self.add_robot_urdf(self.urdf)
        root_link_name = self.get_root_link_of_group(self.robot_group_name)
        self.add_omni_drive_joint(parent_link_name=self.odom_link_name,
                                  child_link_name=root_link_name,
                                  name=self.drive_joint_name,
                                  translation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: np.inf,
                                      Derivatives.jerk: None,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: np.inf,
                                      Derivatives.jerk: None
                                  },
                                  robot_group_name=self.robot_group_name)
        torso2_joint: RevoluteJoint = self.world.joints[self.world.search_for_joint_name('torso2_joint')]
        torso3_joint: RevoluteJoint = self.world.joints[self.world.search_for_joint_name('torso3_joint')]
        torso4_joint: RevoluteJoint = self.world.joints[self.world.search_for_joint_name('torso4_joint')]
        passive_torso_joint = JustinTorso(name=torso4_joint.name,
                                          parent_link_name=torso4_joint.parent_link_name,
                                          child_link_name=torso4_joint.child_link_name,
                                          axis=torso4_joint.axis,
                                          parent_T_child=torso4_joint.original_parent_T_child,
                                          q1=torso2_joint.free_variable,
                                          q2=torso3_joint.free_variable)
        del self.world.joints[torso4_joint.name]
        self.world.add_joint(passive_torso_joint)
        self.set_joint_limits(limit_map={Derivatives.velocity: 0.2}, joint_name='torso1_joint')
        self.set_joint_limits(limit_map={Derivatives.velocity: 0.2}, joint_name='torso2_joint')
        self.set_joint_limits(limit_map={Derivatives.velocity: 0.2}, joint_name='torso3_joint')

class JustinStandaloneInterface(StandAloneRobotInterfaceConfig):

    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__([
            drive_joint_name,
            "torso1_joint",
            "torso2_joint",
            "torso3_joint",
            "head1_joint",
            "head2_joint",
            "left_arm1_joint",
            "left_arm2_joint",
            "left_arm3_joint",
            "left_arm4_joint",
            "left_arm5_joint",
            "left_arm6_joint",
            "left_arm7_joint",
            "left_1thumb1_joint",
            "left_1thumb2_joint",
            "left_1thumb3_joint",
            "left_1thumb4_joint",
            "left_2tip1_joint",
            "left_2tip2_joint",
            "left_2tip3_joint",
            "left_2tip4_joint",
            "left_3middle1_joint",
            "left_3middle2_joint",
            "left_3middle3_joint",
            "left_3middle4_joint",
            "left_4ring1_joint",
            "left_4ring2_joint",
            "left_4ring3_joint",
            "left_4ring4_joint",
            "right_arm1_joint",
            "right_arm2_joint",
            "right_arm3_joint",
            "right_arm4_joint",
            "right_arm5_joint",
            "right_arm6_joint",
            "right_arm7_joint",
            "right_1thumb1_joint",
            "right_1thumb2_joint",
            "right_1thumb3_joint",
            "right_1thumb4_joint",
            "right_3middle1_joint",
            "right_3middle2_joint",
            "right_3middle3_joint",
            "right_3middle4_joint",
            "right_4ring1_joint",
            "right_4ring2_joint",
            "right_4ring3_joint",
            "right_4ring4_joint",
            "right_2tip1_joint",
            "right_2tip2_joint",
            "right_2tip3_joint",
            "right_2tip4_joint"
        ])


class JustinCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.load_self_collision_matrix('package://giskardpy_ros/self_collision_matrices/justin.srdf')
