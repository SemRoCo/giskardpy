#!/usr/bin/env python
import rospy

from giskardpy.configs.data_types import ControlModes
from giskardpy.configs.default_giskard import Giskard
from giskardpy.my_types import Derivatives

rospy.init_node('giskard')

# Set map as root link, because this is the standard in most setup
giskard = Giskard(root_link_name='map')

# Tell Giskard where to find the robot description
giskard.add_robot_from_parameter_server(parameter_name='robot_description',
                                        joint_state_topics=['/joint_states'])

# Enable stand alone mode for testing
giskard.set_control_mode(ControlModes.stand_alone)

# If you want Giskard to publish tf of it's world
giskard.publish_all_tf()

# These two slow down the planning, but gives a smooth visualization, which is probably preferable without a robot
giskard.configure_VisualizationBehavior(in_planning_loop=True)
giskard.configure_CollisionMarker(in_planning_loop=True)

# Create a tf tree

# Mobile robots have a localization, so we add a fixed joint to simulate that
giskard.add_fixed_joint(parent_link='map', child_link='odom_combined')
# Tell giskard what kind of driver the robot should have. It has to connect to the root link of the robot in it's urdf
giskard.add_omni_drive_joint(parent_link_name='odom_combined',
                             child_link_name='base_footprint',
                             name='brumbrum',
                             translation_limits={
                                 Derivatives.velocity: 0.4,
                                 Derivatives.acceleration: 1,
                                 Derivatives.jerk: 5,
                             },
                             rotation_limits={
                                 Derivatives.velocity: 0.2,
                                 Derivatives.acceleration: 1,
                                 Derivatives.jerk: 5
                             }
                             )
# Tell Giskard which joints in the urdf can be controlled. You must also add the joint name for the drive.
giskard.register_controlled_joints([
    'torso_lift_joint',
    'head_pan_joint',
    'head_tilt_joint',
    'r_shoulder_pan_joint',
    'r_shoulder_lift_joint',
    'r_upper_arm_roll_joint',
    'r_forearm_roll_joint',
    'r_elbow_flex_joint',
    'r_wrist_flex_joint',
    'r_wrist_roll_joint',
    'l_shoulder_pan_joint',
    'l_shoulder_lift_joint',
    'l_upper_arm_roll_joint',
    'l_forearm_roll_joint',
    'l_elbow_flex_joint',
    'l_wrist_flex_joint',
    'l_wrist_roll_joint',
    'brumbrum'
])

# Start Giskard.
giskard.live()
