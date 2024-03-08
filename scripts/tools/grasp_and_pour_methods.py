#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped, Quaternion, Point
from tf.transformations import quaternion_about_axis, quaternion_from_matrix

from giskardpy.python_interface.python_interface import GiskardWrapper


def openGripper(giskard: GiskardWrapper):
    giskard.motion_goals.add_motion_goal(motion_goal_class='CloseGripper',
                                         name='openGripper',
                                         as_open=True,
                                         velocity_threshold=100,
                                         effort_threshold=1,
                                         effort=100)
    giskard.motion_goals.allow_all_collisions()
    giskard.add_default_end_motion_conditions()
    giskard.execute()


def closeGripper(giskard: GiskardWrapper):
    giskard.motion_goals.add_motion_goal(motion_goal_class='CloseGripper',
                                         name='closeGripper')
    giskard.motion_goals.allow_all_collisions()
    giskard.add_default_end_motion_conditions()
    giskard.execute()


def align_to(giskard: GiskardWrapper, side: str, axis_align_to_z: Vector3Stamped, object_frame: str, control_frame: str,
             axis_align_to_x: Vector3Stamped = None, distance=0.3, height_offset=0.0, second_distance=0.0):
    """
    side: [front, left, right] relative to the object frame from the pov of a robot standing at the origin of the world frame
    axis_align_to_z: axis of the control_frame that will be aligned to the z-axis of the object frame
    object_frame: name of the tf frame to align to
    control_frame: name of a tf frame attached to the robot that should be moved around
    axis_align_to_x: axis of the control_frame that will be aligned to the x-axis of the object frame
    distance: distance between the object and the control frame along the axis resulting from align to [side]
    height_offset: offset between the two frames on the z-axis
    second_distance: offset on the last free axis. When side == [left, right] this is the x axis of the object frame, otherwise it is the y axis.
    """
    goal_normal = Vector3Stamped()
    goal_normal.header.frame_id = object_frame
    goal_normal.vector.z = 1
    giskard.motion_goals.add_align_planes(goal_normal, control_frame, axis_align_to_z, 'map', name='align_upright')
    if axis_align_to_x:
        second_goal_normal = Vector3Stamped()
        second_goal_normal.header.frame_id = object_frame
        second_goal_normal.vector.x = 1
        giskard.motion_goals.add_align_planes(second_goal_normal, control_frame, axis_align_to_x, 'map',
                                              name='align_second')

    goal_position = PointStamped()
    goal_position.header.frame_id = object_frame
    if side == 'front':
        goal_position.point.x = -distance
        goal_position.point.y = second_distance
        goal_position.point.z = height_offset
    elif side == 'left':
        goal_position.point.x = second_distance
        goal_position.point.y = distance
        goal_position.point.z = height_offset
    elif side == 'right':
        goal_position.point.x = second_distance
        goal_position.point.y = -distance
        goal_position.point.z = height_offset
    giskard.motion_goals.add_cartesian_position(goal_position, control_frame, 'map')
    giskard.add_default_end_motion_conditions()
    giskard.execute()


def tilt(giskard: GiskardWrapper, angle: float, velocity: float, rotation_axis: Vector3Stamped, controlled_frame: str):
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = controlled_frame
    goal_pose.pose.orientation = Quaternion(
        *quaternion_about_axis(angle, [rotation_axis.vector.x, rotation_axis.vector.y, rotation_axis.vector.z]))
    giskard.motion_goals.add_cartesian_pose(goal_pose, controlled_frame, 'map')
    giskard.motion_goals.add_limit_cartesian_velocity(tip_link=controlled_frame, root_link='map',
                                                      max_angular_velocity=velocity)
    giskard.add_default_end_motion_conditions()
    giskard.execute()


if __name__ == '__main__':
    # Before running this script make sure to start a giskard instance using 'roslaunch giskardpy giskardpy_hsr_mujoco.launch'
    # And before that the mujoco simulation has to be running
    rospy.init_node('graspAndPour')
    gis = GiskardWrapper()

    # Define some parameters used in the movement function
    # The endeffector link of the robot
    robot_eeff = 'hand_palm_link'

    # Axis of the eeff that should be upright
    upright_axis = Vector3Stamped()
    upright_axis.header.frame_id = robot_eeff
    upright_axis.vector.x = 1

    # A second axis of the eeff. Can be aligned to the x-axis of goal objects
    second_axis = Vector3Stamped()
    second_axis.header.frame_id = robot_eeff
    second_axis.vector.z = 1

    # Here starts the control
    # Open the gripper. Needs the giskard interface as input, as all the other methods
    openGripper(gis)

    # This aligns the control frame to the front of the object frame in a distance of 0.04m.
    align_to(gis, 'front', axis_align_to_z=upright_axis, object_frame='free_cup', control_frame=robot_eeff,
             axis_align_to_x=second_axis, distance=0.04)

    # Close the gripper
    closeGripper(gis)

    # Here the grasped cup is added to the kinematic model of the robot.
    # It is done to use the frame of the cup as a controlled frame.
    # This can be skipped if the 'hand_palm_link' should be used as a controlled frame after grasping.
    # First define the current pose of the grasped cup
    cup_pose = PoseStamped()
    cup_pose.header.frame_id = 'free_cup'
    cup_pose.pose.position = Point(0, 0, 0)
    cup_pose.pose.orientation.w = 1
    # Add the cup to the robot model name 'grasped_cup' and the known dimensions
    gis.world.add_box('grasped_cup', (0.07, 0.07, 0.18), pose=cup_pose, parent_link=robot_eeff)
    # Now update the robot_eeff reference to use the grasped cup
    robot_eeff = 'grasped_cup'

    # This aligns the control frame to the left of the object frame in a distance of 0.13m.
    # Additionally, the control frame is 0.2m higher than the object frame. The second_distance paramter can be used to
    # to set an offset in the remaining dimension, here the x-axis of the object frame
    align_to(gis, 'left', axis_align_to_z=upright_axis, object_frame='free_cup2', control_frame=robot_eeff,
             axis_align_to_x=second_axis, distance=0.13, height_offset=0.2, second_distance=0.0)

    # Prepare tilting be defining a tilt axis
    rotation_axis = Vector3Stamped()
    rotation_axis.header.frame_id = robot_eeff
    rotation_axis.vector.z = 1

    # Tilt the controlled_frame by angle around the rotation axis with a maximum velocity of velocity.
    tilt(gis, angle=1.7, velocity=1.0, rotation_axis=rotation_axis, controlled_frame='hand_palm_link')
