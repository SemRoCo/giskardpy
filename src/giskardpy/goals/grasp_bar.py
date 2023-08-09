from __future__ import division

from typing import Optional

import numpy as np
from geometry_msgs.msg import Vector3Stamped, PointStamped, Point, PoseStamped

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.cartesian_goals import CartesianPose
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
from giskardpy.model.links import BoxGeometry
from giskardpy.utils.tfwrapper import np_to_pose, vector_to_np


class GraspBar(Goal):
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 tip_grasp_axis: Vector3Stamped,
                 bar_center: PointStamped,
                 bar_axis: Vector3Stamped,
                 bar_length: float,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 reference_linear_velocity: float = 0.1,
                 reference_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA):
        """
        Like a CartesianPose but with more freedom.
        tip_link is allowed to be at any point along bar_axis, that is without bar_center +/- bar_length.
        It will align tip_grasp_axis with bar_axis, but allows rotation around it.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param tip_grasp_axis: axis of tip_link that will be aligned with bar_axis
        :param bar_center: center of the bar to be grasped
        :param bar_axis: alignment of the bar to be grasped
        :param bar_length: length of the bar to be grasped
        :param root_group: if root_link is not unique, search in this group for matches
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: 
        """
        super().__init__()
        self.root = self.world.search_for_link_name(root_link, root_group)
        self.tip = self.world.search_for_link_name(tip_link, tip_group)

        bar_center = self.transform_msg(self.root, bar_center)

        tip_grasp_axis = self.transform_msg(self.tip, tip_grasp_axis)
        tip_grasp_axis.vector = tf.normalize(tip_grasp_axis.vector)

        bar_axis = self.transform_msg(self.root, bar_axis)
        bar_axis.vector = tf.normalize(bar_axis.vector)

        self.bar_axis = bar_axis
        self.tip_grasp_axis = tip_grasp_axis
        self.bar_center = bar_center
        self.bar_length = bar_length
        self.reference_linear_velocity = reference_linear_velocity
        self.reference_angular_velocity = reference_angular_velocity
        self.weight = weight

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root}/{self.tip}'

    def make_constraints(self):
        root_V_bar_axis = w.Vector3(self.bar_axis)
        tip_V_tip_grasp_axis = w.Vector3(self.tip_grasp_axis)
        root_P_bar_center = w.Point3(self.bar_center)

        root_T_tip = self.get_fk(self.root, self.tip)
        root_V_tip_normal = w.dot(root_T_tip, tip_V_tip_grasp_axis)

        self.add_vector_goal_constraints(frame_V_current=root_V_tip_normal,
                                         frame_V_goal=root_V_bar_axis,
                                         reference_velocity=self.reference_angular_velocity,
                                         weight=self.weight)

        root_P_tip = self.get_fk(self.root, self.tip).to_position()

        root_P_line_start = root_P_bar_center + root_V_bar_axis * self.bar_length / 2
        root_P_line_end = root_P_bar_center - root_V_bar_axis * self.bar_length / 2

        dist, nearest = w.distance_point_to_line_segment(root_P_tip, root_P_line_start, root_P_line_end)

        self.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                        frame_P_goal=nearest,
                                        reference_velocity=self.reference_linear_velocity,
                                        weight=self.weight)


def compute_grasp_pose(map_T_cube, map_T_gripper, cube_dimensions) -> np.ndarray:
    """
    :param map_T_cube: 4x4 transformation matrix
    :param map_T_gripper: 4x4 transformation matrix
    :param cube_dimensions: dimensions of cube along x, y and z respectively
    :return: goal pose of gripper relative to cube
    """
    # Extract the rotation matrix part of the transformations
    map_R_cube = map_T_cube[:3, :3]
    cube_R_gripper = map_T_gripper[:3, :3]
    cube_V_gripper = map_T_cube[:3, 3] - map_T_gripper[:3, 3]
    # cube_T_gripper_pos = map_T_gripper[:3, 3]

    # step 1: collect 6 axis corresponding to x, y, z of map_T_cube and their negatives
    axis_pool = [map_R_cube[:, i] for i in range(3)] + [-map_R_cube[:, i] for i in range(3)]

    # step 2: from that pool, find the axis with the highest z value. Choose this as the x axis of cube_T_goal
    x_axis = max(axis_pool, key=lambda v: v[2])

    # step 3: remove this and its opposite axis from the pool.
    axis_pool = [v for v in axis_pool if not np.allclose(v, x_axis) and not np.allclose(v, -x_axis)]

    # step 4: look at the remaining axis and their corresponding cube dimensions.
    # Select the two which correspond to the largest cube dimension
    remaining_axes = sorted([(v, cube_dimensions[i]) for i, v in enumerate(axis_pool[:3])],
                            key=lambda x: x[1], reverse=True)[:2]

    # step 5: out of the remaining 2, select the one that points most away from the gripper, using cube_T_gripper,
    # and choose it as z axis for cube_T_goal
    z_axis = max(remaining_axes, key=lambda x: np.dot(x[0], cube_V_gripper))[0]

    # step 6: compute y axis from x axis of step 2 and z of step 4
    y_axis = np.cross(z_axis, x_axis)

    # step 7: compose a 4x4 transformation matrix, where the position entires are all 0
    cube_T_goal = np.eye(4)
    cube_T_goal[:3, 0] = x_axis
    cube_T_goal[:3, 1] = y_axis
    cube_T_goal[:3, 2] = z_axis

    return cube_T_goal


class GraspBox(Goal):
    def __init__(self,
                 UUID: str,
                 tip_link: str,
                 root_link: str):
        super().__init__()
        self.uuid = UUID
        self.object = self.world.groups[self.uuid]
        self.object_root_link = self.object.root_link_name
        self.tip_link = self.world.search_for_link_name(tip_link)
        self.root_link = self.world.search_for_link_name(root_link)
        self.shape: BoxGeometry = self.object.root_link.collisions[0]
        cube_pose = self.world.compute_fk_np(self.world.root_link_name, self.object_root_link)
        map_R_goal = compute_grasp_pose(cube_pose,
                                        self.world.compute_fk_np(self.world.root_link_name, self.tip_link),
                                        [self.shape.x_size, self.shape.y_size, self.shape.z_size])
        grasp_goal = PoseStamped()
        grasp_goal.header.frame_id = self.world.root_link_name
        grasp_goal.pose = np_to_pose(map_R_goal)
        grasp_goal.pose.position = self.world.compute_fk_pose(self.world.root_link_name,
                                                              self.object_root_link).pose.position
        self.grasp_goal = grasp_goal
        self.weight = WEIGHT_ABOVE_CA
        self.trans_vel = 0.2
        self.rot_vel = 0.5

    def clean_up(self):
        pass

    def make_constraints(self):
        map_T_grasp_goal = w.TransMatrix(self.grasp_goal)
        r_P_g = map_T_grasp_goal.to_position()
        r_R_g = map_T_grasp_goal.to_rotation()
        r_P_c = self.get_fk(self.root_link, self.tip_link).to_position()
        r_R_c = self.get_fk(self.root_link, self.tip_link).to_rotation()
        c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link).to_rotation()

        map_V_pre_grasp_direction = r_R_g[:, 2]
        map_P_pre_grasp_goal = r_P_g - map_V_pre_grasp_direction * 0.2
        distance_error = w.norm(map_P_pre_grasp_goal - r_P_c)
        # self.add_debug_expr('distance_error', distance_error)
        distance_to_line, root_P_on_line = w.distance_point_to_line_segment(r_P_c,
                                                                            r_P_g,
                                                                            map_P_pre_grasp_goal)
        weight_pregrasp = w.if_less(distance_to_line, 0.01, 0, self.weight)
        weight_grasp = w.if_eq(weight_pregrasp, 0, self.weight, 0)
        self.add_point_goal_constraints(frame_P_current=r_P_c,
                                        frame_P_goal=root_P_on_line,
                                        reference_velocity=0.1,
                                        weight=weight_pregrasp,
                                        name='pregrasp')

        self.add_point_goal_constraints(frame_P_goal=r_P_g,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.trans_vel,
                                        weight=weight_grasp,
                                        name='grasp position')

        # self.add_debug_expr('trans', w.norm(r_P_c))
        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=r_R_g,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=self.rot_vel,
                                           weight=self.weight,
                                           name='grasp orientation')

    def __str__(self) -> str:
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'
