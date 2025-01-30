from __future__ import division

from typing import Optional

import numpy as np

from giskardpy import casadi_wrapper as cas
from giskardpy.data_types.data_types import Derivatives, ColorRGBA, PrefixName
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.model.joints import DiffDrive
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition, CartesianOrientation, \
    CartesianPositionStraight, CartesianPose
from giskardpy.symbol_manager import symbol_manager
from giskardpy.motion_statechart.tasks.task import WEIGHT_ABOVE_CA, Task


class DiffDriveBaseGoal(Goal):

    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_pose: cas.TransMatrix,
                 max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA,
                 pointing_axis=None,
                 always_forward: bool = False,
                 name: Optional[str] = None):
        """
        Like a CartesianPose, but specifically for differential drives. It will achieve the goal in 3 phases.
        1. orient towards goal.
        2. drive to goal point.
        3. reach goal orientation.
        :param root_link: root link of the kinematic chain. typically map
        :param tip_link: tip link of the kinematic chain. typically base_footprint or similar
        :param goal_pose:
        :param max_linear_velocity:
        :param max_angular_velocity:
        :param weight:
        :param pointing_axis: the forward direction. default is x-axis
        :param always_forward: if false, it will drive backwards, if it requires less rotation.
        """
        self.always_forward = always_forward
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        if pointing_axis is None:
            pointing_axis = cas.Vector3((1, 0, 0))
            pointing_axis.reference_frame = tip_link
        self.weight = weight
        self.map = root_link
        self.base_footprint = tip_link
        super().__init__(name=name)
        self.goal_pose = god_map.world.transform(self.map, goal_pose)
        self.goal_pose.z = 0
        diff_drive_joints = [v for k, v in god_map.world.joints.items() if isinstance(v, DiffDrive)]
        assert len(diff_drive_joints) == 1
        self.joint: DiffDrive = diff_drive_joints[0]
        self.odom = self.joint.parent_link_name

        if pointing_axis is not None:
            self.base_footprint_V_pointing_axis = god_map.world.transform(self.base_footprint, pointing_axis)
            self.base_footprint_V_pointing_axis.scale(1)
        else:
            self.base_footprint_V_pointing_axis = cas.Vector3()
            self.base_footprint_V_pointing_axis.reference_frame = self.base_footprint
            self.base_footprint_V_pointing_axis.z = 1

        map_T_base_current = god_map.world.compute_fk(self.map, self.base_footprint)
        map_T_odom_current = god_map.world.compute_fk(self.map, self.odom)
        _, map_odom_angle = map_T_odom_current.to_rotation().to_axis_angle()
        map_R_base_current = map_T_base_current.to_rotation()
        axis_start, angle_start = map_R_base_current.to_axis_angle()
        angle_start = cas.if_greater_zero(axis_start[2], angle_start, -angle_start)

        map_T_base_footprint = god_map.world.compose_fk_expression(self.map, self.base_footprint)
        map_P_base_footprint = map_T_base_footprint.to_position()
        # map_R_base_footprint = map_T_base_footprint.to_rotation()
        map_T_base_footprint_goal = self.goal_pose
        map_P_base_footprint_goal = map_T_base_footprint_goal.to_position()
        map_R_base_footprint_goal = map_T_base_footprint_goal.to_rotation()

        map_V_goal_x = map_P_base_footprint_goal - map_P_base_footprint
        distance = map_V_goal_x.norm()

        # axis, map_current_angle = map_R_base_footprint.to_axis_angle()
        # map_current_angle = cas.if_greater_zero(axis.z, map_current_angle, -map_current_angle)
        odom_current_angle = self.joint.yaw.get_symbol(Derivatives.position)
        map_current_angle = map_odom_angle + odom_current_angle

        axis2, map_goal_angle2 = map_R_base_footprint_goal.to_axis_angle()
        map_goal_angle2 = cas.if_greater_zero(axis2.z, map_goal_angle2, -map_goal_angle2)
        final_rotation_error = cas.shortest_angular_distance(map_current_angle, map_goal_angle2)

        map_R_goal = cas.RotationMatrix.from_vectors(x=map_V_goal_x, y=None, z=cas.Vector3((0, 0, 1)))

        map_goal_angle_direction_f = map_R_goal.to_angle(lambda axis: axis[2])

        map_goal_angle_direction_b = cas.if_less_eq(map_goal_angle_direction_f, 0,
                                                    if_result=map_goal_angle_direction_f + np.pi,
                                                    else_result=map_goal_angle_direction_f - np.pi)

        middle_angle = cas.normalize_angle(
            map_goal_angle2 + cas.shortest_angular_distance(map_goal_angle2, angle_start) / 2)

        middle_angle = symbol_manager.evaluate_expr(middle_angle)
        a = symbol_manager.evaluate_expr(cas.shortest_angular_distance(map_goal_angle_direction_f, middle_angle))
        b = symbol_manager.evaluate_expr(cas.shortest_angular_distance(map_goal_angle_direction_b, middle_angle))
        eps = 0.01
        if self.always_forward:
            map_goal_angle1 = map_goal_angle_direction_f
        else:
            map_goal_angle1 = cas.if_less_eq(cas.abs(a) - cas.abs(b), 0.03,
                                             if_result=map_goal_angle_direction_f,
                                             else_result=map_goal_angle_direction_b)
        rotate_to_goal_error = cas.shortest_angular_distance(map_current_angle, map_goal_angle1)

        orient_to_goal = Task(name='orient_to_goal')
        self.add_task(orient_to_goal)
        orient_to_goal.add_equality_constraint(reference_velocity=self.max_angular_velocity,
                                               equality_bound=rotate_to_goal_error,
                                               weight=self.weight,
                                               task_expression=map_current_angle)
        orient_to_goal.observation_expression = cas.less_equal(cas.abs(rotate_to_goal_error), eps)

        drive_to_goal = Task(name='drive_to_goal')
        self.add_task(drive_to_goal)
        drive_to_goal.add_point_goal_constraints(frame_P_current=map_P_base_footprint,
                                                 frame_P_goal=map_P_base_footprint_goal,
                                                 reference_velocity=self.max_linear_velocity,
                                                 weight=self.weight)
        drive_to_goal.observation_expression = cas.less_equal(cas.abs(distance), eps * 2)

        final_orientation = Task(name='final_orientation')
        self.add_task(final_orientation)
        final_orientation.add_equality_constraint(reference_velocity=self.max_angular_velocity,
                                                  equality_bound=final_rotation_error,
                                                  weight=self.weight,
                                                  task_expression=map_current_angle)
        final_orientation.observation_expression = cas.less_equal(cas.abs(final_rotation_error), eps)

        god_map.debug_expression_manager.add_debug_expression('distance', distance)
        # god_map.debug_expression_manager.add_debug_expression('final_orientation.observation_expression', final_orientation.observation_expression)

        orient_to_goal.end_condition = orient_to_goal
        drive_to_goal.start_condition = orient_to_goal
        drive_to_goal.end_condition = drive_to_goal
        final_orientation.start_condition = drive_to_goal
        self.observation_expression = cas.logic_and3(drive_to_goal.observation_expression,
                                                     final_orientation.observation_expression)



class CartesianPoseStraight(Goal):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_pose: cas.TransMatrix,
                 reference_linear_velocity: Optional[float] = None,
                 reference_angular_velocity: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 absolute: bool = False,
                 name: Optional[str] = None):
        """
        See CartesianPose. In contrast to it, this goal will try to move tip_link in a straight line.
        """
        self.root_link = root_link
        self.tip_link = tip_link
        super().__init__(name=name)
        self.add_task(CartesianPositionStraight(root_link=root_link,
                                                tip_link=tip_link,
                                                name=name + '/pos',
                                                goal_point=goal_pose.to_position(),
                                                reference_velocity=reference_linear_velocity,
                                                weight=weight,
                                                absolute=absolute))
        self.add_task(CartesianOrientation(root_link=root_link,
                                           tip_link=tip_link,
                                           name=name + '/rot',
                                           goal_orientation=goal_pose.to_rotation(),
                                           reference_velocity=reference_angular_velocity,
                                           absolute=absolute,
                                           weight=weight,
                                           point_of_debug_matrix=goal_pose.to_position()))


class RelativePositionSequence(Goal):
    def __init__(self,
                 goal1: cas.TransMatrix,
                 goal2: cas.TransMatrix,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 name: Optional[str] = None):
        """
        Only meant for testing.
        """
        if name is None:
            name = f'{self.__class__.__name__}/{root_link}/{tip_link}'
        super().__init__(name=name)
        name1 = f'{self.name}/goal1'
        name2 = f'{self.name}/goal2'
        task1 = CartesianPose(root_link=root_link,
                              tip_link=tip_link,
                              goal_pose=goal1,
                              name=name1,
                              absolute=True)
        self.add_task(task1)
        task2 = CartesianPose(root_link=root_link,
                              tip_link=tip_link,
                              goal_pose=goal2,
                              name=name2,
                              absolute=True)
        self.add_task(task2)
        task2.start_condition = task1
        task1.end_condition = task1
        self.observation_expression = task2.observation_expression
