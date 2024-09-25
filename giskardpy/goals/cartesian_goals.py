from __future__ import division

from typing import Optional

import numpy as np

from giskardpy import casadi_wrapper as cas
from giskardpy.data_types.data_types import Derivatives, ColorRGBA, PrefixName
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.model.joints import DiffDrive
from giskardpy.motion_graph.monitors.monitors import ExpressionMonitor
from giskardpy.motion_graph.tasks.cartesian_tasks import CartesianPosition, CartesianOrientation
from giskardpy.symbol_manager import symbol_manager
from giskardpy.motion_graph.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA


class CartesianPositionStraight(Goal):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_point: cas.Point3,
                 reference_velocity: Optional[float] = None,
                 name: Optional[str] = None,
                 absolute: bool = False,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
        Same as CartesianPosition, but tries to move the tip_link in a straight line to the goal_point.
        """
        if reference_velocity is None:
            reference_velocity = 0.2
        self.reference_velocity = reference_velocity
        self.weight = weight
        self.root_link = root_link
        self.tip_link = tip_link
        if absolute or cas.is_true(start_condition):
            root_P_goal = god_map.world.transform(self.root_link, goal_point)
        else:
            root_T_x = god_map.world.compose_fk_expression(self.root_link, goal_point.reference_frame)
            root_P_goal = root_T_x.dot(goal_point)
            root_P_goal = god_map.motion_graph_manager.register_expression_updater(root_P_goal, start_condition)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)

        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        t_T_r = god_map.world.compose_fk_expression(self.tip_link, self.root_link)
        tip_P_goal = t_T_r.dot(root_P_goal)

        # Create rotation matrix, which rotates the tip link frame
        # such that its x-axis shows towards the goal position.
        # The goal frame is called 'a'.
        # Thus, the rotation matrix is called t_R_a.
        tip_V_error = cas.Vector3(tip_P_goal)
        trans_error = tip_V_error.norm()
        # x-axis
        tip_V_intermediate_error = cas.save_division(tip_V_error, trans_error)
        # y- and z-axis
        tip_V_intermediate_y = cas.Vector3(np.random.random((3,)))
        tip_V_intermediate_y.scale(1)
        y = tip_V_intermediate_error.cross(tip_V_intermediate_y)
        z = tip_V_intermediate_error.cross(y)
        t_R_a = cas.RotationMatrix.from_vectors(x=tip_V_intermediate_error, y=-z, z=y)

        # Apply rotation matrix on the fk of the tip link
        a_T_t = t_R_a.inverse().dot(
            god_map.world.compose_fk_evaluated_expression(self.tip_link, self.root_link)).dot(
            god_map.world.compose_fk_expression(self.root_link, self.tip_link))
        expr_p = a_T_t.to_position()
        dist = cas.norm(root_P_goal - root_P_tip)

        task = self.create_and_add_task('position straight')
        task.add_equality_constraint_vector(reference_velocities=[self.reference_velocity] * 3,
                                            equality_bounds=[dist, 0, 0],
                                            weights=[WEIGHT_ABOVE_CA, WEIGHT_ABOVE_CA * 2, WEIGHT_ABOVE_CA * 2],
                                            task_expression=expr_p[:3],
                                            names=['line/x',
                                                   'line/y',
                                                   'line/z'])
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current_point', root_P_tip,
                                                              color=ColorRGBA(r=1, g=0, b=0, a=1))
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/goal_point', root_P_goal,
                                                              color=ColorRGBA(r=0, g=0, b=1, a=1))
        self.connect_monitors_to_all_tasks(start_condition, pause_condition, end_condition)


class CartesianPose(Goal):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_pose: cas.TransMatrix,
                 reference_linear_velocity: Optional[float] = None,
                 reference_angular_velocity: Optional[float] = None,
                 name: Optional[str] = None,
                 absolute: bool = False,
                 weight=WEIGHT_ABOVE_CA):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param absolute: if False, the goal is updated when start_condition turns True.
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.root_link = root_link
        self.tip_link = tip_link
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name=name)
        position_name = f'{self.name}/position'
        orientation_name = f'{self.name}/orientation'
        if reference_linear_velocity is None:
            reference_linear_velocity = CartesianPosition.default_reference_velocity
        if reference_angular_velocity is None:
            reference_angular_velocity = CartesianOrientation.default_reference_velocity
        self.weight = weight
        position_task = CartesianPosition(root_link=root_link,
                                          tip_link=tip_link,
                                          goal_point=goal_pose.to_position(),
                                          reference_velocity=reference_linear_velocity,
                                          weight=self.weight,
                                          name=position_name,
                                          absolute=absolute)
        self.add_task(position_task)

        orientation_task = CartesianOrientation(root_link=root_link,
                                                tip_link=tip_link,
                                                goal_orientation=goal_pose.to_rotation(),
                                                reference_velocity=reference_angular_velocity,
                                                weight=self.weight,
                                                name=orientation_name,
                                                absolute=absolute,
                                                point_of_debug_matrix=goal_pose.to_position())
        self.add_task(orientation_task)
        self.expression = cas.logic_and(position_task.expression, orientation_task.expression)


class DiffDriveBaseGoal(Goal):

    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 goal_pose: cas.TransMatrix,
                 max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA,
                 pointing_axis=None,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 always_forward: bool = False,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
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
        # TODO make pretty
        self.always_forward = always_forward
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        if pointing_axis is None:
            pointing_axis = cas.Vector3((1, 0, 0))
            pointing_axis.reference_frame = tip_link
        self.weight = weight
        self.map = god_map.world.search_for_link_name(root_link, root_group)
        self.base_footprint = god_map.world.search_for_link_name(tip_link, tip_group)
        if name is None:
            name = f'{self.__class__.__name__}/{self.map}/{self.base_footprint}'
        super().__init__(name)
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

        weight_final_rotation = cas.if_else(cas.logic_and(cas.less_equal(cas.abs(distance), eps * 2),
                                                          cas.greater_equal(cas.abs(final_rotation_error), 0)),
                                            self.weight,
                                            0)
        weight_rotate_to_goal = cas.if_else(cas.logic_and(cas.greater_equal(cas.abs(rotate_to_goal_error), eps),
                                                          cas.greater_equal(cas.abs(distance), eps),
                                                          cas.less_equal(weight_final_rotation, eps)),
                                            self.weight,
                                            0)
        weight_translation = cas.if_else(cas.logic_and(cas.less_equal(cas.abs(rotate_to_goal_error), eps * 2),
                                                       cas.greater_equal(cas.abs(distance), eps)),
                                         self.weight,
                                         0)
        task = self.create_and_add_task()
        task.add_equality_constraint(reference_velocity=self.max_angular_velocity,
                                     equality_bound=rotate_to_goal_error,
                                     weight=weight_rotate_to_goal,
                                     task_expression=map_current_angle,
                                     name='/rot1')
        task.add_point_goal_constraints(frame_P_current=map_P_base_footprint,
                                        frame_P_goal=map_P_base_footprint_goal,
                                        reference_velocity=self.max_linear_velocity,
                                        weight=weight_translation)
        task.add_equality_constraint(reference_velocity=self.max_angular_velocity,
                                     equality_bound=final_rotation_error,
                                     weight=weight_final_rotation,
                                     task_expression=map_current_angle,
                                     name='/rot2')
        self.connect_monitors_to_all_tasks(start_condition, pause_condition, end_condition)


class CartesianPoseStraight(Goal):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_pose: cas.TransMatrix,
                 reference_linear_velocity: Optional[float] = None,
                 reference_angular_velocity: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 absolute: bool = False,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
        See CartesianPose. In contrast to it, this goal will try to move tip_link in a straight line.
        """
        self.root_link = root_link
        self.tip_link = tip_link
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        self.add_constraints_of_goal(CartesianPositionStraight(root_link=root_link,
                                                               tip_link=tip_link,
                                                               goal_point=goal_pose.to_position(),
                                                               reference_velocity=reference_linear_velocity,
                                                               weight=weight,
                                                               absolute=absolute,
                                                               start_condition=start_condition,
                                                               pause_condition=pause_condition,
                                                               end_condition=end_condition))
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal_orientation=goal_pose.to_rotation(),
                                                          reference_velocity=reference_angular_velocity,
                                                          absolute=absolute,
                                                          weight=weight,
                                                          start_condition=start_condition,
                                                          pause_condition=pause_condition,
                                                          end_condition=end_condition,
                                                          point_of_debug_matrix=goal_pose.to_position()))


class TranslationVelocityLimit(Goal):
    def __init__(self, root_link: str, tip_link: str, root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 weight=WEIGHT_ABOVE_CA, max_velocity=0.1, hard=True, name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
        See CartesianVelocityLimit
        """
        self.root_link = god_map.world.search_for_link_name(root_link, root_group)
        self.tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        self.hard = hard
        self.weight = weight
        self.max_velocity = max_velocity

        r_P_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        task = self.create_and_add_task('limit translation vel')
        if not self.hard:
            task.add_translational_velocity_limit(frame_P_current=r_P_c,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight)
        else:
            task.add_translational_velocity_limit(frame_P_current=r_P_c,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight,
                                                  max_violation=0)
        self.connect_monitors_to_all_tasks(start_condition, pause_condition, end_condition)


class RotationVelocityLimit(Goal):
    def __init__(self, root_link: str, tip_link: str, root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 weight=WEIGHT_ABOVE_CA, max_velocity=0.5, hard=True, name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
        See CartesianVelocityLimit
        """
        self.root_link = god_map.world.search_for_link_name(root_link, root_group)
        self.tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        self.hard = hard

        self.weight = weight
        self.max_velocity = max_velocity

        r_R_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_rotation()

        task = self.create_and_add_task('rot vel limit')
        if self.hard:
            task.add_rotational_velocity_limit(frame_R_current=r_R_c,
                                               max_velocity=self.max_velocity,
                                               weight=self.weight)
        else:
            task.add_rotational_velocity_limit(frame_R_current=r_R_c,
                                               max_velocity=self.max_velocity,
                                               weight=self.weight,
                                               max_violation=0)
        self.connect_monitors_to_all_tasks(start_condition, pause_condition, end_condition)


class CartesianVelocityLimit(Goal):
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA,
                 hard: bool = False,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param root_group: if the root_link is not unique, use this to say to which group the link belongs
        :param tip_group: if the tip_link is not unique, use this to say to which group the link belongs
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        self.root_link = root_link
        self.tip_link = tip_link
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        self.add_constraints_of_goal(TranslationVelocityLimit(root_link=root_link,
                                                              root_group=root_group,
                                                              tip_link=tip_link,
                                                              tip_group=tip_group,
                                                              max_velocity=max_linear_velocity,
                                                              weight=weight,
                                                              hard=hard,
                                                              start_condition=start_condition,
                                                              pause_condition=pause_condition,
                                                              end_condition=end_condition))
        self.add_constraints_of_goal(RotationVelocityLimit(root_link=root_link,
                                                           root_group=root_group,
                                                           tip_link=tip_link,
                                                           tip_group=tip_group,
                                                           max_velocity=max_angular_velocity,
                                                           weight=weight,
                                                           hard=hard,
                                                           start_condition=start_condition,
                                                           pause_condition=pause_condition,
                                                           end_condition=end_condition))


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
        goal1 = CartesianPose(root_link=root_link,
                              tip_link=tip_link,
                              goal_pose=goal1,
                              name=name1,
                              absolute=True)
        self.add_goal(goal1)
        goal2 = CartesianPose(root_link=root_link,
                              tip_link=tip_link,
                              goal_pose=goal2,
                              name=name2,
                              absolute=True)
        self.add_goal(goal2)
        goal2.start_condition = goal1.get_observation_state_expression()
        goal1.end_condition = goal1.get_observation_state_expression()
        self.expression = goal2.expression
