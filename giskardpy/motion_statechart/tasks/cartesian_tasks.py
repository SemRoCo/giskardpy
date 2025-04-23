from typing import Optional

from giskardpy.data_types.data_types import Derivatives
from giskardpy import casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName, ColorRGBA
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.monitors.cartesian_monitors import PositionReached, OrientationReached
from giskardpy.motion_statechart.tasks.task import Task, WEIGHT_ABOVE_CA
from giskardpy.symbol_manager import symbol_manager
import numpy as np


class CartesianPosition(Task):
    default_reference_velocity = 0.2

    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_point: cas.Point3,
                 threshold: float = 0.01,
                 reference_velocity: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 absolute: bool = False,
                 name: Optional[str] = None,
                 plot: bool = True):
        """
        See CartesianPose.
        """
        self.root_link = root_link
        self.tip_link = tip_link
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name=name, plot=plot)
        if reference_velocity is None:
            reference_velocity = self.default_reference_velocity
        self.reference_velocity = reference_velocity
        self.weight = weight
        if absolute:
            root_P_goal = god_map.world.transform(self.root_link, goal_point)
        else:
            root_T_x = god_map.world.compose_fk_expression(self.root_link, goal_point.reference_frame)
            root_P_goal = root_T_x.dot(goal_point)
            root_P_goal = self.update_expression_on_starting(root_P_goal)

        r_P_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        self.add_point_goal_constraints(frame_P_goal=root_P_goal,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.reference_velocity,
                                        weight=self.weight)
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/target', root_P_goal.y,
                                                              color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
                                                              derivative=Derivatives.position,
                                                              derivatives_to_plot=[Derivatives.position])

        cap = self.reference_velocity * god_map.qp_controller.mpc_dt * (
                god_map.qp_controller.prediction_horizon - 2)
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/upper_cap', root_P_goal.y + cap,
                                                              derivatives_to_plot=[
                                                                  Derivatives.position,
                                                              ])
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/lower_cap', root_P_goal.y - cap,
                                                              derivatives_to_plot=[
                                                                  Derivatives.position,
                                                              ])
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current', r_P_c.y,
                                                              color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                                                              derivative=Derivatives.position,
                                                              derivatives_to_plot=Derivatives.range(
                                                                  Derivatives.position,
                                                                  Derivatives.jerk)
                                                              )

        distance_to_goal = cas.euclidean_distance(root_P_goal, r_P_c)
        self.observation_expression = cas.less(distance_to_goal, threshold)


class CartesianPositionStraight(Task):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_point: cas.Point3,
                 threshold: float = 0.01,
                 reference_velocity: Optional[float] = None,
                 name: Optional[str] = None,
                 absolute: bool = False,
                 weight: float = WEIGHT_ABOVE_CA):
        """
        Same as CartesianPosition, but tries to move the tip_link in a straight line to the goal_point.
        """
        super().__init__(name=name)
        if reference_velocity is None:
            reference_velocity = CartesianPosition.default_reference_velocity
        self.reference_velocity = reference_velocity
        self.weight = weight
        self.root_link = root_link
        self.tip_link = tip_link
        self.threshold = threshold

        if absolute:
            root_P_goal = god_map.world.transform(self.root_link, goal_point)
        else:
            root_T_x = god_map.world.compose_fk_expression(self.root_link, goal_point.reference_frame)
            root_P_goal = root_T_x.dot(goal_point)
            root_P_goal = self.update_expression_on_starting(root_P_goal)

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

        self.add_equality_constraint_vector(reference_velocities=[self.reference_velocity] * 3,
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
        self.observation_expression = cas.less(dist, self.threshold)


class CartesianOrientation(Task):
    default_reference_velocity = 0.2

    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_orientation: cas.RotationMatrix,
                 threshold: float = 0.01,
                 reference_velocity: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 absolute: bool = False,
                 point_of_debug_matrix: Optional[cas.Point3] = None):
        """
        See CartesianPose.
        """
        self.root_link = root_link
        self.tip_link = tip_link
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name=name)
        if reference_velocity is None:
            reference_velocity = self.default_reference_velocity
        self.reference_velocity = reference_velocity
        self.weight = weight

        if absolute:
            root_R_goal = god_map.world.transform(self.root_link, goal_orientation)
        else:
            root_T_x = god_map.world.compose_fk_expression(self.root_link, goal_orientation.reference_frame)
            root_R_goal = root_T_x.dot(goal_orientation)
            root_R_goal = self.update_expression_on_starting(root_R_goal)

        r_T_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        r_R_c = r_T_c.to_rotation()
        c_R_r_eval = god_map.world.compose_fk_evaluated_expression(self.tip_link, self.root_link).to_rotation()

        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=root_R_goal,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=self.reference_velocity,
                                           weight=self.weight)
        if point_of_debug_matrix is None:
            point = r_T_c.to_position()
        else:
            if absolute:
                point = point_of_debug_matrix
            else:
                root_T_x = god_map.world.compose_fk_expression(self.root_link, point_of_debug_matrix.reference_frame)
                point = root_T_x.dot(point_of_debug_matrix)
                point = self.update_expression_on_starting(point)
        debug_trans_matrix = cas.TransMatrix.from_point_rotation_matrix(point=point,
                                                                        rotation_matrix=root_R_goal)
        debug_current_trans_matrix = cas.TransMatrix.from_point_rotation_matrix(point=r_T_c.to_position(),
                                                                                rotation_matrix=r_R_c)
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/goal_orientation', debug_trans_matrix)
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current_orientation',
        #                                                       debug_current_trans_matrix)

        rotation_error = cas.rotational_error(r_R_c, root_R_goal)
        self.observation_expression = cas.less(cas.abs(rotation_error), threshold)


class CartesianPose(Task):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_pose: cas.TransMatrix,
                 reference_linear_velocity: Optional[float] = None,
                 reference_angular_velocity: Optional[float] = None,
                 threshold: float = 0.01,
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
        if weight is None:
            weight = WEIGHT_ABOVE_CA
        self.root_link = root_link
        self.tip_link = tip_link
        super().__init__(name=name)
        if reference_linear_velocity is None:
            reference_linear_velocity = CartesianOrientation.default_reference_velocity
        self.reference_linear_velocity = reference_linear_velocity
        if reference_angular_velocity is None:
            reference_angular_velocity = CartesianOrientation.default_reference_velocity
        self.reference_angular_velocity = reference_angular_velocity

        self.weight = weight
        goal_orientation = goal_pose.to_rotation()
        goal_point = goal_pose.to_position()

        if absolute:
            root_P_goal = god_map.world.transform(self.root_link, goal_point)
            root_R_goal = god_map.world.transform(self.root_link, goal_orientation)
        else:
            root_T_x = god_map.world.compose_fk_expression(self.root_link, goal_point.reference_frame)
            root_P_goal = root_T_x.dot(goal_point)
            root_P_goal = self.update_expression_on_starting(root_P_goal)
            root_R_goal = root_T_x.dot(goal_orientation)
            root_R_goal = self.update_expression_on_starting(root_R_goal)

        r_P_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        self.add_point_goal_constraints(frame_P_goal=root_P_goal,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.reference_linear_velocity,
                                        weight=self.weight)
        if tip_link in ['rollin_justin/l_gripper_tool_frame', 'box/box']:
            god_map.debug_expression_manager.add_debug_expression(f'{self.name}/l/current_point', r_P_c,
                                                                  color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0))
        if tip_link in ['rollin_justin/r_gripper_tool_frame']:
            god_map.debug_expression_manager.add_debug_expression(f'{self.name}/r/current_point', r_P_c,
                                                                  color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
        if tip_link in ['Schnibbler/Schnibbler']:
            god_map.debug_expression_manager.add_debug_expression(f'{self.name}/r_P_c', r_P_c,
                                                                  color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0))

        distance_to_goal = cas.euclidean_distance(root_P_goal, r_P_c)

        r_T_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        r_R_c = r_T_c.to_rotation()
        c_R_r_eval = god_map.world.compose_fk_evaluated_expression(self.tip_link, self.root_link).to_rotation()

        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=root_R_goal,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=self.reference_angular_velocity,
                                           weight=self.weight)
        debug_trans_matrix = cas.TransMatrix.from_point_rotation_matrix(point=goal_point,
                                                                        rotation_matrix=root_R_goal)
        debug_current_trans_matrix = cas.TransMatrix.from_point_rotation_matrix(point=r_T_c.to_position(),
                                                                                rotation_matrix=r_R_c)
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/goal_orientation', debug_trans_matrix)
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current_orientation',
        #                                                       debug_current_trans_matrix)

        rotation_error = cas.rotational_error(r_R_c, root_R_goal)
        self.observation_expression = cas.logic_and(cas.less(cas.abs(rotation_error), threshold),
                                                    cas.less(distance_to_goal, threshold))


class CartesianPositionVelocityLimit(Task):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 name: str,
                 max_linear_velocity: float = 0.2,
                 weight: float = WEIGHT_ABOVE_CA):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        self.root_link = root_link
        self.tip_link = tip_link
        super().__init__(name=name)
        r_P_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        self.add_translational_velocity_limit(frame_P_current=r_P_c,
                                              max_velocity=max_linear_velocity,
                                              weight=weight)


class CartesianRotationVelocityLimit(Task):
    def __init__(self,
                 name: str,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 weight=WEIGHT_ABOVE_CA,
                 max_velocity: Optional[float] = None):
        """
        See CartesianVelocityLimit
        """
        self.root_link = root_link
        self.tip_link = tip_link
        super().__init__(name=name)
        self.weight = weight
        self.max_velocity = max_velocity

        r_R_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_rotation()

        self.add_rotational_velocity_limit(frame_R_current=r_R_c,
                                           max_velocity=self.max_velocity,
                                           weight=self.weight)


class CartesianVelocityLimit(Task):
    def __init__(self,
                 name: str,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.root_link = root_link
        self.tip_link = tip_link
        super().__init__(name=name)
        r_T_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        r_P_c = r_T_c.to_position()
        r_R_c = r_T_c.to_rotation()
        self.add_translational_velocity_limit(frame_P_current=r_P_c,
                                              max_velocity=max_linear_velocity,
                                              weight=weight)
        self.add_rotational_velocity_limit(frame_R_current=r_R_c,
                                           max_velocity=max_angular_velocity,
                                           weight=weight)


class CartesianPositionVelocityTarget(Task):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 name: str,
                 x_vel: float,
                 y_vel: float,
                 z_vel: float,
                 weight: float = WEIGHT_ABOVE_CA):
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
        super().__init__(name=name)
        r_P_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/target',
                                                              cas.Expression(y_vel),
                                                              derivative=Derivatives.velocity,
                                                              derivatives_to_plot=[
                                                                  # Derivatives.position,
                                                                  Derivatives.velocity
                                                              ])
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current', r_P_c.y,
                                                              derivative=Derivatives.position,
                                                              derivatives_to_plot=Derivatives.range(
                                                                  Derivatives.position,
                                                                  Derivatives.jerk)
                                                              )
        self.add_velocity_eq_constraint_vector(velocity_goals=cas.Expression([x_vel, y_vel, z_vel]),
                                               task_expression=r_P_c,
                                               reference_velocities=[
                                                   CartesianPosition.default_reference_velocity,
                                                   CartesianPosition.default_reference_velocity,
                                                   CartesianPosition.default_reference_velocity,
                                               ],
                                               names=[
                                                   f'{name}/x',
                                                   f'{name}/y',
                                                   f'{name}/z',
                                               ],
                                               weights=[weight] * 3)


class JustinTorsoLimitCart(Task):
    def __init__(self, *, root_link: PrefixName, tip_link: PrefixName,
                 forward_distance: float, backward_distance: float, weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None, plot: bool = True):
        super().__init__(name=name, plot=plot)
        torso_root_T_torso_tip = god_map.world.compose_fk_expression(root_link, tip_link)
        torso_root_V_up = cas.Vector3((0, 0, 1))
        torso_root_V_up.reference_frame = root_link
        torso_root_V_up.vis_frame = root_link

        torso_root_V_left = cas.Vector3((0, 1, 0))
        torso_root_V_left.reference_frame = root_link
        torso_root_V_left.vis_frame = root_link

        torso_root_P_torso_tip = torso_root_T_torso_tip.to_position()

        distance, nearest = cas.distance_point_to_plane_signed(frame_P_current=torso_root_P_torso_tip,
                                                               frame_V_v1=torso_root_V_left,
                                                               frame_V_v2=torso_root_V_up)
        # distance = cas.distance_point_to_line(torso_root_P_torso_tip, cas.Point3((0, 0, 0)), torso_root_V_up)

        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/torso_root_V_up',
        #                                                       expression=torso_root_V_up)
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/torso_root_P_torso_tip',
        #                                                       expression=torso_root_P_torso_tip)
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/distance',
                                                              expression=distance)

        self.add_inequality_constraint(reference_velocity=CartesianPosition.default_reference_velocity,
                                       lower_error=-backward_distance - distance,
                                       upper_error=forward_distance - distance,
                                       weight=weight,
                                       task_expression=distance,
                                       name=f'{name}/distance')
        self.observation_expression = cas.less_equal(distance, forward_distance)
