from typing import Optional

from giskardpy import casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName, ColorRGBA
from giskardpy.god_map import god_map
from giskardpy.motion_graph.monitors.cartesian_monitors import PositionReached, OrientationReached
from giskardpy.motion_graph.tasks.task import Task, WEIGHT_ABOVE_CA


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
            root_P_goal = self.update_expression_on_enter_running(root_P_goal)

        r_P_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        self.add_point_goal_constraints(frame_P_goal=root_P_goal,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.reference_velocity,
                                        weight=self.weight)
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current_point', r_P_c,
                                                              color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/goal_point', root_P_goal,
                                                              color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0))

        distance_to_goal = cas.euclidean_distance(root_P_goal, r_P_c)
        self.expression = cas.less(distance_to_goal, threshold)


class CartesianOrientation(Task):
    default_reference_velocity = 0.5

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
            root_R_goal = self.update_expression_on_enter_running(root_R_goal)

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
                point = self.update_expression_on_enter_running(point)
        debug_trans_matrix = cas.TransMatrix.from_point_rotation_matrix(point=point,
                                                                        rotation_matrix=root_R_goal)
        debug_current_trans_matrix = cas.TransMatrix.from_point_rotation_matrix(point=r_T_c.to_position(),
                                                                                rotation_matrix=r_R_c)
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/goal_orientation', debug_trans_matrix)
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current_orientation',
                                                              debug_current_trans_matrix)

        rotation_error = cas.rotational_error(r_R_c, root_R_goal)
        self.expression = cas.less(cas.abs(rotation_error), threshold)
