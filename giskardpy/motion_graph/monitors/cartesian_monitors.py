from typing import Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_graph.monitors.monitors import ExpressionMonitor


class PoseReached(ExpressionMonitor):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_pose: cas.TransMatrix,
                 position_threshold: float = 0.01,
                 orientation_threshold: float = 0.01,
                 absolute: bool = False,
                 name: Optional[str] = None):
        super().__init__(name=name)
        if absolute or cas.is_true(start_condition):
            root_T_goal = god_map.world.transform(root_link, goal_pose)
        else:
            root_T_x = god_map.world.compose_fk_expression(root_link, goal_pose.reference_frame)
            root_T_goal = root_T_x.dot(goal_pose)
            root_T_goal = god_map.motion_graph_manager.register_expression_updater(root_T_goal, start_condition)

        # %% position error
        r_P_g = root_T_goal.to_position()
        r_P_c = god_map.world.compose_fk_expression(root_link, tip_link).to_position()
        distance_to_goal = cas.euclidean_distance(r_P_g, r_P_c)
        position_reached = cas.less(distance_to_goal, position_threshold)

        # %% orientation error
        r_R_g = root_T_goal.to_rotation()
        r_R_c = god_map.world.compose_fk_expression(root_link, tip_link).to_rotation()
        rotation_error = cas.rotational_error(r_R_c, r_R_g)
        orientation_reached = cas.less(cas.abs(rotation_error), orientation_threshold)

        self.expression = cas.logic_and(position_reached, orientation_reached)


class PositionReached(ExpressionMonitor):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_point: cas.Point3,
                 threshold: float = 0.01,
                 absolute: bool = False,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)
        if absolute or cas.is_true(start_condition):
            root_P_goal = god_map.world.transform(root_link, goal_point)
        else:
            root_P_x = god_map.world.compose_fk_expression(root_link, goal_point.reference_frame)
            root_P_goal = root_P_x.dot(goal_point)
            root_P_goal = god_map.motion_graph_manager.register_expression_updater(root_P_goal, start_condition)

        r_P_c = god_map.world.compose_fk_expression(root_link, tip_link).to_position()
        distance_to_goal = cas.euclidean_distance(root_P_goal, r_P_c)
        self.expression = cas.less(distance_to_goal, threshold)


class OrientationReached(ExpressionMonitor):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_orientation: cas.RotationMatrix,
                 threshold: float = 0.01,
                 absolute: bool = False,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)
        if absolute or cas.is_true(start_condition):
            r_R_g = god_map.world.transform(root_link, goal_orientation)
        else:
            root_T_x = god_map.world.compose_fk_expression(root_link, goal_orientation.reference_frame)
            root_R_goal = root_T_x.dot(goal_orientation)
            r_R_g = god_map.motion_graph_manager.register_expression_updater(root_R_goal, start_condition)

        r_R_c = god_map.world.compose_fk_expression(root_link, tip_link).to_rotation()
        rotation_error = cas.rotational_error(r_R_c, r_R_g)
        self.expression = cas.less(cas.abs(rotation_error), threshold)


class PointingAt(ExpressionMonitor):
    def __init__(self,
                 tip_link: PrefixName,
                 goal_point: cas.Point3,
                 root_link: PrefixName,
                 pointing_axis: cas.Vector3,
                 threshold: float = 0.01,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)
        self.root = root_link
        self.tip = tip_link
        self.root_P_goal_point = god_map.world.transform(self.root, goal_point)

        tip_V_pointing_axis = god_map.world.transform(self.tip, pointing_axis)
        tip_V_pointing_axis.scale(1)
        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_P_tip = root_T_tip.to_position()

        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip
        distance = cas.distance_point_to_line(frame_P_point=self.root_P_goal_point,
                                              frame_P_line_point=root_P_tip,
                                              frame_V_line_direction=root_V_pointing_axis)
        expr = cas.less(cas.abs(distance), threshold)
        self.expression = expr


class VectorsAligned(ExpressionMonitor):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_normal: cas.Vector3,
                 tip_normal: cas.Vector3,
                 threshold: float = 0.01,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)
        self.root = root_link
        self.tip = tip_link

        self.tip_V_tip_normal = god_map.world.transform(self.tip, tip_normal)
        self.tip_V_tip_normal.scale(1)

        self.root_V_root_normal = god_map.world.transform(self.root, goal_normal)
        self.root_V_root_normal.scale(1)

        root_R_tip = god_map.world.compose_fk_expression(self.root, self.tip).to_rotation()
        root_V_tip_normal = root_R_tip.dot(self.tip_V_tip_normal)
        error = cas.angle_between_vector(root_V_tip_normal, self.root_V_root_normal)
        expr = cas.less(error, threshold)
        self.expression = expr


class DistanceToLine(ExpressionMonitor):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 center_point: cas.Point3,
                 line_axis: cas.Vector3,
                 line_length: float,
                 threshold: float = 0.01,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)
        self.root = root_link
        self.tip = tip_link

        root_P_current = god_map.world.compose_fk_expression(self.root, self.tip).to_position()
        root_V_line_axis = god_map.world.transform(self.root, line_axis)
        root_V_line_axis.scale(1)
        root_P_center = god_map.world.transform(self.root, center_point)
        root_P_line_start = root_P_center + root_V_line_axis * (line_length / 2)
        root_P_line_end = root_P_center - root_V_line_axis * (line_length / 2)

        distance, closest_point = cas.distance_point_to_line_segment(frame_P_current=root_P_current,
                                                                     frame_P_line_start=root_P_line_start,
                                                                     frame_P_line_end=root_P_line_end)
        expr = cas.less(distance, threshold)
        self.expression = expr
