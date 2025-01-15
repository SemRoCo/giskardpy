from __future__ import division

from typing import Optional

import numpy as np

from giskardpy import casadi_wrapper as cas
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.tasks.task import WEIGHT_ABOVE_CA, Task
from giskardpy.god_map import god_map


class InsertCylinder(Goal):
    def __init__(self,
                 name: str,
                 cylinder_name: str,
                 hole_point: cas.Point3,
                 cylinder_height: Optional[float] = None,
                 up: Optional[cas.Vector3] = None,
                 pre_grasp_height: float = 0.1,
                 tilt: float = np.pi / 10,
                 get_straight_after: float = 0.02):
        self.cylinder_name = cylinder_name
        self.get_straight_after = get_straight_after
        self.root = god_map.world.root_link_name
        self.tip = god_map.world.search_for_link_name(self.cylinder_name)
        super().__init__(name=name)
        if cylinder_height is None:
            self.cylinder_height = god_map.world.links[self.tip].collisions[0].height
        else:
            self.cylinder_height = cylinder_height
        self.tilt = tilt
        self.pre_grasp_height = pre_grasp_height
        self.root_P_hole = god_map.world.transform(self.root, hole_point)
        if up is None:
            up = cas.Vector3((0, 0, 1))
            up.reference_frame = self.root
        self.root_V_up = god_map.world.transform(self.root, up)

        self.weight = WEIGHT_ABOVE_CA

        root_P_hole = cas.Point3(self.root_P_hole)
        root_V_up = cas.Vector3(self.root_V_up)
        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_P_tip = root_T_tip.to_position()
        tip_P_cylinder_bottom = cas.Vector3([0, 0, self.cylinder_height / 2])
        root_P_cylinder_bottom = root_T_tip.dot(tip_P_cylinder_bottom)
        root_P_tip = root_P_tip + root_P_cylinder_bottom
        root_V_cylinder_z = root_T_tip.dot(cas.Vector3([0, 0, -1]))

        # %% straight line goal
        root_P_top = root_P_hole + root_V_up * self.pre_grasp_height
        distance_to_top = cas.euclidean_distance(root_P_tip, root_P_top)

        distance_to_line, root_P_on_line = cas.distance_point_to_line_segment(root_P_tip, root_P_hole, root_P_top)
        distance_to_hole = cas.norm(root_P_hole - root_P_tip)

        reach_top = Task(name='Reach Top')
        self.add_task(reach_top)
        reach_top.add_point_goal_constraints(frame_P_current=root_P_tip,
                                             frame_P_goal=root_P_top,
                                             reference_velocity=0.1,
                                             weight=self.weight)
        reach_top.expression = cas.less(distance_to_top, 0.01)

        # %% tilted orientation goal
        tilt_error = cas.angle_between_vector(root_V_cylinder_z, root_V_up)
        tilt_task = Task(name='Slightly Tilted')
        self.add_task(tilt_task)
        tilt_task.add_position_constraint(expr_current=tilt_error,
                                          expr_goal=self.tilt,
                                          reference_velocity=0.1,
                                          weight=self.weight)
        root_V_cylinder_z.vis_frame = self.tip
        tilt_task.expression = cas.less_equal(cas.abs(tilt_error - self.tilt), 0.01)

        init_done = cas.logic_and(reach_top.get_observation_state_expression(),
                                  tilt_task.get_observation_state_expression())

        reach_top.end_condition = init_done

        # %% move down
        stay_on_line = Task(name='Stay on Straight Line ')
        self.add_task(stay_on_line)
        stay_on_line.add_point_goal_constraints(frame_P_current=root_P_tip,
                                                frame_P_goal=root_P_on_line,
                                                reference_velocity=0.1,
                                                weight=self.weight,
                                                name='pregrasp')
        stay_on_line.expression = cas.less(distance_to_line, 0.01)

        insert_task = Task(name='Insert')
        self.add_task(insert_task)
        insert_task.add_point_goal_constraints(frame_P_current=root_P_tip,
                                               frame_P_goal=root_P_hole,
                                               reference_velocity=0.1,
                                               weight=self.weight,
                                               name='insertion')
        insert_task.start_condition = init_done
        insert_task.expression = cas.less(distance_to_hole, 0.01)

        bottom_reached = cas.logic_and(insert_task.get_observation_state_expression(),
                                       stay_on_line.get_observation_state_expression())

        tilt_task.end_condition = bottom_reached
        # %% tilt straight
        # tilt_monitor.expression = cas.less(tilt_error, 0.01)

        tilt_straight_task = Task(name='Tilt Straight')
        self.add_task(tilt_straight_task)
        tilt_straight_task.add_vector_goal_constraints(frame_V_current=root_V_cylinder_z,
                                                       frame_V_goal=root_V_up,
                                                       reference_velocity=0.1,
                                                       weight=self.weight)
        tilt_straight_task.start_condition = bottom_reached
        # tilt_straight_task.end_condition = tilt_monitor.get_observation_state_expression()
        tilt_straight_task.expression = cas.less_equal(tilt_error, 0.01)

        self.expression = tilt_straight_task.expression
