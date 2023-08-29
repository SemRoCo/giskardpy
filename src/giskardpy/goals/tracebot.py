from __future__ import division

from typing import Optional

import numpy as np
from geometry_msgs.msg import Vector3Stamped, PointStamped

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.goals.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.monitors.joint_monitors import PositionMonitor
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.tasks.joint_tasks import PositionTask
from giskardpy.goals.tasks.task import Task


class InsertCylinder(Goal):
    def __init__(self,
                 cylinder_name: str,
                 hole_point: PointStamped,
                 cylinder_height: Optional[float] = None,
                 up: Vector3Stamped = None,
                 pre_grasp_height: float = 0.1,
                 tilt: float = np.pi / 10,
                 get_straight_after: float = 0.02):
        super().__init__()
        self.cylinder_name = cylinder_name
        self.get_straight_after = get_straight_after
        self.root = self.world.root_link_name
        self.tip = self.world.search_for_link_name(self.cylinder_name)
        if cylinder_height is None:
            self.cylinder_height = self.world.links[self.tip].collisions[0].height
        else:
            self.cylinder_height = cylinder_height
        self.tilt = tilt
        self.pre_grasp_height = pre_grasp_height
        self.root_P_hole = self.transform_msg(self.root, hole_point)
        if up is None:
            up = Vector3Stamped()
            up.header.frame_id = self.root
            up.vector.z = 1
        self.root_V_up = self.transform_msg(self.root, up)

        self.weight = WEIGHT_ABOVE_CA

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root}/{self.tip}'

    def make_constraints(self):
        root_P_hole = cas.Point3(self.root_P_hole)
        root_V_up = cas.Vector3(self.root_V_up)
        root_T_tip = self.get_fk(self.root, self.tip)
        root_P_tip = root_T_tip.to_position()
        tip_P_cylinder_bottom = cas.Vector3([0, 0, self.cylinder_height / 2])
        root_P_cylinder_bottom = root_T_tip.dot(tip_P_cylinder_bottom)
        root_P_tip = root_P_tip + root_P_cylinder_bottom
        root_V_cylinder_z = root_T_tip.dot(cas.Vector3([0, 0, -1]))

        # straight line goal
        root_P_top = root_P_hole + root_V_up * self.pre_grasp_height
        distance_to_top = cas.euclidean_distance(root_P_tip, root_P_top)
        top_reached = cas.less(distance_to_top, 0.01)
        top_reached_monitor = Monitor(name='top reached', crucial=False, stay_one=True)
        self.add_monitor(top_reached_monitor)
        top_reached_monitor.set_expression(top_reached)

        distance_to_line, root_P_on_line = cas.distance_point_to_line_segment(root_P_tip, root_P_hole, root_P_top)
        distance_to_hole = cas.norm(root_P_hole - root_P_tip)
        bottom_reached = cas.less(distance_to_hole, 0.01)
        bottom_reached_monitor = Monitor('bottom reached', crucial=True, stay_one=True)
        bottom_reached_monitor.set_expression(bottom_reached)
        self.add_monitor(bottom_reached_monitor)

        reach_top = Task(name='reach top', to_end=top_reached_monitor)
        reach_top.add_point_goal_constraints(frame_P_current=root_P_tip,
                                             frame_P_goal=root_P_top,
                                             reference_velocity=0.1,
                                             weight=self.weight)
        self.add_task(reach_top)

        go_to_line = Task(name='straight line', to_start=top_reached_monitor)
        go_to_line.add_point_goal_constraints(frame_P_current=root_P_tip,
                                              frame_P_goal=root_P_on_line,
                                              reference_velocity=0.1,
                                              weight=self.weight,
                                              name='pregrasp')
        self.add_task(go_to_line)
        # self.add_debug_expr('root_P_goal', root_P_goal)
        # self.add_debug_expr('root_P_tip', root_P_tip)
        # self.add_debug_expr('weight_pregrasp', weight_pregrasp)

        # tilted orientation goal
        angle = cas.angle_between_vector(root_V_cylinder_z, root_V_up)
        tilt_task = Task(name='tilted', to_end=bottom_reached_monitor)
        tilt_task.add_position_constraint(expr_current=angle,
                                          expr_goal=self.tilt,
                                          reference_velocity=0.1,
                                          weight=self.weight)
        root_V_cylinder_z.vis_frame = self.tip
        self.add_task(tilt_task)
        # self.add_debug_expr('root_V_cylinder_z', root_V_cylinder_z)

        # # move down
        insert_task = Task(name='insert', to_start=top_reached_monitor)
        insert_task.add_point_goal_constraints(frame_P_current=root_P_tip,
                                               frame_P_goal=root_P_hole,
                                               reference_velocity=0.1,
                                               weight=self.weight,
                                               name='insertion')
        self.add_task(insert_task)
        # self.add_debug_expr('root_P_hole', root_P_hole)
        # self.add_debug_expr('weight_insert', weight_insert)
        #
        # # tilt straight
        tilt_error = cas.angle_between_vector(root_V_cylinder_z, root_V_up)
        tilt_monitor = Monitor(name='straight', crucial=True)
        tilt_monitor.set_expression(cas.less(tilt_error, 0.01))
        self.add_monitor(tilt_monitor)

        tilt_straight_task = Task(name='tilt straight', to_start=bottom_reached_monitor, to_end=tilt_monitor)
        tilt_straight_task.add_vector_goal_constraints(frame_V_current=root_V_cylinder_z,
                                                       frame_V_goal=root_V_up,
                                                       reference_velocity=0.1,
                                                       weight=self.weight)
        self.add_task(tilt_straight_task)
