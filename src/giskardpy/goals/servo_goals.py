import math

import rospy

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.tasks.task import Task, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.goal import Goal
from typing import Optional
from geometry_msgs.msg import Vector3Stamped, PointStamped, PoseStamped
from giskardpy.goals.cartesian_goals import CartesianOrientation, RotationVelocityLimit
from giskardpy.goals.align_planes import AlignPlanes
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from visualization_msgs.msg import Marker


class VisualServoPointGoal(Goal):
    def __init__(self, root, tip, goal_point: PointStamped, max_vel=0.3, weight=WEIGHT_BELOW_CA):
        super().__init__()
        self.point_sub = rospy.Subscriber('/point_feature', PointStamped, self.callback)
        self.point: PointStamped = PointStamped()
        self.root_link = god_map.world.search_for_link_name(root, None)
        self.tip_link = god_map.world.search_for_link_name(tip, None)
        self.goal_point: w.Point3 = w.Point3(god_map.world.transform_point(self.root_link, goal_point))
        self.max_vel = max_vel
        self.weight = weight
        task = Task(name='visualServoTask')
        map_P_point = w.Point3(
            symbol_manager.get_expr(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].point',
                                    output_type_hint=w.Point3, input_type_hint=PointStamped))
        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        s_dot_desired = self.goal_point - map_P_point

        task.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                            equality_bounds=s_dot_desired[:3],
                                            weights=[self.weight] * 3,
                                            task_expression=root_T_tip.to_position()[:3],
                                            names=['pointServoX', 'pointServoY', 'pointServoZ'])
        self.add_task(task)

    def callback(self, point: PointStamped):
        self.point = point

    def __str__(self) -> str:
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'
