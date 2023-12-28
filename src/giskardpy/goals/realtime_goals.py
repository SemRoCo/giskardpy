from __future__ import division

from typing import Optional, List

import rospy
from geometry_msgs.msg import Vector3Stamped, PointStamped

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal
from giskardpy.goals.monitors.monitors import ExpressionMonitor
from giskardpy.goals.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.pointing import Pointing
from giskardpy.utils.expression_definition_utils import transform_msg


class RealTimePointing(Pointing):

    def __init__(self,
                 tip_link: str,
                 root_link: str,
                 tip_group: Optional[str] = None,
                 root_group: Optional[str] = None,
                 pointing_axis: Vector3Stamped = None,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA,
                 start_monitors: Optional[List[ExpressionMonitor]] = None,
                 hold_monitors: Optional[List[ExpressionMonitor]] = None,
                 end_monitors: Optional[List[ExpressionMonitor]] = None
                 ):
        initial_goal = PointStamped()
        initial_goal.header.frame_id = 'base_footprint'
        initial_goal.point.x = 1
        initial_goal.point.z = 1
        super().__init__(tip_link=tip_link,
                         goal_point=initial_goal,
                         root_link=root_link,
                         pointing_axis=pointing_axis)
        self.sub = rospy.Subscriber('muh', PointStamped, self.cb)

    def cb(self, data: PointStamped):
        data = transform_msg(self.root, data)
        self.root_P_goal_point = data

