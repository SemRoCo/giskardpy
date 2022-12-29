from __future__ import division

from typing import Optional

import rospy
from geometry_msgs.msg import Vector3Stamped, PointStamped

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.goals.pointing import Pointing


class RealTimePointing(Pointing):

    def __init__(self,
                 tip_link: str,
                 root_link: str,
                 tip_group: Optional[str] = None,
                 root_group: Optional[str] = None,
                 pointing_axis: Vector3Stamped = None,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA):

        super().__init__(tip_link=tip_link,
                         goal_point=PointStamped(),
                         root_link=root_link,
                         pointing_axis=pointing_axis)
        self.sub = rospy.Subscriber('muh', PointStamped, self.cb)

    def cb(self, data: PointStamped):
        self.root_P_goal_point = data

