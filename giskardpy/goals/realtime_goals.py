from __future__ import division

from typing import Optional

import rospy
from geometry_msgs.msg import PointStamped

from giskardpy.data_types.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.tasks.task import WEIGHT_BELOW_CA
from giskardpy.goals.pointing import Pointing
import giskardpy.casadi_wrapper as cas
import giskardpy.middleware.ros1.msg_converter as msg_converter


class RealTimePointing(Pointing):

    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 pointing_axis: cas.Vector3,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        initial_goal = cas.Point3((1, 0, 1), reference_frame=god_map.world.search_for_link_name('base_footprint'))
        super().__init__(tip_link=tip_link,
                         goal_point=initial_goal,
                         root_link=root_link,
                         pointing_axis=pointing_axis,
                         max_velocity=max_velocity,
                         weight=weight,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition)
        self.sub = rospy.Subscriber('muh', PointStamped, self.cb)

    def cb(self, data: PointStamped):
        data = msg_converter.ros_msg_to_giskard_obj(data, god_map.world)
        data = god_map.world.transform(self.root, data).to_np()
        self.root_P_goal_point = data
