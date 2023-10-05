from typing import List, Dict, Optional

from geometry_msgs.msg import PointStamped, QuaternionStamped, PoseStamped

import giskardpy.casadi_wrapper as cas
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.god_map import god_map
from giskardpy.my_types import Derivatives


class PoseReached(Monitor):
    def __init__(self,
                 name: str,
                 root_link: str, tip_link: str,
                 goal_pose: PoseStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 position_threshold: float = 0.01,
                 orientation_threshold: float = 0.01,
                 crucial: bool = True):
        super().__init__(name, crucial=crucial)
        root_link = god_map.world.search_for_link_name(root_link, root_group)
        tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        goal_pose = self.transform_msg(root_link, goal_pose)
        r_T_g = cas.TransMatrix(goal_pose)

        # %% position error
        r_P_g = r_T_g.to_position()
        r_P_c = god_map.world.compose_fk_expression(root_link, tip_link).to_position()
        distance_to_goal = cas.euclidean_distance(r_P_g, r_P_c)
        position_reached = cas.less(distance_to_goal, position_threshold)

        # %% orientation error
        r_R_g = r_T_g.to_rotation()
        r_R_c = god_map.world.compose_fk_expression(root_link, tip_link).to_rotation()
        rotation_error = cas.rotational_error(r_R_c, r_R_g)
        orientation_reached = cas.less(cas.abs(rotation_error), orientation_threshold)

        self.set_expression(cas.logic_and(position_reached, orientation_reached))


class PositionReached(Monitor):
    def __init__(self,
                 name: str,
                 root_link: str, tip_link: str,
                 goal_point: PointStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 threshold: float = 0.01,
                 crucial: bool = True):
        super().__init__(name, crucial=crucial)
        root_link = god_map.world.search_for_link_name(root_link, root_group)
        tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        goal_point = self.transform_msg(root_link, goal_point)
        r_P_g = cas.Point3(goal_point)
        r_P_c = god_map.world.compose_fk_expression(root_link, tip_link).to_position()
        distance_to_goal = cas.euclidean_distance(r_P_g, r_P_c)
        self.set_expression(cas.less(distance_to_goal, threshold))


class OrientationReached(Monitor):
    def __init__(self,
                 name: str,
                 root_link: str,
                 tip_link: str,
                 goal_orientation: QuaternionStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 threshold: float = 0.01,
                 crucial: bool = True):
        super().__init__(name, crucial=crucial)
        root_link = god_map.world.search_for_link_name(root_link, root_group)
        tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        goal_orientation = self.transform_msg(root_link, goal_orientation)
        r_R_g = cas.RotationMatrix(goal_orientation)
        r_R_c = god_map.world.compose_fk_expression(root_link, tip_link).to_rotation()
        rotation_error = cas.rotational_error(r_R_c, r_R_g)
        self.set_expression(cas.less(cas.abs(rotation_error), threshold))
