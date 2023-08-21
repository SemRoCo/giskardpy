from __future__ import division

from typing import Optional

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from tf.transformations import rotation_from_matrix

from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.model.joints import DiffDrive, OmniDrivePR22
from giskardpy.my_types import Derivatives
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import normalize
from giskardpy import identifier
from giskardpy.goals.cartesian_goals import CartesianOrientation
from giskardpy.goals.pouring_goals import KeepObjectAbovePlane
from giskardpy.goals.pouring_goals import TiltObject


class PouringAction(Goal):
    def __init__(self, tip_link: str, root_link: str, upright_orientation: QuaternionStamped,
                 down_orientation: QuaternionStamped, container_plane: PointStamped,
                 tip_group: str = None, root_group: str = None,
                 max_velocity: float = 0.3, weight: float = WEIGHT_ABOVE_CA):
        super(PouringAction, self).__init__()
        self.root_link = self.world.search_for_link_name(root_link, root_group)
        self.tip_link2 = self.world.search_for_link_name('hand_camera_frame', root_group)
        self.tip_link = self.world.search_for_link_name(tip_link, tip_group)
        self.max_vel = max_velocity
        self.weight = weight
        self.upright_orientation = upright_orientation
        self.down_orientation = down_orientation
        self.container_plane = container_plane
        self.root_group = root_group
        self.tip_group = tip_group

    def make_constraints(self):
        root_T_tip = self.get_fk(self.root_link, self.tip_link)
        root_P_tip = root_T_tip.to_position()

        is_forward = self.god_map.to_expr(identifier.pouring_forward)
        is_left = self.god_map.to_expr(identifier.pouring_left)
        is_up = self.god_map.to_expr(identifier.pouring_up)
        is_backward = self.god_map.to_expr(identifier.pouring_backward)
        is_right = self.god_map.to_expr(identifier.pouring_right)
        is_down = self.god_map.to_expr(identifier.pouring_down)

        is_translation = w.min(1, is_forward + is_left + is_up + is_backward + is_right + is_down)
        self.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                            equality_bounds=[
                                                self.max_vel * is_forward + self.max_vel * -1 * is_backward,
                                                self.max_vel * is_left + self.max_vel * -1 * is_right,
                                                self.max_vel * is_up + self.max_vel * -1 * is_down],
                                            weights=[is_translation] * 3,
                                            task_expression=root_P_tip[:3],
                                            names=['forward-back', 'left-right', 'up-down'])

        is_uprigth = self.god_map.to_expr(identifier.pouring_keep_upright)
        self.add_constraints_of_goal(CartesianOrientation(root_link=self.root_link,
                                                          root_group=self.root_group,
                                                          tip_link=self.tip_link,
                                                          tip_group=self.tip_group,
                                                          goal_orientation=self.upright_orientation,
                                                          max_velocity=self.max_vel,
                                                          reference_velocity=self.max_vel,
                                                          weight=self.weight * is_uprigth,
                                                          name_extra='keep_upright'))

        is_tilt = self.god_map.to_expr(identifier.pouring_tilt)
        self.add_constraints_of_goal(CartesianOrientation(root_link=self.root_link,
                                                          root_group=self.root_group,
                                                          tip_link=self.tip_link,
                                                          tip_group=self.tip_group,
                                                          goal_orientation=self.down_orientation,
                                                          max_velocity=self.max_vel,
                                                          reference_velocity=self.max_vel,
                                                          weight=self.weight * is_tilt,
                                                          name_extra='tilt'))

        lower_distance = 0.2
        upper_distance = 0.3
        plane_radius = 0
        is_move_to = self.god_map.to_expr(identifier.pouring_move_to)
        self.add_constraints_of_goal(KeepObjectAbovePlane(object_link=self.tip_link,
                                                          plane_center_point=self.container_plane,
                                                          lower_distance=lower_distance,
                                                          upper_distance=upper_distance,
                                                          plane_radius=plane_radius,
                                                          root_link=self.root_link,
                                                          weight=self.weight * is_move_to))

        self.add_debug_expr('forward', is_forward)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'
