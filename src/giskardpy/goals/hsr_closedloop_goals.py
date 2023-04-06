import math

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy import identifier
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from typing import Optional
from geometry_msgs.msg import Vector3Stamped, PointStamped, QuaternionStamped
from giskardpy.goals.cartesian_goals import CartesianOrientation, RotationVelocityLimit
from giskardpy.goals.align_planes import AlignPlanes


class MoveAlongUntilForce(Goal):
    def __init__(self,
                 tip_link: str,
                 move_direction: Vector3Stamped,
                 root_link: str,
                 force_threshold: float = 6,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA):
        """
        Aligns the axis of the object frame and the reference frame.
        Maybe an allowed error for deviation in degrees is introduced.
        """
        super().__init__()
        self.weight = weight
        self.max_velocity = max_velocity
        self.root = self.world.get_link_name(root_link, None)
        self.tip = self.world.get_link_name(tip_link, None)
        self.root_V_direction = self.transform_msg(self.root, move_direction)
        self.max_force = force_threshold

    def make_constraints(self):
        root_P_tip = self.get_fk(self.root, self.tip).to_position()
        error = w.Vector3(self.root_V_direction) * 0.01
        max_force = w.max(self.god_map.to_expr(identifier.ft_msg + ['wrench', 'force', 'x']),
                          self.god_map.to_expr(identifier.ft_msg + ['wrench', 'force', 'y']))
        max_force = w.max(max_force,
                          self.god_map.to_expr(identifier.ft_msg + ['wrench', 'force', 'z']))
        weight = w.if_greater(max_force, self.max_force, 0, WEIGHT_BELOW_CA)
        self.add_constraint_vector(reference_velocities=[self.max_velocity] * 3,
                                   lower_errors=error[:3],
                                   upper_errors=error[:3],
                                   weights=[weight] * 3,
                                   task_expression=root_P_tip[:3],
                                   names=['movex', 'movey', 'movez'])
        # TODO: test this on the real robot

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root}/{self.tip}'