from typing import Optional

from geometry_msgs.msg import Vector3Stamped

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
from giskardpy.utils.logging import logwarn


class AlignPlanes(Goal):
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 goal_normal: Vector3Stamped,
                 tip_normal: Vector3Stamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 max_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA,
                 **kwargs):
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param goal_normal:
        :param tip_normal:
        :param root_group: if root_link is not unique, search in this group for matches.
        :param tip_group: if tip_link is not unique, search in this group for matches.
        :param max_angular_velocity: rad/s
        :param weight:
        """
        super().__init__()
        if 'root_normal' in kwargs:
            logwarn('Deprecated warning: use goal_normal instead of root_normal')
            goal_normal = kwargs['root_normal']
        self.root = self.world.search_for_link_name(root_link, root_group)
        self.tip = self.world.search_for_link_name(tip_link, tip_group)
        self.max_velocity = max_angular_velocity
        self.weight = weight

        self.tip_V_tip_normal = self.transform_msg(self.tip, tip_normal)
        self.tip_V_tip_normal.vector = tf.normalize(self.tip_V_tip_normal.vector)

        self.root_V_root_normal = self.transform_msg(self.root, goal_normal)
        self.root_V_root_normal.vector = tf.normalize(self.root_V_root_normal.vector)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root}/{self.tip}' \
               f'_X:{self.tip_V_tip_normal.vector.x}' \
               f'_Y:{self.tip_V_tip_normal.vector.y}' \
               f'_Z:{self.tip_V_tip_normal.vector.z}'

    def make_constraints(self):
        tip_V_tip_normal = w.Vector3(self.tip_V_tip_normal)
        root_R_tip = self.get_fk(self.root, self.tip).to_rotation()
        root_V_tip_normal = root_R_tip.dot(tip_V_tip_normal)
        root_V_root_normal = w.Vector3(self.root_V_root_normal)
        self.add_vector_goal_constraints(frame_V_current=root_V_tip_normal,
                                         frame_V_goal=root_V_root_normal,
                                         reference_velocity=self.max_velocity,
                                         weight=self.weight)
