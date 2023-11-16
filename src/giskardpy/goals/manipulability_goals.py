

from typing import Optional, List

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from tf.transformations import rotation_from_matrix

from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE, Task
from giskardpy.god_map import god_map
from giskardpy.model.joints import DiffDrive, OmniDrivePR22
from giskardpy.my_types import Derivatives
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils import logging
from giskardpy.utils.expression_definition_utils import transform_msg_and_turn_to_expr, transform_msg
from giskardpy.utils.tfwrapper import normalize
from giskardpy.utils.utils import split_pose_stamped


class MaxManipulability(Goal):
    def __init__(self, root_link: str, tip_link: str,
                 name: Optional[str] = None,
                 to_start: Optional[List[Monitor]] = None,
                 to_hold: Optional[List[Monitor]] = None,
                 to_end: Optional[List[Monitor]] = None
                 ):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        self.gain = 2
        task = Task(name='manipulability')
        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        root_P_goal = cas.Point3([0, 0, 0])
        task.add_point_goal_constraints(frame_P_goal=root_P_goal,
                                        frame_P_current=root_T_tip.to_position(),
                                        reference_velocity=0.2,
                                        weight=0)
        self.add_task(task)
        self.connect_monitors_to_all_tasks(to_start=to_start, to_hold=to_hold, to_end=to_end)
    """
    This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
    This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
    Currently only maximizes the translational and not the rotational manipulability.
    The expressions this goal writes onto the god_map will be used in the qp building logic to adapt the linear weights 
    of the QP. To be more precise, first the Jacobian of the expressions is calculated and then the gradient of 
    the jacobian is calculated. The jacobian is then evaluated and used to calculate the manipulability index
    m=sqrt(det(JJ^T)). The index is then multiplied with the gradient of J. Finally, the linear weight is calculated by 
    multiplying this with a negative gain value.
    Future work makes it possible to set the gain through the motion goal and to monitor the m index.
    """