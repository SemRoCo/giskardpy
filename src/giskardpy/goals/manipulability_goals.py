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
    def __init__(self, root_link: str, tip_link: str, gain: float = 0.5,
                 name: Optional[str] = None,
                 to_start: Optional[List[Monitor]] = None,
                 to_hold: Optional[List[Monitor]] = None,
                 to_end: Optional[List[Monitor]] = None
                 ):
        god_map.manip_gain = gain
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)

        task = Task(name='manipulability')
        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        task.add_manipulability_constraint_vector(names=['x', 'y', 'z'],
                                                  task_expressions=root_T_tip.to_position()[:3])
        self.add_task(task)
        m = symbol_manager.get_symbol(f'god_map.m_index[0]')
        old_m = symbol_manager.get_symbol(f'god_map.m_index[1]')
        god_map.debug_expression_manager.add_debug_expression('mIndex_percentualDifference',
                                                              1 - cas.min(cas.save_division(old_m, m), 1))
        percentual_diff = 1 - cas.min(cas.save_division(old_m, m), 1)
        monitor = Monitor(name=f'manipMonitor{tip_link}', crucial=True)
        monitor.set_expression(cas.less(percentual_diff, 0.01))
        task.add_to_end_monitor(monitor)
        self.add_monitor(monitor)

    """
    This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
    This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
    Currently only maximizes the translational and not the rotational manipulability.
    The expressions this goal writes onto the god_map will be used in the qp building logic to adapt the linear weights 
    of the QP. To be more precise, first the Jacobian of the expressions is calculated and then the gradient of 
    the jacobian is calculated. The jacobian is then evaluated and used to calculate the manipulability index
    m=sqrt(det(JJ^T)). The index is then multiplied with the gradient of J. Finally, the linear weight is calculated by 
    multiplying this with a negative gain value.
    Future work combines translation and rotation, and improves the monitoring to not stop to early during the main motion.
    """
