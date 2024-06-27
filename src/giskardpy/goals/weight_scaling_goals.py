from typing import Optional, List

from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from geometry_msgs.msg import PointStamped
from giskardpy.utils.expression_definition_utils import transform_msg_and_turn_to_expr
from giskardpy.data_types import Derivatives
from giskardpy.tasks.task import WEIGHT_ABOVE_CA
from collections import defaultdict


class BaseArmWeightScaling(Goal):
    """
    This goals adds weight scaling constraints with the distance between a tip_link and its goal Position as a
    scaling expression. The larger the scaling expression the more is the base movement used toa achieve
    all other constraints instead of arm movements. When the expression decreases this relation changes to favor
    arm movements instead of base movements.
    """

    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 tip_goal: PointStamped,
                 arm_joints: List[str],
                 base_joints: List[str],
                 gain: float = 100000,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)

        task = self.create_and_add_task('weight_scaling')
        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        root_P_goal = transform_msg_and_turn_to_expr(self.root_link, tip_goal, cas.TrueSymbol)
        scaling_exp = root_P_goal - root_P_tip

        list_gains = []
        for t in range(god_map.qp_controller_config.prediction_horizon):
            gains = defaultdict(dict)
            arm_v = None
            for name in arm_joints:
                vs = god_map.world.joints[god_map.world.search_for_joint_name(name)].free_variables
                for v in vs:
                    v_gain = gain * cas.norm(scaling_exp / v.get_upper_limit(Derivatives.velocity))
                    arm_v = v
                    gains[Derivatives.velocity][v] = v_gain
                    gains[Derivatives.acceleration][v] = v_gain
                    gains[Derivatives.jerk][v] = v_gain
            base_v = None
            for name in base_joints:
                vs = god_map.world.joints[god_map.world.search_for_joint_name(name)].free_variables
                for v in vs:
                    v_gain = gain / 100 * cas.save_division(1, cas.norm(scaling_exp / v.get_upper_limit(Derivatives.velocity)))
                    base_v = v
                    gains[Derivatives.velocity][v] = v_gain
                    gains[Derivatives.acceleration][v] = v_gain
                    gains[Derivatives.jerk][v] = v_gain
            list_gains.append(gains)

        god_map.debug_expression_manager.add_debug_expression('base_scaling', gain * cas.save_division(1, cas.norm(scaling_exp / base_v.get_upper_limit(Derivatives.velocity))))
        god_map.debug_expression_manager.add_debug_expression('arm_scaling', gain * cas.norm(scaling_exp / arm_v.get_upper_limit(Derivatives.velocity)))
        god_map.debug_expression_manager.add_debug_expression('norm', cas.norm(scaling_exp))
        god_map.debug_expression_manager.add_debug_expression('division', 1 / cas.norm(scaling_exp))
        task.add_quadratic_weight_gain('baseToArmScaling',
                                       gains=list_gains)


class MaxManipulabilityLinWeight(Goal):
    """
       This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
       This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
       """
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 gain: float = 0.5,
                 name: Optional[str] = None,
                 prediction_horizon: int = 9,
                 m_threshold: float = 0.16,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)

        results = god_map.world.compute_split_chain(self.root_link, self.tip_link, True, True, False, False)
        for joint in results[2]:
            if 'joint' in joint and not god_map.world.is_joint_rotational(joint):
                raise Exception('Non rotational joint in kinematic chain of Maximize Manipulability Goal')

        task = self.create_and_add_task('MaxManipulability')
        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()[:3]

        J = cas.jacobian(root_P_tip, root_P_tip.free_symbols())
        JJT = J.dot(J.T)
        m = cas.sqrt(cas.det(JJT))
        list_gains = []
        for t in range(prediction_horizon):
            gains = defaultdict(dict)
            for symbol in root_P_tip.free_symbols():
                J_dq = cas.total_derivative(J, [symbol], [1])
                product = cas.matrix_inverse(JJT).dot(J_dq).dot(J.T)
                trace = cas.trace(product)
                v = self.get_free_variable(symbol)
                gains[Derivatives.velocity][v] = cas.if_greater(m, m_threshold, 0, trace * m * -gain)
                gains[Derivatives.acceleration][v] = cas.if_greater(m, m_threshold, 0, trace * m * -gain)
                gains[Derivatives.jerk][v] = cas.if_greater(m, m_threshold, 0, trace * m * -gain)
            list_gains.append(gains)
        task.add_linear_weight_gain(name, gains=list_gains)

        god_map.debug_expression_manager.add_debug_expression(f'mIndex{tip_link}', m)

    def get_free_variable(self, symbol):
        for f in god_map.world.free_variables:
            for d in Derivatives:
                if str(god_map.world.free_variables[f].get_symbol(d)) == str(symbol):
                    return god_map.world.free_variables[f]

