from typing import Optional, List

from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from geometry_msgs.msg import PointStamped
from giskardpy.utils.expression_definition_utils import transform_msg_and_turn_to_expr
from giskardpy.data_types import Derivatives


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
                 end_condition: cas.Expression = cas.TrueSymbol):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)

        task = self.create_and_add_task('weight_scaling')
        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        root_P_goal = transform_msg_and_turn_to_expr(self.root_link, tip_goal, cas.TrueSymbol)
        scaling_exp = root_P_goal - root_P_tip

        free_variables = []
        gains = {}
        for name in arm_joints:
            vs = god_map.world.joints[god_map.world.search_for_joint_name(name)].free_variables
            for v in vs:
                free_variables.append(v.name)
                v_gain = gain * cas.norm(scaling_exp / v.get_upper_limit(Derivatives.velocity))
                gains[v.name] = v_gain
        for name in base_joints:
            vs = god_map.world.joints[god_map.world.search_for_joint_name(name)].free_variables
            for v in vs:
                free_variables.append(v.name)
                v_gain = gain * cas.save_division(1, cas.norm(scaling_exp / v.get_upper_limit(Derivatives.velocity)))
                gains[v.name] = v_gain

        task.add_quadratic_weight_gain('baseToArmScaling',
                                       free_variable_names=free_variables,
                                       gains=gains)
