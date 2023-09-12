from typing import List, Dict

import giskardpy.casadi_wrapper as cas
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.my_types import Derivatives


class JointGoalReached(Monitor):
    def __init__(self,
                 name: str,
                 goal_state: Dict[str, float],
                 threshold: float,
                 crucial: bool):
        comparison_list = []
        for joint_name, goal in goal_state.items():
            joint_name = self.world.search_for_joint_name(joint_name)
            current = self.world.get_one_dof_joint_symbol(joint_name, Derivatives.position)
            comparison_list.append(cas.less(cas.abs(goal - current), threshold))
        expression = cas.logic_all(cas.Expression(comparison_list))
        super().__init__(name, crucial=crucial)
        self.set_expression(expression)
