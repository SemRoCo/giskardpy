from typing import List, Dict, Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.goals.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.god_map import god_map
from giskardpy.my_types import Derivatives


class JointGoalReached(ExpressionMonitor):
    def __init__(self,
                 name: str,
                 goal_state: Dict[str, float],
                 threshold: float,
                 stay_one: bool = True,
                 start_monitors: Optional[List[Monitor]] = None):
        comparison_list = []
        for joint_name, goal in goal_state.items():
            joint_name = god_map.world.search_for_joint_name(joint_name)
            current = god_map.world.get_one_dof_joint_symbol(joint_name, Derivatives.position)
            if god_map.world.is_joint_continuous(joint_name):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current
            # god_map.debug_expression_manager.add_debug_expression(str(joint_name), cas.min(cas.abs(error), 0.01))
            comparison_list.append(cas.less(cas.abs(error), threshold))
        expression = cas.logic_all(cas.Expression(comparison_list))
        super().__init__(name, stay_one=stay_one, start_monitors=start_monitors)
        self.set_expression(expression)
        god_map.debug_expression_manager.add_debug_expression(f'joints reached', self.get_expression())

