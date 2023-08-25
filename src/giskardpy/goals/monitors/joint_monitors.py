from typing import List

import giskardpy.casadi_wrapper as cas
from giskardpy.goals.monitors.monitors import Monitor


class PositionMonitor(Monitor):
    def __init__(self,
                 current_positions: List[cas.symbol_expr_float],
                 goal_positions: List[cas.symbol_expr_float],
                 thresholds: List[cas.symbol_expr_float],
                 crucial: bool):
        comparison_list = []
        for current, goal, threshold in zip(current_positions, goal_positions, thresholds):
            comparison_list.append(cas.less(cas.abs(goal - current), threshold))
        expression = cas.logic_all(cas.Expression(comparison_list))
        super().__init__(expression, crucial)


