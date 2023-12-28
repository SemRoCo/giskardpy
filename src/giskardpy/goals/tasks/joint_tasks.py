from typing import Optional, List
import giskardpy.casadi_wrapper as cas
from giskardpy.goals.monitors.monitors import ExpressionMonitor
from giskardpy.goals.tasks.task import Task


class JointPositionTask(Task):
    def __init__(self, joint_current: cas.symbol_expr_float,
                 joint_goal: cas.symbol_expr_float, weight: cas.symbol_expr_float,
                 velocity_limit: cas.symbol_expr_float,
                 start_monitors: Optional[ExpressionMonitor] = None,
                 hold_monitors: Optional[ExpressionMonitor] = None,
                 end_monitors: Optional[ExpressionMonitor] = None):
        error = joint_goal - joint_current
        if end_monitors is None:
            end_monitors = ExpressionMonitor(cas.less(cas.abs(error), 0.1))

        super().__init__(None, start_monitors, hold_monitors, end_monitors)

        self.add_equality_constraint(reference_velocity=velocity_limit,
                                     equality_bound=error,
                                     weight=weight,
                                     task_expression=joint_current)


class PositionTask(Task):
    def __init__(self,
                 names: List[str],
                 current_positions: List[cas.symbol_expr_float],
                 goal_positions: List[cas.symbol_expr_float],
                 velocity_limits: List[cas.symbol_expr_float],
                 weight: cas.symbol_expr_float,
                 start_monitors: Optional[ExpressionMonitor] = None,
                 hold_monitors: Optional[ExpressionMonitor] = None,
                 end_monitors: Optional[ExpressionMonitor] = None):

        super().__init__(None, start_monitors, hold_monitors, end_monitors)
        for i in range(len(names)):
            name = names[i]
            current = current_positions[i]
            goal = goal_positions[i]
            velocity_limit = velocity_limits[i]
            error = goal - current
            self.add_equality_constraint(name=name,
                                         reference_velocity=velocity_limit,
                                         equality_bound=error,
                                         weight=weight,
                                         task_expression=current)
