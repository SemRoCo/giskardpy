from typing import Optional
import giskardpy.casadi_wrapper as cas
from giskardpy.goals.monitors.monitors import SwitchMonitor, MonitorInterface
from giskardpy.goals.tasks.task import Task


class JointPositionTask(Task):
    def __init__(self, joint_current: cas.symbol_expr_float,
                 joint_goal: cas.symbol_expr_float, weight: cas.symbol_expr_float,
                 velocity_limit: cas.symbol_expr_float,
                 to_start: Optional[SwitchMonitor] = None, to_hold: Optional[MonitorInterface] = None,
                 to_end: Optional[SwitchMonitor] = None):
        error = joint_goal - joint_current
        to_end = SwitchMonitor(cas.less(cas.abs(error), 0.1))

        super().__init__(None, to_start, to_hold, to_end)

        self.add_equality_constraint(reference_velocity=velocity_limit,
                                     equality_bound=error,
                                     weight=weight,
                                     task_expression=joint_current)
