from typing import Union, Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import MonitorInitalizationException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.monitors.monitors import PayloadMonitor
from giskardpy.qp.qp_solver_ids import SupportedQPSolver




class SetQPSolver(PayloadMonitor):

    def __init__(self, qp_solver_id: Union[SupportedQPSolver, int], name: Optional[str] = None,
                 start_condition: cas.Expression = cas.BinaryTrue,
                 pause_condition: cas.Expression = cas.BinaryFalse,
                 end_condition: cas.Expression = cas.BinaryFalse):
        if not cas.is_true_symbol(start_condition):
            raise MonitorInitalizationException(f'{self.__class__.__name__}: start_condition must be True.')
        if name is None:
            name = self.__class__.__name__
        super().__init__(run_call_in_thread=False, name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)
        qp_solver_id = SupportedQPSolver(qp_solver_id)
        god_map.qp_controller.set_qp_solver(qp_solver_id)

    def __call__(self):
        self.state = True
