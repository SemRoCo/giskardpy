import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.god_map_user import GodMapWorshipper


class MonitorManager(GodMapWorshipper):
    compiled_monitors: CompiledFunction
    state: np.ndarray
    switch_filter = np.ndarray
    switch_state = np.ndarray
    crucial_filter = np.ndarray

    def compile_monitors(self):
        expressions = []
        for monitor in self.monitors:
            expressions.append(monitor.get_expression())
        expressions = cas.Expression(expressions)
        self.compiled_monitors = expressions.compile(expressions.free_symbols())
        self.filter_mask = np.array([1 if x.stay_one else 0 for x in self.monitors], dtype=int)
        self.state = np.zeros_like(self.filter_mask)
        self.switches_state = np.zeros_like(self.filter_mask)
        self.crucial_filter = [m.crucial for m in self.monitors]

    def update_state(self, new_state):  # Assuming new_state is a NumPy array with only 1 and 0
        new_state = new_state.astype(int)
        self.switches_state[self.filter_mask] = self.switches_state[self.filter_mask] | new_state[self.filter_mask]
        self.state = self.switches_state | new_state

    def evaluate_monitors(self):
        args = self.god_map.get_values(self.compiled_monitors.str_params)
        self.update_state(self.compiled_monitors.fast_call(args))

    def crucial_monitors_satisfied(self):
        return np.all(self.state[self.crucial_filter])
