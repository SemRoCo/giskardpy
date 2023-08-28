import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.god_map_user import GodMapWorshipper


def custom_op_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    0, 0 -> 0
    0, 1 -> 1
    1, 0 -> 0
    1, 1 -> 0
    """
    return np.logical_and(np.logical_not(a), np.logical_or(a, b))





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
        self.stay_one_filter = np.array([x.stay_one for x in self.monitors], dtype=bool)
        self.state = np.zeros_like(self.stay_one_filter)
        self.switches_state = np.zeros_like(self.stay_one_filter)
        self.crucial_filter = [m.crucial for m in self.monitors]

    def update_state(self, new_state):  # Assuming new_state is a NumPy array with only 1 and 0
        new_state = new_state.astype(int)
        filtered_switches = self.switches_state[self.stay_one_filter]
        filtered_new_state = new_state[self.stay_one_filter]
        new_flips = custom_op_numpy(filtered_switches, filtered_new_state)
        if np.any(new_flips):
            self.trigger_monitor_flips(new_flips)
        self.switches_state[self.stay_one_filter] = filtered_switches | filtered_new_state
        self.state = self.switches_state | new_state

    def trigger_monitor_flips(self, flips: np.ndarray):
        flipped_monitors = np.array(self.monitors)[self.stay_one_filter][flips]
        for m in flipped_monitors:
            m.update_substitution_values()

    def evaluate_monitors(self):
        args = self.god_map.get_values(self.compiled_monitors.str_params)
        self.update_state(self.compiled_monitors.fast_call(args))

    def crucial_monitors_satisfied(self):
        return np.all(self.state[self.crucial_filter])
