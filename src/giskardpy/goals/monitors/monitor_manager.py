import traceback
from typing import List

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.exceptions import GiskardException, UnknownConstraintException, ConstraintInitalizationException
from giskardpy.goals.monitors.monitors import Monitor
import giskard_msgs.msg as giskard_msgs
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils import logging
from giskardpy.utils.utils import json_to_kwargs, get_all_classes_in_package, json_str_to_kwargs
from giskardpy.goals.monitors.monitors import LocalMinimumReached

def flipped_to_one(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    0, 0 -> 0
    0, 1 -> 1
    1, 0 -> 0
    1, 1 -> 0
    """
    return np.logical_and(np.logical_not(a), np.logical_or(a, b))


class MonitorManager:
    compiled_monitors: CompiledFunction
    state: np.ndarray
    switch_filter = np.ndarray
    switch_state = np.ndarray
    crucial_filter = np.ndarray
    monitors: List[Monitor] = None

    def __init__(self):
        self.monitors = []
        self.allowed_monitor_types = {}
        self.allowed_monitor_types.update(get_all_classes_in_package('giskardpy.goals.monitors', Monitor))
        self.robot_names = god_map.collision_scene.robot_names
        self.local_minimum_monitor_id = None

    def compile_monitors(self):
        expressions = []
        for monitor in self.monitors:
            monitor.compile()
            expressions.append(monitor.get_expression())
        expressions = cas.Expression(expressions)
        self.compiled_monitors = expressions.compile(expressions.free_symbols())
        self.stay_one_filter = np.array([x.stay_one for x in self.monitors], dtype=bool)
        self.state = np.zeros_like(self.stay_one_filter)
        self.switches_state = np.zeros_like(self.stay_one_filter)
        self.crucial_filter = [m.crucial for m in self.monitors]

        local_minimum_monitors = [i for i, x in enumerate(self.monitors) if isinstance(x, LocalMinimumReached)]
        if local_minimum_monitors:
            self.local_minimum_monitor_id = local_minimum_monitors[0]
        else:
            self.local_minimum_monitor_id = None

    def get_monitor(self, name: str) -> Monitor:
        for monitor in self.monitors:
            if monitor.name == name:
                return monitor
        raise KeyError(f'No monitor of name {name} found.')

    def is_local_minimum_reached(self):
        if self.local_minimum_monitor_id is None:
            return False
        return bool(self.state[self.local_minimum_monitor_id])

    @profile
    def update_state(self, new_state):  # Assuming new_state is a NumPy array with only 1 and 0
        new_state = new_state.astype(int)
        filtered_switches = self.switches_state[self.stay_one_filter]
        filtered_new_state = new_state[self.stay_one_filter]
        new_flips = flipped_to_one(filtered_switches, filtered_new_state)
        if np.any(new_flips):
            self.trigger_monitor_flips(new_flips)
        self.switches_state[self.stay_one_filter] = filtered_switches | filtered_new_state
        next_state = self.switches_state | new_state
        any_flips = np.logical_xor(self.state, next_state)
        if np.any(any_flips):
            for i, state in enumerate(any_flips):
                if state:
                    self.monitors[i].notify_flipped(god_map.time)
        self.state = next_state

    def add_monitor(self, monitor: Monitor):
        if [x for x in self.monitors if x.name == monitor.name]:
            raise GiskardException(f'Monitor named {monitor.name} already exists.')
        self.monitors.append(monitor)
        monitor.set_id(len(self.monitors) - 1)

    def get_state_dict(self, only_crucial: bool = False):
        return {monitor.name: bool(self.state[i]) for i, monitor in enumerate(self.monitors)
                if not only_crucial or monitor.crucial}

    @profile
    def trigger_monitor_flips(self, flips: np.ndarray):
        flipped_monitors = np.array(self.monitors)[self.stay_one_filter][flips]
        for m in flipped_monitors:
            m.update_substitution_values()

    @profile
    def evaluate_monitors(self):
        args = symbol_manager.resolve_symbols(self.compiled_monitors.str_params)
        self.update_state(self.compiled_monitors.fast_call(args))

    @profile
    def crucial_monitors_satisfied(self):
        return np.all(self.state[self.crucial_filter])


    @profile
    def parse_monitors(self, monitor_msgs: List[giskard_msgs.Monitor]):
        for monitor_msg in monitor_msgs:
            try:
                logging.loginfo(f'Adding monitor of type: \'{monitor_msg.type}\'')
                C = self.allowed_monitor_types[monitor_msg.type]
            except KeyError:
                raise UnknownConstraintException(f'unknown monitor type: \'{monitor_msg.type}\'.')
            try:
                params = json_str_to_kwargs(monitor_msg.parameter_value_pair)
                monitor: Monitor = C(monitor_msg.name, **params)
                self.add_monitor(monitor)
            except Exception as e:
                traceback.print_exc()
                error_msg = f'Initialization of \'{C.__name__}\' constraint failed: \n {e} \n'
                if not isinstance(e, GiskardException):
                    raise ConstraintInitalizationException(error_msg)
                raise e
