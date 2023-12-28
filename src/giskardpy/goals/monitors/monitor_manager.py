import traceback
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Callable, Union

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.exceptions import GiskardException, UnknownConstraintException, ConstraintInitalizationException
from giskardpy.goals.monitors.monitors import ExpressionMonitor, Monitor
import giskard_msgs.msg as giskard_msgs
from giskardpy.goals.monitors.payload_monitors import PayloadMonitor, EndMotion
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils import logging
from giskardpy.utils.utils import get_all_classes_in_package, json_str_to_kwargs


def flipped_to_one(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    0, 0 -> 0
    0, 1 -> 1
    1, 0 -> 0
    1, 1 -> 0
    """
    return np.logical_and(np.logical_not(a), np.logical_or(a, b))


def monitor_list_to_monitor_name_tuple(monitors: List[Union[str, ExpressionMonitor]]) -> Tuple[str, ...]:
    return tuple(sorted(monitor.name if isinstance(monitor, Monitor) else monitor for monitor in monitors))


class MonitorManager:
    compiled_monitors: CompiledFunction
    state: np.ndarray  # order: ExpressionMonitors, PayloadMonitors
    switch_filter = np.ndarray
    switch_state = np.ndarray
    expression_monitors: List[ExpressionMonitor]
    payload_monitors: List[PayloadMonitor]
    substitution_values: Dict[Tuple[str, ...], Dict[str, float]]
    triggers: Dict[Tuple[int, ...], Callable]

    def __init__(self):
        self.expression_monitors = []
        self.payload_monitors = []
        self.allowed_monitor_types = {}
        self.allowed_monitor_types.update(get_all_classes_in_package('giskardpy.goals.monitors',
                                                                     Monitor))
        self.substitution_values = {}
        self.triggers = {}

    def compile_expression_monitors(self):
        expressions = []
        for monitor in self.expression_monitors:
            monitor.compile()
            expressions.append(monitor.get_expression())
        expressions = cas.Expression(expressions)
        self.compiled_monitors = expressions.compile(expressions.free_symbols())
        self.stay_one_filter = np.array([x.stay_one for x in self.expression_monitors], dtype=bool)
        self.switches_state = np.zeros_like(self.stay_one_filter)

    def compile_payload_monitors(self):
        self.end_motion_ids = np.array([isinstance(monitor, EndMotion) for monitor in self.monitors])
        for monitor in self.payload_monitors:
            god_map.tree.control_loop_branch.check_monitors.add_monitor(monitor)

    def compile_monitors(self):
        self.state = np.zeros(len(self.monitors))
        self.compile_expression_monitors()
        self.compile_payload_monitors()
        self._register_expression_update_triggers()

    def get_monitor(self, name: str) -> ExpressionMonitor:
        for monitor in self.monitors:
            if monitor.name == name:
                return monitor
        raise KeyError(f'No monitor of name {name} found.')

    @profile
    def update_expr_monitor_state(self, new_state) -> np.ndarray:  # Assuming new_state is a NumPy array with only 1 and 0
        new_state = new_state.astype(int)
        filtered_switches = self.switches_state[self.stay_one_filter]
        filtered_new_state = new_state[self.stay_one_filter]
        # new_flips = flipped_to_one(filtered_switches, filtered_new_state)
        self.switches_state[self.stay_one_filter] = filtered_switches | filtered_new_state
        next_state = self.switches_state | new_state
        any_flips = np.logical_xor(new_state, next_state)
        if np.any(any_flips):
            for i, state in enumerate(any_flips):
                if state:
                    self.expression_monitors[i].notify_flipped(god_map.time)
        return next_state

    @property
    def monitors(self):
        return self.expression_monitors + self.payload_monitors

    def add_expression_monitor(self, monitor: ExpressionMonitor):
        if [x for x in self.monitors if x.name == monitor.name]:
            raise GiskardException(f'Monitor named {monitor.name} already exists.')
        self.expression_monitors.append(monitor)
        monitor.set_id(len(self.expression_monitors) - 1)
        # increase all payload monitor ids because expression monitors always come first
        for payload_monitor in self.payload_monitors:
            payload_monitor.set_id(payload_monitor.id + 1)

    def add_payload_monitor(self, monitor: PayloadMonitor):
        if [x for x in self.monitors if x.name == monitor.name]:
            raise GiskardException(f'Monitor named {monitor.name} already exists.')
        self.payload_monitors.append(monitor)
        monitor.set_id(len(self.monitors) - 1)

    def get_state_dict(self) -> Dict[str, bool]:
        return OrderedDict((monitor.name, bool(self.state[i])) for i, monitor in enumerate(self.monitors))

    def register_expression_updater(self, expression: cas.PreservedCasType,
                                    monitors: Tuple[Union[str, ExpressionMonitor], ...]) \
            -> cas.PreservedCasType:
        """
        Expression is updated when all monitors are 1 at the same time, but only once.
        """
        monitor_names = monitor_list_to_monitor_name_tuple(monitors)
        old_symbols = []
        new_symbols = []
        for i, symbol in enumerate(expression.free_symbols()):
            old_symbols.append(symbol)
            new_symbols.append(self.get_substitution_key(monitor_names, str(symbol)))
        new_expression = cas.substitute(expression, old_symbols, new_symbols)
        self.update_substitution_values(monitor_names, [str(s) for s in old_symbols])
        return new_expression

    def to_state_filter(self, monitor_names: List[str]) -> np.ndarray:
        monitor_names = monitor_list_to_monitor_name_tuple(monitor_names)
        return np.array([monitor.id for monitor in self.monitors if monitor.name in monitor_names])

    def _register_expression_update_triggers(self):
        for monitor_names, values in self.substitution_values.items():
            trigger_filter = tuple([i for i, m in enumerate(self.expression_monitors) if m.name in monitor_names])
            self.triggers[trigger_filter] = lambda: self.update_substitution_values(monitor_names, list(values.keys()))

    @profile
    def update_substitution_values(self, monitor_names: Tuple[str, ...], keys: Optional[List[str]] = None):
        if keys is None:
            keys = list(self.substitution_values[monitor_names].keys())
        values = symbol_manager.resolve_symbols(keys)
        self.substitution_values[monitor_names] = {key: value for key, value in zip(keys, values)}

    @profile
    def get_substitution_key(self, monitor_names: Tuple[str, ...], original_expr: str) -> cas.Symbol:
        return symbol_manager.get_symbol(
            f'god_map.monitor_manager.substitution_values[{monitor_names}]["{original_expr}"]')

    @profile
    def trigger_update_triggers(self, state: np.ndarray):
        non_zeros = state.nonzero()[0]
        for trigger_map, trigger_function in list(self.triggers.items()):
            if np.all(np.isin(trigger_map, non_zeros)):
                trigger_function()
                del self.triggers[trigger_map]

    @profile
    def evaluate_monitors(self):
        args = symbol_manager.resolve_symbols(self.compiled_monitors.str_params)
        next_state = np.zeros_like(self.state)
        num_expr_monitors = len(self.expression_monitors)
        expr_monitor_state = self.compiled_monitors.fast_call(args)
        expr_monitor_state = self.update_expr_monitor_state(expr_monitor_state)
        next_state[:num_expr_monitors] = expr_monitor_state
        for i in range(num_expr_monitors, len(self.monitors)):
            next_state[i] = self.payload_monitors[i-num_expr_monitors].get_state()
        self.state = next_state
        self.trigger_update_triggers(self.state)

    def search_for_monitors(self, monitor_names: List[str]) -> List[Monitor]:
        return [self.get_monitor(monitor_name) for monitor_name in monitor_names]

    @profile
    def parse_monitors(self, monitor_msgs: List[giskard_msgs.Monitor]):
        for monitor_msg in monitor_msgs:
            try:
                logging.loginfo(f'Adding monitor of type: \'{monitor_msg.monitor_class}\'')
                C = self.allowed_monitor_types[monitor_msg.monitor_class]
            except KeyError:
                raise UnknownConstraintException(f'unknown monitor type: \'{monitor_msg.monitor_class}\'.')
            try:
                kwargs = json_str_to_kwargs(monitor_msg.kwargs)
                start_monitors = god_map.monitor_manager.search_for_monitors(monitor_msg.start_monitors)
                monitor = C(name=monitor_msg.name,
                            start_monitors=start_monitors,
                            **kwargs)
                if isinstance(monitor, ExpressionMonitor):
                    self.add_expression_monitor(monitor)
                elif isinstance(monitor, PayloadMonitor):
                    self.add_payload_monitor(monitor)
            except Exception as e:
                traceback.print_exc()
                error_msg = f'Initialization of \'{C.__name__}\' monitor failed: \n {e} \n'
                if not isinstance(e, GiskardException):
                    raise ConstraintInitalizationException(error_msg)
                raise e
