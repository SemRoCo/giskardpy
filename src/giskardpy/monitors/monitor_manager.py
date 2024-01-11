import ast
import traceback
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Callable, Union, Iterable

import numpy as np

import giskard_msgs.msg as giskard_msgs
import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types import TaskState
from giskardpy.exceptions import GiskardException, MonitorInitalizationException, UnknownMonitorException
from giskardpy.god_map import god_map
from giskardpy.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.monitors.payload_monitors import PayloadMonitor, CancelMotion
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


def monitor_list_to_monitor_name_tuple(monitors: Iterable[Union[str, ExpressionMonitor]]) -> Tuple[str, ...]:
    return tuple(sorted(monitor.name if isinstance(monitor, Monitor) else monitor for monitor in monitors))


class MonitorManager:
    compiled_state_updater: CompiledFunction
    compiled_life_cycle_state_updater: CompiledFunction
    state: np.ndarray  # order: ExpressionMonitors, PayloadMonitors
    life_cycle_state: np.ndarray  # order: ExpressionMonitors, PayloadMonitors
    expression_monitors: List[ExpressionMonitor]
    payload_monitors: List[PayloadMonitor]
    substitution_values: Dict[Tuple[str, ...], Dict[str, float]]
    triggers: Dict[Tuple[int, ...], Callable]
    state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)

    def __init__(self):
        self.expression_monitors = []
        self.payload_monitors = []
        self.allowed_monitor_types = {}
        self.allowed_monitor_types.update(get_all_classes_in_package(package_name='giskardpy.monitors',
                                                                     parent_class=Monitor))
        self.substitution_values = {}
        self.triggers = {}
        self.state_history = []

    def evaluate_expression(self, node):
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return cas.logic_and(self.evaluate_expression(node.values[0]),
                                     self.evaluate_expression(node.values[1]))
            elif isinstance(node.op, ast.Or):
                return cas.logic_or(self.evaluate_expression(node.values[0]),
                                    self.evaluate_expression(node.values[1]))
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return cas.logic_not(self.evaluate_expression(node.operand))
        elif isinstance(node, ast.Str):
            # replace monitor name with its state expression
            return self.get_monitor(node.value).get_state_expression()
        raise Exception(f'failed to parse {node}')

    def logic_str_to_expr(self, logic_str: str) -> cas.Expression:
        if logic_str == '':
            return cas.TrueSymbol
        tree = ast.parse(logic_str, mode='eval')
        try:
            return self.evaluate_expression(tree.body)
        except KeyError as e:
            raise GiskardException(f'Unknown symbol {e}')
        except Exception as e:
            raise GiskardException(str(e))

    @profile
    def add_payload_monitors_to_behavior_tree(self):
        self.payload_monitors = sorted(self.payload_monitors, key=lambda x: isinstance(x, CancelMotion))
        for monitor in self.payload_monitors:
            god_map.tree.control_loop_branch.check_monitors.add_monitor(monitor)

    @profile
    def compile_monitors(self):
        self.state_history = []
        self.state = np.zeros(len(self.monitors))
        self.life_cycle_state = np.zeros(len(self.monitors))
        self.set_initial_life_cycle_state()
        self.compile_monitor_state_updater()
        self.add_payload_monitors_to_behavior_tree()
        self._register_expression_update_triggers()

    @profile
    def get_monitor(self, name: str) -> ExpressionMonitor:
        for monitor in self.monitors:
            if monitor.name == name:
                return monitor
        raise KeyError(f'No monitor of name \'{name}\' found.')

    def set_initial_life_cycle_state(self):
        for monitor in self.monitors:
            if cas.is_true(monitor.start_condition):
                self.life_cycle_state[monitor.id] = TaskState.running
            else:
                self.life_cycle_state[monitor.id] = TaskState.not_started

    @profile
    def compile_monitor_state_updater(self):
        symbols = []
        for i in range(len(god_map.monitor_manager.monitors)):
            symbols.append(self.monitors[i].get_state_expression())
        monitor_state = cas.Expression(symbols)

        symbols = []
        for i in range(len(god_map.monitor_manager.monitors)):
            symbols.append(self.monitors[i].get_life_cycle_state_expression())
        monitor_life_cycle_state = cas.Expression(symbols)

        state_updater = []
        life_cycle_state_updater = []
        for i, monitor in enumerate(self.monitors):
            state_symbol = monitor_state[monitor.id]
            life_cycle_state_symbol = monitor_life_cycle_state[monitor.id]

            if not cas.is_true(monitor.start_condition):
                start_if = cas.if_else(monitor.start_condition,
                                       if_result=int(TaskState.running),
                                       else_result=life_cycle_state_symbol)
            else:
                start_if = life_cycle_state_symbol

            done_if = cas.if_else(condition=cas.logic_and(cas.Expression(monitor.stay_true), state_symbol),
                                  if_result=int(TaskState.succeeded),
                                  else_result=life_cycle_state_symbol)

            life_cycle_state_f = cas.if_eq_cases(a=state_symbol,
                                                 b_result_cases=[(int(TaskState.not_started), start_if),
                                                                 (int(TaskState.running), done_if)],
                                                 else_result=life_cycle_state_symbol)
            life_cycle_state_updater.append(life_cycle_state_f)

            if isinstance(monitor, ExpressionMonitor):
                monitor.compile()
                state_f = cas.if_eq(life_cycle_state_symbol, int(TaskState.running),
                                    if_result=monitor.get_expression(),
                                    else_result=state_symbol)
            else:
                state_f = state_symbol  # if payload monitor, copy last state
            state_updater.append(state_f)

        symbols = monitor_state.free_symbols() + monitor_life_cycle_state.free_symbols()
        life_cycle_state_updater = cas.Expression(life_cycle_state_updater)
        state_updater = cas.Expression(state_updater)
        self.compiled_state_updater = state_updater.compile(state_updater.free_symbols())
        self.compiled_life_cycle_state_updater = life_cycle_state_updater.compile(symbols)

    @property
    def monitors(self):
        return self.expression_monitors + self.payload_monitors

    @profile
    def add_expression_monitor(self, monitor: ExpressionMonitor):
        if [x for x in self.monitors if x.name == monitor.name]:
            raise MonitorInitalizationException(f'Monitor named {monitor.name} already exists.')
        self.expression_monitors.append(monitor)
        monitor.set_id(len(self.expression_monitors) - 1)
        # increase all payload monitor ids because expression monitors always come first
        for payload_monitor in self.payload_monitors:
            payload_monitor.set_id(payload_monitor.id + 1)

    @profile
    def add_payload_monitor(self, monitor: PayloadMonitor):
        if [x for x in self.monitors if x.name == monitor.name]:
            raise MonitorInitalizationException(f'Monitor named {monitor.name} already exists.')
        self.payload_monitors.append(monitor)
        monitor.set_id(len(self.monitors) - 1)

    def get_state_dict(self) -> Dict[str, Tuple[str, bool]]:
        return OrderedDict((monitor.name, (str(TaskState(self.life_cycle_state[i])), bool(self.state[i])))
                           for i, monitor in enumerate(self.monitors))

    @profile
    def register_expression_updater(self, expression: cas.PreservedCasType,
                                    monitors: Tuple[Union[str, ExpressionMonitor], ...]) \
            -> cas.PreservedCasType:
        """
        Expression is updated when all monitors are 1 at the same time, but only once.
        """
        if not monitors:
            raise ValueError('monitors is empty.')
        monitor_names = monitor_list_to_monitor_name_tuple(monitors)
        old_symbols = []
        new_symbols = []
        for i, symbol in enumerate(expression.free_symbols()):
            old_symbols.append(symbol)
            new_symbols.append(self.get_substitution_key(monitor_names, str(symbol)))
        new_expression = cas.substitute(expression, old_symbols, new_symbols)
        self.update_substitution_values(monitor_names, [str(s) for s in old_symbols])
        return new_expression

    @profile
    def to_state_filter(self, monitor_names: List[Union[str, Monitor]]) -> np.ndarray:
        monitor_names = monitor_list_to_monitor_name_tuple(monitor_names)
        return np.array([monitor.id for monitor in self.monitors if monitor.name in monitor_names])

    @profile
    def _register_expression_update_triggers(self):
        for monitor_names, values in self.substitution_values.items():
            trigger_filter = tuple([i for i, m in enumerate(self.expression_monitors) if m.name in monitor_names])

            class Callback:
                def __init__(self, monitor_names, values):
                    self.monitor_names = monitor_names
                    self.keys = list(values.keys())

                def __call__(self):
                    return god_map.monitor_manager.update_substitution_values(self.monitor_names, self.keys)

            self.triggers[trigger_filter] = Callback(monitor_names, values)

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
        # %% update life cycle state
        args = np.concatenate((self.state, self.life_cycle_state))
        self.life_cycle_state = self.compiled_life_cycle_state_updater.fast_call(args)

        # %% update monitor state
        args = symbol_manager.resolve_symbols(self.compiled_state_updater.str_params)
        self.state = self.compiled_state_updater.fast_call(args)

        num_expr_monitors = len(self.expression_monitors)
        self.state[num_expr_monitors:] = self.evaluate_payload_monitors()
        self.trigger_update_triggers(self.state)
        self.state_history.append((god_map.time, (self.state.copy(), self.life_cycle_state.copy())))
        god_map.motion_goal_manager.update_task_state(self.state)

    def evaluate_payload_monitors(self) -> np.ndarray:
        next_state = np.zeros(len(self.payload_monitors))
        for i in range(len(self.payload_monitors)):
            next_state[i] = self.payload_monitors[i].get_state()
        return next_state

    @profile
    def search_for_monitors(self, monitor_names: List[str]) -> List[Monitor]:
        return [self.get_monitor(monitor_name) for monitor_name in monitor_names]

    @profile
    def parse_monitors(self, monitor_msgs: List[giskard_msgs.Monitor]):
        for monitor_msg in monitor_msgs:
            try:
                logging.loginfo(f'Adding monitor of type: \'{monitor_msg.monitor_class}\'')
                C = self.allowed_monitor_types[monitor_msg.monitor_class]
            except KeyError:
                raise UnknownMonitorException(f'unknown monitor type: \'{monitor_msg.monitor_class}\'.')
            try:
                kwargs = json_str_to_kwargs(monitor_msg.kwargs)
                start_condition = self.logic_str_to_expr(monitor_msg.start_condition)
                monitor = C(name=monitor_msg.name,
                            start_condition=start_condition,
                            **kwargs)
                if isinstance(monitor, ExpressionMonitor):
                    self.add_expression_monitor(monitor)
                elif isinstance(monitor, PayloadMonitor):
                    self.add_payload_monitor(monitor)
            except Exception as e:
                traceback.print_exc()
                error_msg = f'Initialization of \'{C.__name__}\' monitor failed: \n {e} \n'
                if not isinstance(e, GiskardException):
                    raise MonitorInitalizationException(error_msg)
                raise e
