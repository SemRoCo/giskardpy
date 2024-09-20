import ast
from collections import OrderedDict
from functools import cached_property
from typing import List, Tuple, Dict, Optional, Callable, Union, Iterable
from line_profiler import profile

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types.data_types import LifeCycleState, PrefixName
from giskardpy.data_types.exceptions import GiskardException, MonitorInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_graph.helpers import compile_graph_node_state_updater
from giskardpy.motion_graph.monitors.monitors import ExpressionMonitor, Monitor, EndMotion
from giskardpy.motion_graph.monitors.payload_monitors import PayloadMonitor, CancelMotion
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.utils import get_all_classes_in_package


def monitor_list_to_monitor_name_tuple(monitors: Iterable[Union[str, ExpressionMonitor]]) -> Tuple[str, ...]:
    return tuple(sorted(monitor.name if isinstance(monitor, Monitor) else monitor for monitor in monitors))


class MonitorManager:
    compiled_state_updater: CompiledFunction
    state: np.ndarray  # order: ExpressionMonitors, PayloadMonitors
    life_cycle_state: np.ndarray  # order: ExpressionMonitors, PayloadMonitors
    compiled_life_cycle_state_updater: CompiledFunction
    state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)

    monitors: Dict[PrefixName, Monitor]

    substitution_values: Dict[int, Dict[str, float]]  # id -> (old_symbol, value)
    triggers: Dict[int, Callable]  # id -> updater callback
    trigger_conditions: List[cas.Expression]  # id -> condition
    compiled_trigger_conditions: cas.CompiledFunction  # stacked compiled function which returns array of evaluated conditions

    def __init__(self):
        self.allowed_monitor_types = {}
        self.add_monitor_package_path('giskardpy.motion_graph.monitors')
        self.reset()

    def add_monitor_package_path(self, path: str) -> None:
        self.allowed_monitor_types.update(get_all_classes_in_package(path, Monitor))

    def reset(self):
        try:
            del self.payload_monitor_filter
        except Exception as e:
            pass
        self.monitors = OrderedDict()
        self.state_history = []
        self.substitution_values = {}
        self.triggers = {}
        self.trigger_conditions = []

    def evaluate_expression(self, node,
                            monitor_name_to_state_expr: Dict[str, cas.Expression]) -> cas.Expression:
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return cas.logic_and(*[self.evaluate_expression(x, monitor_name_to_state_expr) for x in node.values])
            elif isinstance(node.op, ast.Or):
                return cas.logic_or(*[self.evaluate_expression(x, monitor_name_to_state_expr) for x in node.values])
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return cas.logic_not(self.evaluate_expression(node.operand, monitor_name_to_state_expr))
        elif isinstance(node, ast.Str):
            # replace monitor name with its state expression
            return monitor_name_to_state_expr[node.value]
        raise Exception(f'failed to parse {node}')

    def logic_str_to_expr(self, logic_str: str, default: cas.Expression,
                          monitor_name_to_state_expr: Optional[Dict[str, cas.Expression]] = None) -> cas.Expression:
        if monitor_name_to_state_expr is None:
            monitor_name_to_state_expr = {key: value.get_state_expression() for key, value in self.monitors.items()}
        if logic_str == '':
            return default
        tree = ast.parse(logic_str, mode='eval')
        try:
            return self.evaluate_expression(tree.body, monitor_name_to_state_expr)
        except KeyError as e:
            raise GiskardException(f'Unknown symbol {e}')
        except Exception as e:
            raise GiskardException(str(e))

    @profile
    def compile_monitors(self) -> None:
        self.state = np.zeros(len(self.monitors))
        self.life_cycle_state = np.zeros(len(self.monitors))
        self.set_initial_life_cycle_state()
        self.compile_monitor_state_updater()
        self._register_expression_update_triggers()

    @profile
    def get_monitor(self, name: str) -> Monitor:
        for monitor in self.monitors.values():
            if monitor.name == name:
                return monitor
        raise KeyError(f'No monitor of name \'{name}\' found.')

    def get_monitor_state_expr(self) -> cas.Expression:
        symbols = []
        for monitor in self.monitors.values():
            symbols.append(monitor.get_state_expression())
        return cas.Expression(symbols)

    def set_initial_life_cycle_state(self):
        for monitor in self.monitors.values():
            if cas.is_true(monitor.start_condition):
                self.life_cycle_state[monitor.id] = LifeCycleState.running
            else:
                self.life_cycle_state[monitor.id] = LifeCycleState.not_started

    def get_monitor_from_state_expr(self, expr: cas.Expression) -> Monitor:
        for monitor in self.monitors.values():
            if cas.is_true(monitor.get_state_expression() == expr):
                return monitor
        raise GiskardException('No monitor found.')

    def is_monitor_registered(self, monitor_state_expr: cas.Expression) -> bool:
        try:
            self.get_monitor_from_state_expr(monitor_state_expr)
            return True
        except GiskardException as e:
            return False

    def format_condition(self, condition: cas.Expression, new_line: str = '\n') -> str:
        """
        Takes a logical expression, replaces the state symbols with monitor names and formats it nicely.
        """
        free_symbols = condition.free_symbols()
        if not free_symbols:
            return str(cas.is_true(condition))
        condition = str(condition)
        state_to_monitor_map = {str(x): f'\'{self.get_monitor_from_state_expr(x).name}\'' for x in free_symbols}
        state_to_monitor_map['&&'] = f'{new_line}and '
        state_to_monitor_map['||'] = f'{new_line}or '
        state_to_monitor_map['!'] = 'not '
        for state_str, monitor_name in state_to_monitor_map.items():
            condition = condition.replace(state_str, monitor_name)
        return condition

    @profile
    def compile_monitor_state_updater(self) -> None:
        state_updater = []
        for monitor in self.monitors.values():
            state_symbol = monitor.get_state_expression()

            if isinstance(monitor, ExpressionMonitor):
                monitor.pre_compile()
                state_f = cas.if_eq(monitor.get_life_cycle_state_expression(), int(LifeCycleState.running),
                                    if_result=monitor.expression,
                                    else_result=state_symbol)
            else:
                state_f = state_symbol  # if payload monitor, copy last state
            state_updater.append(state_f)

        state_updater = cas.Expression(state_updater)
        self.compiled_state_updater = state_updater.compile(state_updater.free_symbols())

        self.compiled_life_cycle_state_updater = compile_graph_node_state_updater(self.monitors)

    @property
    def expression_monitors(self) -> List[ExpressionMonitor]:
        return [x for x in self.monitors if isinstance(x, ExpressionMonitor)]

    @property
    def payload_monitors(self) -> List[PayloadMonitor]:
        return [x for x in self.monitors.values() if isinstance(x, PayloadMonitor)]

    def add_monitor(self, monitor: Monitor) -> None:
        if [x for x in self.monitors.values() if x.name == monitor.name]:
            raise MonitorInitalizationException(f'Monitor named {monitor.name} already exists.')
        self.monitors[monitor.name] = monitor
        monitor.id = len(self.monitors) - 1

    def get_state_dict(self) -> Dict[str, Tuple[str, bool]]:
        return OrderedDict((monitor.name, (str(LifeCycleState(self.life_cycle_state[i])), bool(self.state[i])))
                           for i, monitor in enumerate(self.monitors.values()))

    @profile
    def register_expression_updater(self, expression: cas.PreservedCasType,
                                    condition: cas.Expression) \
            -> cas.PreservedCasType:
        """
        Expression is updated when all monitors are 1 at the same time, but only once.
        """
        updater_id = len(self.substitution_values)
        if cas.is_true(condition):
            raise ValueError('condition is always true')
        old_symbols = []
        new_symbols = []
        for i, symbol in enumerate(expression.free_symbols()):
            old_symbols.append(symbol)
            new_symbols.append(self.get_substitution_key(updater_id, str(symbol)))
        new_expression = cas.substitute(expression, old_symbols, new_symbols)
        self.update_substitution_values(updater_id, [str(s) for s in old_symbols])
        self.trigger_conditions.append(condition)
        return new_expression

    @profile
    def to_state_filter(self, monitor_names: List[Union[str, Monitor]]) -> np.ndarray:
        monitor_names = monitor_list_to_monitor_name_tuple(monitor_names)
        return np.array([monitor.id for monitor in self.monitors.values() if monitor.name in monitor_names])

    def get_state_expression_symbols(self) -> List[cas.Symbol]:
        return [m.get_state_expression() for m in self.monitors.values()]

    @profile
    def _register_expression_update_triggers(self):
        for updater_id, values in self.substitution_values.items():
            class Callback:
                def __init__(self, updater_id: int, values, motion_graph_manager: MonitorManager):
                    self.updater_id = updater_id
                    self.keys = list(values.keys())
                    self.motion_graph_manager = motion_graph_manager

                def __call__(self):
                    return self.motion_graph_manager.update_substitution_values(self.updater_id, self.keys)

            self.triggers[updater_id] = Callback(updater_id, values, self)
        expr = cas.Expression(self.trigger_conditions)
        self.compiled_trigger_conditions = expr.compile(self.get_state_expression_symbols())

    @profile
    def update_substitution_values(self, updater_id: int, keys: Optional[List[str]] = None):
        if keys is None:
            keys = list(self.substitution_values[updater_id].keys())
        values = symbol_manager.resolve_symbols(keys)
        self.substitution_values[updater_id] = {key: value for key, value in zip(keys, values)}

    @profile
    def get_substitution_key(self, updater_id: int, original_expr: str) -> cas.Symbol:
        return symbol_manager.get_symbol(
            f'god_map.motion_graph_manager.substitution_values[{updater_id}]["{original_expr}"]')

    @profile
    def trigger_update_triggers(self, state: np.ndarray):
        condition_state = self.compiled_trigger_conditions.fast_call(state)
        for updater_id, value in enumerate(condition_state):
            if updater_id in self.triggers and value:
                self.triggers[updater_id]()
                del self.triggers[updater_id]

    @cached_property
    def payload_monitor_filter(self):
        return np.array([i for i, m in enumerate(self.monitors.values()) if isinstance(m, PayloadMonitor)])

    @profile
    def evaluate_monitors(self):
        # %% update monitor state
        args = symbol_manager.resolve_symbols(self.compiled_state_updater.str_params)
        self.state = self.compiled_state_updater.fast_call(args)

        # %% update life cycle state
        args = np.concatenate((self.life_cycle_state, self.state))
        self.life_cycle_state = self.compiled_life_cycle_state_updater.fast_call(args)

        if len(self.payload_monitor_filter) > 0:
            self.state[self.payload_monitor_filter] = self.evaluate_payload_monitors()
        self.trigger_update_triggers(self.state)
        self.state_history.append((god_map.time, (self.state.copy(), self.life_cycle_state.copy())))
        god_map.motion_graph_manager.update_task_state(self.state)

    def evaluate_payload_monitors(self) -> np.ndarray:
        next_state = np.zeros(len(self.payload_monitors))
        for i in range(len(self.payload_monitors)):
            next_state[i] = self.payload_monitors[i].get_state()
        return next_state

    @profile
    def search_for_monitors(self, monitor_names: List[str]) -> List[Monitor]:
        return [self.get_monitor(monitor_name) for monitor_name in monitor_names]

    def has_end_motion_monitor(self) -> bool:
        for m in self.monitors.values():
            if isinstance(m, EndMotion):
                return True
        return False

    def has_cancel_motion_monitor(self) -> bool:
        for m in self.monitors.values():
            if isinstance(m, CancelMotion):
                return True
        return False

    def has_payload_monitors_which_are_not_end_nor_cancel(self) -> bool:
        for m in self.monitors.values():
            if not isinstance(m, (CancelMotion, EndMotion)) and isinstance(m, PayloadMonitor):
                return True
        return False
