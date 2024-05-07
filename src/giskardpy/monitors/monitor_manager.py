import ast
import traceback
from collections import OrderedDict
from functools import cached_property
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
    state: np.ndarray  # order: ExpressionMonitors, PayloadMonitors
    life_cycle_state: np.ndarray  # order: ExpressionMonitors, PayloadMonitors
    compiled_life_cycle_state_updater: CompiledFunction
    state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)

    monitors: List[Monitor]

    substitution_values: Dict[int, Dict[str, float]]  # id -> (old_symbol, value)
    triggers: Dict[int, Callable]  # id -> updater callback
    trigger_conditions: List[cas.Expression]  # id -> condition
    compiled_trigger_conditions: cas.CompiledFunction  # stacked compiled function which returns array of evaluated conditions

    def __init__(self):
        self.monitors = []
        self.allowed_monitor_types = {}
        for path in god_map.giskard.monitor_package_paths:
            self.allowed_monitor_types.update(get_all_classes_in_package(path, Monitor))
        self.state_history = []
        self.substitution_values = {}
        self.triggers = {}
        self.trigger_conditions = []

    def evaluate_expression(self, node):
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return cas.logic_and(*[self.evaluate_expression(x) for x in node.values])
            elif isinstance(node.op, ast.Or):
                return cas.logic_or(*[self.evaluate_expression(x) for x in node.values])
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return cas.logic_not(self.evaluate_expression(node.operand))
        elif isinstance(node, ast.Str):
            # replace monitor name with its state expression
            return self.get_monitor(node.value).get_state_expression()
        raise Exception(f'failed to parse {node}')

    def logic_str_to_expr(self, logic_str: str, default: cas.Expression) -> cas.Expression:
        if logic_str == '':
            return default
        tree = ast.parse(logic_str, mode='eval')
        try:
            return self.evaluate_expression(tree.body)
        except KeyError as e:
            raise GiskardException(f'Unknown symbol {e}')
        except Exception as e:
            raise GiskardException(str(e))

    @profile
    def add_payload_monitors_to_behavior_tree(self, traj_tracking: bool = False) -> None:
        payload_monitors = sorted(self.payload_monitors, key=lambda x: isinstance(x, CancelMotion))
        for monitor in payload_monitors:
            if traj_tracking:
                god_map.tree.execute_traj.base_closed_loop.check_monitors.add_monitor(monitor)
            else:
                god_map.tree.control_loop_branch.check_monitors.add_monitor(monitor)

    @profile
    def compile_monitors(self, traj_tracking: bool = False) -> None:
        self.state_history = []
        self.state = np.zeros(len(self.monitors))
        self.life_cycle_state = np.zeros(len(self.monitors))
        self.set_initial_life_cycle_state()
        self.compile_monitor_state_updater()
        self.add_payload_monitors_to_behavior_tree(traj_tracking=traj_tracking)
        self._register_expression_update_triggers()

    @profile
    def get_monitor(self, name: str) -> Monitor:
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

    def get_monitor_from_state_expr(self, expr: cas.Expression) -> Monitor:
        for monitor in self.monitors:
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
    def compile_monitor_state_updater(self):
        monitor_state = cas.Expression(self.get_state_expression_symbols())

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
                                    if_result=monitor.expression,
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
    def expression_monitors(self) -> List[ExpressionMonitor]:
        return [x for x in self.monitors if isinstance(x, ExpressionMonitor)]

    @property
    def payload_monitors(self) -> List[PayloadMonitor]:
        return [x for x in self.monitors if isinstance(x, PayloadMonitor)]

    @profile
    def add_expression_monitor(self, monitor: ExpressionMonitor):
        if [x for x in self.monitors if x.name == monitor.name]:
            raise MonitorInitalizationException(f'Monitor named {monitor.name} already exists.')
        self.monitors.append(monitor)
        monitor.set_id(len(self.monitors) - 1)

    @profile
    def add_payload_monitor(self, monitor: PayloadMonitor):
        if [x for x in self.monitors if x.name == monitor.name]:
            raise MonitorInitalizationException(f'Monitor named {monitor.name} already exists.')
        self.monitors.append(monitor)
        monitor.set_id(len(self.monitors) - 1)

    def get_state_dict(self) -> Dict[str, Tuple[str, bool]]:
        return OrderedDict((monitor.name, (str(TaskState(self.life_cycle_state[i])), bool(self.state[i])))
                           for i, monitor in enumerate(self.monitors))

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
        return np.array([monitor.id for monitor in self.monitors if monitor.name in monitor_names])

    def get_state_expression_symbols(self) -> List[cas.Symbol]:
        return [m.get_state_expression() for m in self.monitors]

    @profile
    def _register_expression_update_triggers(self):
        for updater_id, values in self.substitution_values.items():
            class Callback:
                def __init__(self, updater_id: int, values):
                    self.updater_id = updater_id
                    self.keys = list(values.keys())

                def __call__(self):
                    return god_map.monitor_manager.update_substitution_values(self.updater_id, self.keys)

            self.triggers[updater_id] = Callback(updater_id, values)
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
            f'god_map.monitor_manager.substitution_values[{updater_id}]["{original_expr}"]')

    @profile
    def trigger_update_triggers(self, state: np.ndarray):
        condition_state = self.compiled_trigger_conditions.fast_call(state)
        for updater_id, value in enumerate(condition_state):
            if updater_id in self.triggers and value:
                self.triggers[updater_id]()
                del self.triggers[updater_id]

    def evaluate_trigger_conditions(self, state: np.ndarray) -> None:
        pass

    @cached_property
    def payload_monitor_filter(self):
        return np.array([i for i, m in enumerate(self.monitors) if isinstance(m, PayloadMonitor)])

    @profile
    def evaluate_monitors(self):
        # %% update monitor state
        args = symbol_manager.resolve_symbols(self.compiled_state_updater.str_params)
        self.state = self.compiled_state_updater.fast_call(args)

        # %% update life cycle state
        args = np.concatenate((self.state, self.life_cycle_state))
        self.life_cycle_state = self.compiled_life_cycle_state_updater.fast_call(args)

        if len(self.payload_monitor_filter) > 0:
            self.state[self.payload_monitor_filter] = self.evaluate_payload_monitors()
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
                start_condition = self.logic_str_to_expr(monitor_msg.start_condition, default=cas.TrueSymbol)
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
