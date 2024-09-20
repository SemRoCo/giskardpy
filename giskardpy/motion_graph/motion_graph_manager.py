import ast
import itertools
from collections import OrderedDict
from functools import cached_property
from typing import List, Tuple, Dict, Optional, Callable, Union, Iterable, Type
from line_profiler import profile

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types.data_types import LifeCycleState, PrefixName
from giskardpy.data_types.exceptions import GiskardException, MonitorInitalizationException, UnknownMonitorException, \
    UnknownTaskException, TaskInitalizationException, GoalInitalizationException
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.motion_graph.graph_node import MotionGraphNode
from giskardpy.motion_graph.helpers import compile_graph_node_state_updater
from giskardpy.motion_graph.monitors.monitors import ExpressionMonitor, Monitor, EndMotion
from giskardpy.motion_graph.monitors.payload_monitors import PayloadMonitor, CancelMotion
from giskardpy.motion_graph.tasks.task import Task
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.utils import get_all_classes_in_package, ImmutableDict


def monitor_list_to_monitor_name_tuple(monitors: Iterable[Union[str, ExpressionMonitor]]) -> Tuple[str, ...]:
    return tuple(sorted(monitor.name if isinstance(monitor, Monitor) else monitor for monitor in monitors))


class MotionGraphManager:
    tasks: Dict[PrefixName, Task]
    monitors: Dict[PrefixName, Monitor]

    task_observation_state: np.ndarray  # order: ExpressionMonitors, PayloadMonitors
    compiled_task_observation_state: CompiledFunction

    monitor_observation_state: np.ndarray  # order: Tasks, ExpressionMonitors, PayloadMonitors
    compiled_monitor_observation_state: CompiledFunction

    task_life_cycle_state: np.ndarray
    compiled_task_life_cycle_state: CompiledFunction
    monitor_life_cycle_state: np.ndarray  # order: ExpressionMonitors, PayloadMonitors
    compiled_monitor_life_cycle_state_updater: CompiledFunction

    task_state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)
    monitor_state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)

    substitution_values: Dict[int, Dict[str, float]]  # id -> (old_symbol, value)
    triggers: Dict[int, Callable]  # id -> updater callback
    trigger_conditions: List[cas.Expression]  # id -> condition
    compiled_trigger_conditions: cas.CompiledFunction  # stacked compiled function which returns array of evaluated conditions

    def __init__(self):
        self.allowed_monitor_types = {}
        self.allowed_task_types = {}
        self.allowed_goal_types = {}
        self.add_monitor_package_path('giskardpy.motion_graph.monitors')
        self.add_task_package_path('giskardpy.motion_graph.tasks')
        self.add_goal_package_path('giskardpy.goals')
        self.reset()

    def add_monitor_package_path(self, path: str) -> None:
        self.allowed_monitor_types.update(get_all_classes_in_package(path, Monitor))

    def add_task_package_path(self, path: str) -> None:
        self.allowed_task_types.update(get_all_classes_in_package(path, Task))

    def add_goal_package_path(self, path: str) -> None:
        self.allowed_goal_types.update(get_all_classes_in_package(path, Goal))

    def reset(self):
        try:
            del self.payload_monitor_filter
        except Exception as e:
            pass
        self.monitors = OrderedDict()
        self.tasks = OrderedDict()
        self.task_state_history = []
        self.monitor_state_history = []
        self.substitution_values = {}
        self.triggers = {}
        self.trigger_conditions = []

    def add_monitor(self, monitor: Monitor) -> None:
        if [x for x in self.monitors.values() if x.name == monitor.name]:
            raise MonitorInitalizationException(f'Monitor named {monitor.name} already exists.')
        self.monitors[monitor.name] = monitor
        monitor.id = len(self.monitors) - 1

    def add_task(self, task: Task) -> None:
        if [x for x in self.tasks.values() if x.name == task.name]:
            raise TaskInitalizationException(f'Task named {task.name} already exists.')
        self.tasks[task.name] = task
        task.id = len(self.tasks) - 1

    # def add_motion_goal(self, motion_goal: Goal) -> None:
    #     if [x for x in self.monitors.values() if x.name == motion_goal.name]:
    #         raise MonitorInitalizationException(f'Monitor named {motion_goal.name} already exists.')
    #     self.monitors[motion_goal.name] = motion_goal
    #     motion_goal.id = len(self.monitors) - 1

    def parse_motion_graph(self, *,
                           tasks: List[Tuple[str, str, str, str, str, str, dict]],
                           monitors: List[Tuple[str, str, str, str, str, str, dict]],
                           goals: List[Tuple[str, str, str, str, str, str, dict]]) -> None:
        task_states = {}
        for task_id, (_, name, _, _, _, _, _) in enumerate(tasks):
            path = f'god_map.motion_graph_manager.task_observation_state[{task_id}]'
            symbol = symbol_manager.get_symbol(path)
            task_states[name] = symbol
        monitor_states = {}
        for monitor_id, (_, name, _, _, _, _, _) in enumerate(monitors):
            path = f'god_map.motion_graph_manager.monitor_observation_state[{monitor_id}]'
            symbol = symbol_manager.get_symbol(path)
            monitor_states[name] = symbol
        observation_state_symbols = {**task_states, **monitor_states}
        # %% add tasks
        for class_name, name, start, reset, pause, end, kwargs in tasks:
            try:
                get_middleware().loginfo(f'Adding task of type: \'{class_name}\'')
                C = god_map.motion_graph_manager.allowed_task_types[class_name]
            except KeyError:
                raise UnknownTaskException(f'unknown task type: \'{class_name}\'.')
            start_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=start,
                default=cas.TrueSymbol,
                observation_state_symbols=observation_state_symbols)
            reset_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=reset,
                default=cas.FalseSymbol,
                observation_state_symbols=observation_state_symbols)
            pause_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=pause,
                default=cas.FalseSymbol,
                observation_state_symbols=observation_state_symbols)
            end_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=end,
                default=cas.FalseSymbol,
                observation_state_symbols=observation_state_symbols)
            task: Task = C(name=name,
                           start_condition=start_condition,
                           # reset_condition=reset_condition,
                           pause_condition=pause_condition,
                           end_condition=end_condition,
                           **kwargs)
            self.add_task(task)
        # %% add monitors
        for class_name, name, start, reset, pause, end, kwargs in monitors:
            try:
                get_middleware().loginfo(f'Adding monitor of type: \'{class_name}\'')
                C = god_map.motion_graph_manager.allowed_monitor_types[class_name]
            except KeyError:
                raise UnknownMonitorException(f'unknown monitor type: \'{class_name}\'.')
            start_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=start,
                default=cas.TrueSymbol,
                observation_state_symbols=observation_state_symbols)
            reset_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=reset,
                default=cas.FalseSymbol,
                observation_state_symbols=observation_state_symbols)
            pause_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=pause,
                default=cas.FalseSymbol,
                observation_state_symbols=observation_state_symbols)
            end_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=end,
                default=cas.FalseSymbol,
                observation_state_symbols=observation_state_symbols)
            monitor: Monitor = C(name=name,
                                 start_condition=start_condition,
                                 # reset_condition=reset_condition,
                                 pause_condition=pause_condition,
                                 end_condition=end_condition,
                                 **kwargs)
            self.add_monitor(monitor)

    def evaluate_expression(self, node,
                            observation_state_symbols: Dict[str, cas.Expression]) -> cas.Expression:
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return cas.logic_and(*[self.evaluate_expression(x, observation_state_symbols) for x in node.values])
            elif isinstance(node.op, ast.Or):
                return cas.logic_or(*[self.evaluate_expression(x, observation_state_symbols) for x in node.values])
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return cas.logic_not(self.evaluate_expression(node.operand, observation_state_symbols))
        elif isinstance(node, ast.Str):
            # replace monitor name with its state expression
            return observation_state_symbols[node.value]
        raise Exception(f'failed to parse {node}')

    def logic_str_to_expr(self, logic_str: str, default: cas.Expression,
                          observation_state_symbols: Optional[Dict[str, cas.Expression]] = None) -> cas.Expression:
        if observation_state_symbols is None:
            task_state_symbols = {key: value.get_state_expression() for key, value in self.tasks.items()}
            monitor_state_symbols = {key: value.get_state_expression() for key, value in self.monitors.items()}
            observation_state_symbols = {**task_state_symbols, **monitor_state_symbols}
        if logic_str == '':
            return default
        tree = ast.parse(logic_str, mode='eval')
        try:
            return self.evaluate_expression(tree.body, observation_state_symbols)
        except KeyError as e:
            raise GiskardException(f'Unknown symbol {e}')
        except Exception as e:
            raise GiskardException(str(e))

    @profile
    def compile_monitors(self) -> None:
        self.initialize_states()
        self.compile_task_state_updater()
        self.compile_monitor_state_updater()
        # self._register_expression_update_triggers()

    @profile
    def get_monitor(self, name: str) -> Monitor:
        for monitor in self.monitors.values():
            if monitor.name == name:
                return monitor
        raise KeyError(f'No monitor of name \'{name}\' found.')

    def get_observation_state_symbols(self) -> List[cas.Symbol]:
        symbols = []
        for monitor in self.tasks.values():
            symbols.append(monitor.get_state_expression())
        for monitor in self.monitors.values():
            symbols.append(monitor.get_state_expression())
        return symbols

    def initialize_states(self) -> None:
        self.task_observation_state = np.zeros(len(self.tasks))
        self.monitor_observation_state = np.zeros(len(self.monitors))
        self.task_life_cycle_state = np.zeros(len(self.tasks))
        self.monitor_life_cycle_state = np.zeros(len(self.monitors))
        for task in self.tasks.values():
            if cas.is_true(task.start_condition):
                self.task_life_cycle_state[task.id] = LifeCycleState.running
            else:
                self.task_life_cycle_state[task.id] = LifeCycleState.not_started
        for monitor in self.monitors.values():
            if cas.is_true(monitor.start_condition):
                self.monitor_life_cycle_state[monitor.id] = LifeCycleState.running
            else:
                self.monitor_life_cycle_state[monitor.id] = LifeCycleState.not_started

    def get_monitor_from_state_expr(self, expr: cas.Expression) -> Monitor:
        for task in self.tasks.values():
            if cas.is_true(task.get_state_expression() == expr):
                return task
        for monitor in self.monitors.values():
            if cas.is_true(monitor.get_state_expression() == expr):
                return monitor
        raise GiskardException(f'No task/monitor found for {str(expr)}.')

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
    def compile_task_state_updater(self) -> None:
        task_state_updater = []
        node: MotionGraphNode
        for node in self.tasks.values():
            state_symbol = node.get_state_expression()
            node.pre_compile()
            state_f = cas.if_eq(node.get_life_cycle_state_expression(), int(LifeCycleState.running),
                                if_result=node.expression,
                                else_result=state_symbol)
            task_state_updater.append(state_f)
        task_state_updater = cas.Expression(task_state_updater)
        self.compiled_task_observation_state = task_state_updater.compile(task_state_updater.free_symbols())
        self.compiled_task_life_cycle_state = compile_graph_node_state_updater(self.tasks)

    @profile
    def compile_monitor_state_updater(self) -> None:
        monitor_state_updater = []
        for node in self.monitors.values():
            state_symbol = node.get_state_expression()
            if isinstance(node, PayloadMonitor):
                state_f = state_symbol  # if payload monitor, copy last state
            else:
                node.pre_compile()
                state_f = cas.if_eq(node.get_life_cycle_state_expression(), int(LifeCycleState.running),
                                    if_result=node.expression,
                                    else_result=state_symbol)
            monitor_state_updater.append(state_f)
        monitor_state_updater = cas.Expression(monitor_state_updater)
        self.compiled_monitor_observation_state = monitor_state_updater.compile(monitor_state_updater.free_symbols())
        self.compiled_monitor_life_cycle_state_updater = compile_graph_node_state_updater(self.monitors)

    @property
    def expression_monitors(self) -> List[ExpressionMonitor]:
        return [x for x in self.monitors if isinstance(x, ExpressionMonitor)]

    @property
    def payload_monitors(self) -> List[PayloadMonitor]:
        return [x for x in self.monitors.values() if isinstance(x, PayloadMonitor)]

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
                def __init__(self, updater_id: int, values, motion_graph_manager: MotionGraphManager):
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
        args = symbol_manager.resolve_symbols(self.compiled_task_observation_state.str_params)
        self.task_observation_state = self.compiled_task_observation_state.fast_call(args)

        args = symbol_manager.resolve_symbols(self.compiled_monitor_observation_state.str_params)
        self.monitor_observation_state = self.compiled_monitor_observation_state.fast_call(args)

        # %% update life cycle state
        args = np.concatenate((self.task_life_cycle_state, self.task_observation_state, self.monitor_observation_state))
        self.task_life_cycle_state = self.compiled_task_life_cycle_state.fast_call(args)
        args = np.concatenate((self.monitor_life_cycle_state, self.task_observation_state, self.monitor_observation_state))
        self.monitor_life_cycle_state = self.compiled_monitor_life_cycle_state_updater.fast_call(args)

        if len(self.payload_monitor_filter) > 0:
            self.monitor_observation_state[self.payload_monitor_filter] = self.evaluate_payload_monitors()
        # self.trigger_update_triggers(self.state)
        self.task_state_history.append((god_map.time,
                                        (self.task_observation_state.copy(),
                                         self.task_life_cycle_state.copy())))
        self.monitor_state_history.append((god_map.time,
                                           (self.monitor_observation_state.copy(),
                                            self.monitor_life_cycle_state.copy())))

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

    @profile
    def get_constraints_from_tasks(self) \
            -> Tuple[List[EqualityConstraint],
            List[InequalityConstraint],
            List[DerivativeInequalityConstraint],
            List[QuadraticWeightGain],
            List[LinearWeightGain]]:
        eq_constraints = ImmutableDict()
        neq_constraints = ImmutableDict()
        derivative_constraints = ImmutableDict()
        quadratic_weight_gains = ImmutableDict()
        linear_weight_gains = ImmutableDict()
        for task_name, task in list(self.tasks.items()):
            try:
                new_eq_constraints = OrderedDict()
                new_neq_constraints = OrderedDict()
                new_derivative_constraints = OrderedDict()
                new_quadratic_weight_gains = OrderedDict()
                new_linear_weight_gains = OrderedDict()
                for constraint in task.get_eq_constraints():
                    new_eq_constraints[constraint.name] = constraint
                for constraint in task.get_neq_constraints():
                    new_neq_constraints[constraint.name] = constraint
                for constraint in task.get_derivative_constraints():
                    new_derivative_constraints[constraint.name] = constraint
                for gain in task.get_quadratic_gains():
                    new_quadratic_weight_gains[gain.name] = gain
                for gain in task.get_linear_gains():
                    new_linear_weight_gains[gain.name] = gain
            except Exception as e:
                raise GoalInitalizationException(str(e))
            eq_constraints.update(new_eq_constraints)
            neq_constraints.update(new_neq_constraints)
            derivative_constraints.update(new_derivative_constraints)
            quadratic_weight_gains.update(new_quadratic_weight_gains)
            linear_weight_gains.update(new_linear_weight_gains)
            # logging.loginfo(f'{goal_name} added {len(_constraints)+len(_vel_constraints)} constraints.')
        return (list(eq_constraints.values()), list(neq_constraints.values()), list(derivative_constraints.values()),
                list(quadratic_weight_gains.values()), list(linear_weight_gains.values()))
