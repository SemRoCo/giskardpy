import ast
from collections import OrderedDict
from functools import cached_property
from typing import List, Tuple, Dict, Optional, Callable, Union, Iterable, Set
from line_profiler import profile

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types.data_types import LifeCycleState, ObservationState
from giskardpy.data_types.exceptions import GiskardException, UnknownMonitorException, \
    UnknownTaskException, GoalInitalizationException
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.motion_graph.graph_node import MotionGraphNode
from giskardpy.motion_graph.helpers import compile_graph_node_state_updater, MotionGraphNodeStateManager
from giskardpy.motion_graph.monitors.monitors import Monitor, Monitor, EndMotion, CancelMotion
from giskardpy.motion_graph.monitors.payload_monitors import PayloadMonitor
from giskardpy.motion_graph.tasks.task import Task
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.utils import get_all_classes_in_package, ImmutableDict


def monitor_list_to_monitor_name_tuple(monitors: Iterable[Union[str, Monitor]]) -> Tuple[str, ...]:
    return tuple(sorted(monitor.name if isinstance(monitor, Monitor) else monitor for monitor in monitors))


class MotionGraphManager:
    task_state: MotionGraphNodeStateManager[Task]
    monitor_state: MotionGraphNodeStateManager[Monitor]
    goal_state: MotionGraphNodeStateManager[Goal]

    compiled_task_observation_state: CompiledFunction
    compiled_monitor_observation_state: CompiledFunction
    compiled_goal_observation_state: CompiledFunction

    compiled_task_life_cycle_state: CompiledFunction
    compiled_monitor_life_cycle_state: CompiledFunction
    compiled_goal_life_cycle_state: CompiledFunction

    task_state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)
    monitor_state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)
    goal_state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)

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

    def get_parent_node_name_of_node(self, node: MotionGraphNode) -> str:
        for goal in self.goal_state.nodes:
            if node in goal.tasks + goal.monitors + goal.goals:
                return goal.name
        return ''

    def reset(self):
        try:
            del self.payload_monitor_filter
        except Exception as e:
            pass
        self.task_state = MotionGraphNodeStateManager(god_map_path='god_map.motion_graph_manager.task_state')
        self.monitor_state = MotionGraphNodeStateManager(god_map_path='god_map.motion_graph_manager.monitor_state')
        self.goal_state = MotionGraphNodeStateManager(god_map_path='god_map.motion_graph_manager.goal_state')
        self.task_state_history = []
        self.monitor_state_history = []
        self.goal_state_history = []

    def get_all_node_names(self) -> Set[str]:
        return self.task_state.get_node_names() | self.monitor_state.get_node_names() | self.goal_state.get_node_names()

    def add_monitor(self, monitor: Monitor) -> None:
        self.check_if_node_name_unique(monitor.name)
        self.monitor_state.append(monitor)

    def add_task(self, task: Task) -> None:
        self.check_if_node_name_unique(task.name)
        self.task_state.append(task)

    def add_goal(self, goal: Goal) -> None:
        self.check_if_node_name_unique(goal.name)
        self.goal_state.append(goal)
        for sub_goal in goal.goals:
            self.add_goal(sub_goal)
        for task in goal.tasks:
            self.add_task(task)
        for monitor in goal.monitors:
            self.add_monitor(monitor)

    def check_if_node_name_unique(self, node_name: str) -> None:
        if node_name in self.get_all_node_names():
            raise ValueError(f'Node "{node_name}" already exists')

    def parse_motion_graph(self, *,
                           tasks: List[Tuple[str, str, str, str, str, str, dict]],
                           monitors: List[Tuple[str, str, str, str, str, str, dict]],
                           goals: List[Tuple[str, str, str, str, str, str, dict]]) -> None:
        # %% add goals
        for class_name, name, start, reset, pause, end, kwargs in goals:
            try:
                get_middleware().loginfo(f'Adding task of type: \'{class_name}\'')
                C = god_map.motion_graph_manager.allowed_goal_types[class_name]
            except KeyError:
                raise UnknownTaskException(f'unknown task type: \'{class_name}\'.')
            goal: Goal = C(name=name, **kwargs)
            self.add_goal(goal)

        # %% add tasks
        for class_name, name, start, reset, pause, end, kwargs in tasks:
            try:
                get_middleware().loginfo(f'Adding task of type: \'{class_name}\'')
                C = god_map.motion_graph_manager.allowed_task_types[class_name]
            except KeyError:
                raise UnknownTaskException(f'unknown task type: \'{class_name}\'.')
            task: Task = C(name=name, **kwargs)
            self.add_task(task)

        # %% add monitors
        for class_name, name, start, reset, pause, end, kwargs in monitors:
            try:
                get_middleware().loginfo(f'Adding monitor of type: \'{class_name}\'')
                C = god_map.motion_graph_manager.allowed_monitor_types[class_name]
            except KeyError:
                raise UnknownMonitorException(f'unknown monitor type: \'{class_name}\'.')
            monitor: Monitor = C(name=name, **kwargs)
            self.add_monitor(monitor)

        task_state_symbols = self.task_state.get_observation_state_symbol_map()
        monitor_state_symbols = self.monitor_state.get_observation_state_symbol_map()
        goal_state_symbols = self.goal_state.get_observation_state_symbol_map()
        observation_state_symbols = {**task_state_symbols, **monitor_state_symbols, **goal_state_symbols}

        for (class_name, name, start, reset, pause, end, kwargs) in tasks + monitors + goals:
            node = self.get_node(name)
            start_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=start,
                default=cas.BinaryTrue,
                observation_state_symbols=observation_state_symbols)
            pause_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=pause,
                default=cas.BinaryFalse,
                observation_state_symbols=observation_state_symbols)
            end_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=end,
                default=cas.BinaryFalse,
                observation_state_symbols=observation_state_symbols)
            reset_condition = god_map.motion_graph_manager.logic_str_to_expr(
                logic_str=reset,
                default=cas.BinaryFalse,
                observation_state_symbols=observation_state_symbols)
            node.set_conditions(start_condition=start_condition,
                                reset_condition=reset_condition,
                                pause_condition=pause_condition,
                                end_condition=end_condition)

        # %% apply goal conditions to sub nodes
        for goal in self.goal_state.nodes:
            if self.get_parent_node_name_of_node(goal) == '':
                self.apply_conditions_to_sub_nodes(goal)

    def create_logic3_conditions(self) -> None:
        for node in self.task_state.nodes + self.monitor_state.nodes + self.goal_state.nodes:
            node.logic3_start_condition = cas.replace_with_three_logic(node.start_condition)
            node.logic3_pause_condition = cas.replace_with_three_logic(node.pause_condition)
            node.logic3_end_condition = cas.replace_with_three_logic(node.end_condition)
            node.logic3_reset_condition = cas.replace_with_three_logic(node.reset_condition)

    def apply_conditions_to_sub_nodes(self, goal: Goal):
        for node in goal.tasks + goal.monitors + goal.goals:
            if cas.is_true_symbol(node.start_condition):
                node.start_condition = goal.start_condition
            node.reset_condition = cas.logic_or(node.reset_condition, goal.reset_condition)
            node.pause_condition = cas.logic_or(node.pause_condition, goal.pause_condition)
            node.end_condition = cas.logic_or(node.end_condition, goal.end_condition)
            if isinstance(node, Goal):
                self.apply_conditions_to_sub_nodes(node)

    def parse_ast_expression(self, node,
                             observation_state_symbols: Dict[str, cas.Expression]) -> cas.Expression:
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return cas.logic_and(*[self.parse_ast_expression(x, observation_state_symbols) for x in node.values])
            elif isinstance(node.op, ast.Or):
                return cas.logic_or(*[self.parse_ast_expression(x, observation_state_symbols) for x in node.values])
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return cas.logic_not(self.parse_ast_expression(node.operand, observation_state_symbols))
        elif isinstance(node, ast.Str):
            # replace monitor name with its state expression
            return observation_state_symbols[node.value]
        raise Exception(f'failed to parse {node}')

    def logic_str_to_expr(self, logic_str: str, default: cas.Expression,
                          observation_state_symbols: Dict[str, cas.Expression]) -> cas.Expression:
        if logic_str == '':
            return default
        tree = ast.parse(logic_str, mode='eval')
        try:
            expr = self.parse_ast_expression(tree.body, observation_state_symbols)
            return expr
        except KeyError as e:
            raise GiskardException(f'Unknown symbol {e}')
        except Exception as e:
            raise GiskardException(str(e))

    @profile
    def compile_node_state_updaters(self) -> None:
        self.create_logic3_conditions()
        self.compiled_task_observation_state, self.compiled_task_life_cycle_state = self.compile_node_state_updater(
            self.task_state)
        self.compiled_monitor_observation_state, self.compiled_monitor_life_cycle_state = self.compile_node_state_updater(
            self.monitor_state)
        self.compiled_goal_observation_state, self.compiled_goal_life_cycle_state = self.compile_node_state_updater(
            self.goal_state)
        self.initialize_states()

    def get_observation_state_symbols(self) -> List[cas.Symbol]:
        return (list(self.task_state.get_observation_state_symbol_map().values()) +
                list(self.monitor_state.get_observation_state_symbol_map().values()) +
                list(self.goal_state.get_observation_state_symbol_map().values()))

    def get_life_cycle_state_symbols(self) -> List[cas.Symbol]:
        return (self.task_state.get_life_cycle_state_symbols() +
                self.monitor_state.get_life_cycle_state_symbols() +
                self.goal_state.get_life_cycle_state_symbols())

    def initialize_states(self) -> None:
        self.task_state.init_states()
        self.monitor_state.init_states()
        self.goal_state.init_states()
        self.log_states()

    def get_node_from_state_expr(self, expr: cas.Expression) -> MotionGraphNode:
        for task in self.task_state.nodes:
            if cas.is_true_symbol(task.get_observation_state_expression() == expr):
                return task
        for monitor in self.monitor_state.nodes:
            if cas.is_true_symbol(monitor.get_observation_state_expression() == expr):
                return monitor
        for goal in self.goal_state.nodes:
            if cas.is_true_symbol(goal.get_observation_state_expression() == expr):
                return goal
        raise GiskardException(f'No goal/task/monitor found for {str(expr)}.')

    def get_node(self, name: str) -> MotionGraphNode:
        try:
            return self.task_state.get_node(name)
        except KeyError:
            pass
        try:
            return self.monitor_state.get_node(name)
        except KeyError:
            pass
        try:
            return self.goal_state.get_node(name)
        except KeyError:
            pass
        raise GiskardException(f'No goal/task/monitor found for {str(name)}.')

    def is_node_registered(self, monitor_state_expr: cas.Expression) -> bool:
        try:
            self.get_node_from_state_expr(monitor_state_expr)
            return True
        except GiskardException as e:
            return False

    def format_condition(self, condition: cas.Expression, new_line: str = '\n') -> str:
        """
        Takes a logical expression, replaces the state symbols with monitor names and formats it nicely.
        """
        free_symbols = condition.free_symbols()
        if not free_symbols:
            return str(cas.is_true_symbol(condition))
        condition_str = condition.pretty_str()[0][0]
        state_to_monitor_map = {str(x): f'\'{self.get_node_from_state_expr(x).name}\'' for x in free_symbols}
        state_to_monitor_map['&&'] = f'{new_line}and '
        state_to_monitor_map['||'] = f'{new_line}or '
        state_to_monitor_map['!'] = 'not '
        state_to_monitor_map['==1'] = ''
        for state_str, monitor_name in state_to_monitor_map.items():
            condition_str = condition_str.replace(state_str, monitor_name)
        return condition_str

    @profile
    def compile_node_state_updater(self, node_state: MotionGraphNodeStateManager) \
            -> Tuple[cas.CompiledFunction, cas.CompiledFunction]:
        observation_state_updater = []
        node: MotionGraphNode
        for node in node_state.nodes:
            state_symbol = node.get_observation_state_expression()
            node.pre_compile()
            if isinstance(node, PayloadMonitor):
                expression = state_symbol  # if payload monitor, copy last state
            else:
                expression = node.expression
            state_f = cas.if_eq_cases(a=node.get_life_cycle_state_expression(),
                                      b_result_cases=[(int(LifeCycleState.running), expression),
                                                      (int(LifeCycleState.not_started), ObservationState.unknown)],
                                      else_result=state_symbol)
            observation_state_updater.append(state_f)
        observation_state_updater = cas.Expression(observation_state_updater)
        compiled_node_observation_state = observation_state_updater.compile(observation_state_updater.free_symbols())
        compiled_node_life_cycle_state = compile_graph_node_state_updater(node_state)
        return compiled_node_observation_state, compiled_node_life_cycle_state

    @property
    def payload_monitors(self) -> List[PayloadMonitor]:
        return [x for x in self.monitor_state.nodes if isinstance(x, PayloadMonitor)]

    @profile
    def register_expression_updater(self, expression: cas.PreservedCasType, node: MotionGraphNode) \
            -> cas.PreservedCasType:
        """
        Expression is updated when all monitors are 1 at the same time, but only once.
        """
        if isinstance(node, Task):
            return self.task_state.register_expression_updater(node=node, expression=expression)
        if isinstance(node, Monitor):
            return self.monitor_state.register_expression_updater(node=node, expression=expression)
        if isinstance(node, Goal):
            return self.goal_state.register_expression_updater(node=node, expression=expression)

    @profile
    def trigger_update_triggers(self):
        self.task_state.trigger_update_triggers()
        self.monitor_state.trigger_update_triggers()
        self.goal_state.trigger_update_triggers()

    @cached_property
    def payload_monitor_filter(self):
        return np.array([i for i, m in enumerate(self.monitor_state.nodes) if isinstance(m, PayloadMonitor)])

    @profile
    def evaluate_node_states(self) -> bool:
        # %% update observation state
        task_args = symbol_manager.resolve_symbols(self.compiled_task_observation_state.str_params)
        monitor_args = symbol_manager.resolve_symbols(self.compiled_monitor_observation_state.str_params)
        goal_args = symbol_manager.resolve_symbols(self.compiled_goal_observation_state.str_params)

        next_state, done = self.evaluate_payload_monitors()

        self.task_state.observation_state = self.compiled_task_observation_state.fast_call(task_args)
        self.monitor_state.observation_state = self.compiled_monitor_observation_state.fast_call(monitor_args)
        self.goal_state.observation_state = self.compiled_goal_observation_state.fast_call(goal_args)
        self.monitor_state.observation_state[self.payload_monitor_filter] = next_state

        obs_state = np.concatenate((self.task_state.observation_state,
                                    self.monitor_state.observation_state,
                                    self.goal_state.observation_state))

        # %% update life cycle state
        args = np.concatenate((self.task_state.life_cycle_state,
                               obs_state))
        self.task_state.life_cycle_state = self.compiled_task_life_cycle_state.fast_call(args)
        args = np.concatenate((self.monitor_state.life_cycle_state,
                               obs_state))
        self.monitor_state.life_cycle_state = self.compiled_monitor_life_cycle_state.fast_call(args)
        args = np.concatenate((self.goal_state.life_cycle_state,
                               obs_state))
        self.goal_state.life_cycle_state = self.compiled_goal_life_cycle_state.fast_call(args)

        self.trigger_update_triggers()

        self.log_states()
        if isinstance(done, Exception):
            raise done
        return done

    def log_states(self) -> None:
        self.task_state_history.append((god_map.time,
                                        (self.task_state.observation_state.copy(),
                                         self.task_state.life_cycle_state.copy())))
        self.monitor_state_history.append((god_map.time,
                                           (self.monitor_state.observation_state.copy(),
                                            self.monitor_state.life_cycle_state.copy())))
        self.goal_state_history.append((god_map.time,
                                        (self.goal_state.observation_state.copy(),
                                         self.goal_state.life_cycle_state.copy())))

    def evaluate_payload_monitors(self) -> Tuple[np.ndarray, Union[bool, Exception]]:
        done = False
        cancel_exception = None
        next_state = np.zeros(len(self.payload_monitors))
        filtered_life_cycle_state = self.monitor_state.life_cycle_state[self.payload_monitor_filter]
        for i, payload_monitor in enumerate(self.payload_monitors):
            if filtered_life_cycle_state[i] == LifeCycleState.running:
                try:
                    payload_monitor()
                except Exception as e:
                    # the call of cancel motion might trigger exceptions
                    # only raise it if no end motion triggered at the same time
                    cancel_exception = e
            elif filtered_life_cycle_state[i] == LifeCycleState.not_started:
                payload_monitor.state = ObservationState.unknown
            next_state[i] = payload_monitor.get_state()
            done = done or (isinstance(payload_monitor, EndMotion) and next_state[i] == ObservationState.true)
        if not done and cancel_exception is not None:
            done = cancel_exception
        return next_state, done

    def has_end_motion_monitor(self) -> bool:
        for m in self.monitor_state.nodes:
            if isinstance(m, EndMotion):
                return True
        return False

    def has_cancel_motion_monitor(self) -> bool:
        for m in self.monitor_state.nodes:
            if isinstance(m, CancelMotion):
                return True
        return False

    def has_payload_monitors_which_are_not_end_nor_cancel(self) -> bool:
        for m in self.monitor_state.nodes:
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
        for task in self.task_state.nodes:
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
