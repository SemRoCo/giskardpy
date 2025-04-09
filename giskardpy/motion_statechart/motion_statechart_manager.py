import ast
from collections import OrderedDict
from functools import cached_property
from itertools import chain
from typing import List, Tuple, Dict, Optional, Union, Iterable, Set
from line_profiler import profile

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types.data_types import LifeCycleState, ObservationState
from giskardpy.data_types.exceptions import GiskardException, GoalInitalizationException, UnknownGoalException
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.helpers import compile_graph_node_state_updater, MotionGraphNodeStateManager
from giskardpy.motion_statechart.monitors.monitors import Monitor, EndMotion, CancelMotion
from giskardpy.motion_statechart.monitors.payload_monitors import PayloadMonitor
from giskardpy.motion_statechart.tasks.task import Task
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.utils import get_all_classes_in_package, ImmutableDict
from giskardpy.qp.constraint import DerivativeEqualityConstraint


def monitor_list_to_monitor_name_tuple(monitors: Iterable[Union[str, Monitor]]) -> Tuple[str, ...]:
    return tuple(sorted(monitor.name if isinstance(monitor, Monitor) else monitor for monitor in monitors))


class MotionStatechartManager:
    task_state: MotionGraphNodeStateManager[Task]
    monitor_state: MotionGraphNodeStateManager[Monitor]
    goal_state: MotionGraphNodeStateManager[Goal]

    observation_state_updater: cas.StackedCompiledFunction
    life_cycle_updater: cas.StackedCompiledFunction

    task_state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)
    monitor_state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)
    goal_state_history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]  # time -> (state, life_cycle_state)

    def __init__(self):
        self.allowed_monitor_types = {}
        self.allowed_task_types = {}
        self.allowed_goal_types = {}
        self.add_monitor_package_path('giskardpy.motion_statechart.monitors')
        self.add_task_package_path('giskardpy.motion_statechart.tasks')
        self.add_goal_package_path('giskardpy.motion_statechart.goals')
        self.reset()

    def add_monitor_package_path(self, path: str) -> None:
        self.allowed_monitor_types.update(get_all_classes_in_package(path, Monitor))

    def add_task_package_path(self, path: str) -> None:
        self.allowed_task_types.update(get_all_classes_in_package(path, Task))

    def add_goal_package_path(self, path: str) -> None:
        self.allowed_goal_types.update(get_all_classes_in_package(path, Goal))

    def get_parent_node_name_of_node(self, node: MotionStatechartNode) -> str:
        for goal in self.goal_state.nodes:
            if node in goal.tasks + goal.monitors + goal.goals:
                return goal.name
        return ''

    def reset(self):
        try:
            del self.payload_monitor_filter
        except Exception as e:
            pass
        self.task_state = MotionGraphNodeStateManager(god_map_path='god_map.motion_statechart_manager.task_state')
        self.monitor_state = MotionGraphNodeStateManager(god_map_path='god_map.motion_statechart_manager.monitor_state')
        self.goal_state = MotionGraphNodeStateManager(god_map_path='god_map.motion_statechart_manager.goal_state')
        self.task_state_history = []
        self.monitor_state_history = []
        self.goal_state_history = []

    def get_all_node_names(self) -> Set[str]:
        return self.task_state.get_node_names() | self.monitor_state.get_node_names() | self.goal_state.get_node_names()

    def add_node(self, node: Union[Monitor, Task, Goal]) -> None:
        if isinstance(node, Monitor):
            self.add_monitor(node)
        elif isinstance(node, Task):
            self.add_task(node)
        elif isinstance(node, Goal):
            self.add_goal(node)
        else:
            raise NotImplementedError(f'Cannot add node type {type(node)}.')

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

    def parse_conditions(self) -> None:
        for goal in self.goal_state.nodes:
            if self.get_parent_node_name_of_node(goal) == '':
                self.apply_conditions_to_sub_nodes(goal)

        task_state_symbols = self.task_state.get_observation_state_symbol_map()
        monitor_state_symbols = self.monitor_state.get_observation_state_symbol_map()
        goal_state_symbols = self.goal_state.get_observation_state_symbol_map()
        observation_state_symbols = {**task_state_symbols, **monitor_state_symbols, **goal_state_symbols}

        for node in chain(self.monitor_state.nodes, self.task_state.nodes, self.goal_state.nodes):
            start_condition = god_map.motion_statechart_manager.logic_str_to_expr(
                logic_str=node.start_condition,
                default=cas.BinaryTrue,
                observation_state_symbols=observation_state_symbols)
            pause_condition = god_map.motion_statechart_manager.logic_str_to_expr(
                logic_str=node.pause_condition,
                default=cas.BinaryFalse,
                observation_state_symbols=observation_state_symbols)
            end_condition = god_map.motion_statechart_manager.logic_str_to_expr(
                logic_str=node.end_condition,
                default=cas.BinaryFalse,
                observation_state_symbols=observation_state_symbols)
            reset_condition = god_map.motion_statechart_manager.logic_str_to_expr(
                logic_str=node.reset_condition,
                default=cas.BinaryFalse,
                observation_state_symbols=observation_state_symbols)
            node.set_conditions(start_condition=start_condition,
                                reset_condition=reset_condition,
                                pause_condition=pause_condition,
                                end_condition=end_condition)

    # def _parse_layer(self, monitors: List[Monitor], tasks: List[Task], goals: List[Goal]):
    #     for goal in goals:
    #         if self.get_parent_node_name_of_node(goal) == '':
    #             self.apply_conditions_to_sub_nodes(goal)

    def apply_conditions_to_sub_nodes(self, goal: Goal):
        for node in goal.tasks + goal.monitors + goal.goals:
            if node.start_condition == 'True':
                node.start_condition = goal.start_condition

            if node.pause_condition == 'False':
                node.pause_condition = goal.pause_condition
            elif goal.pause_condition != 'False':
                node.pause_condition = f'({node.pause_condition}) or ({goal.pause_condition})'

            if node.end_condition == 'False':
                node.end_condition = goal.end_condition
            elif goal.pause_condition != 'False':
                node.end_condition = f'({node.end_condition}) or ({goal.end_condition})'

            if node.reset_condition == 'False':
                node.reset_condition = goal.reset_condition
            elif goal.pause_condition != 'False':
                node.reset_condition = f'({node.reset_condition}) or ({goal.reset_condition})'

            if isinstance(node, Goal):
                self.apply_conditions_to_sub_nodes(node)

    def parse_ast_expression(self, node,
                             observation_state_symbols: Dict[str, cas.Expression]) -> cas.Expression:
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return cas.logic_and3(*[self.parse_ast_expression(x, observation_state_symbols) for x in node.values])
            elif isinstance(node.op, ast.Or):
                return cas.logic_or3(*[self.parse_ast_expression(x, observation_state_symbols) for x in node.values])
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return cas.logic_not3(self.parse_ast_expression(node.operand, observation_state_symbols))
        elif isinstance(node, ast.Str):
            # replace monitor name with its state expression
            return observation_state_symbols[node.value]
        elif isinstance(node, ast.Constant):  # Handle True, False, and other literals
            if isinstance(node.value, bool):  # Check if it's True or False
                if node.value:
                    return cas.TrinaryTrue
                else:
                    return cas.TrinaryFalse
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
        task_life_cycle_expr, task_obs_expr = self.compile_node_state_updater(self.task_state)
        monitor_life_cycle_expr, monitor_obs_expr = self.compile_node_state_updater(self.monitor_state)
        goal_life_cycle_expr, goal_obs_expr = self.compile_node_state_updater(self.goal_state)

        self.life_cycle_updater = cas.StackedCompiledFunction(
            expressions=[task_life_cycle_expr,
                         monitor_life_cycle_expr,
                         goal_life_cycle_expr],
            parameters=self.task_state.get_life_cycle_state_symbols()
                       + self.monitor_state.get_life_cycle_state_symbols()
                       + self.goal_state.get_life_cycle_state_symbols()
                       + god_map.motion_statechart_manager.get_observation_state_symbols())

        self.observation_state_updater = cas.StackedCompiledFunction(
            expressions=[task_obs_expr,
                         monitor_obs_expr,
                         goal_obs_expr],
            parameters=task_obs_expr.free_symbols() + monitor_obs_expr.free_symbols() + goal_obs_expr.free_symbols())

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

    def get_node_from_state_expr(self, expr: cas.Expression) -> MotionStatechartNode:
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

    def get_node(self, name: str) -> MotionStatechartNode:
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
            -> Tuple[cas.Expression, cas.Expression]:
        observation_state_updater = []
        node: MotionStatechartNode
        for node in node_state.nodes:
            state_symbol = node.get_observation_state_expression()
            node.pre_compile()
            if isinstance(node, PayloadMonitor):
                expression = state_symbol  # if payload monitor, copy last state
            else:
                expression = node.observation_expression
            state_f = cas.if_eq_cases(a=node.get_life_cycle_state_expression(),
                                      b_result_cases=[(int(LifeCycleState.running), expression),
                                                      (int(LifeCycleState.not_started), ObservationState.unknown)],
                                      else_result=state_symbol)
            observation_state_updater.append(state_f)
        observation_state_updater = cas.Expression(observation_state_updater)
        life_cycle_expression = compile_graph_node_state_updater(node_state)
        return life_cycle_expression, observation_state_updater

    @property
    def payload_monitors(self) -> List[PayloadMonitor]:
        return [x for x in self.monitor_state.nodes if isinstance(x, PayloadMonitor)]

    @profile
    def register_expression_updater(self, expression: cas.PreservedCasType, node: MotionStatechartNode) \
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
        obs_args = symbol_manager.resolve_symbols(self.observation_state_updater.str_params)

        next_state, done, exception = self.evaluate_payload_monitors()

        obs_result = self.observation_state_updater.fast_call(obs_args)
        self.task_state.observation_state = obs_result[0]
        self.monitor_state.observation_state = obs_result[1]
        self.goal_state.observation_state = obs_result[2]

        if len(self.payload_monitors) > 0:
            self.monitor_state.observation_state[self.payload_monitor_filter] = next_state

        # %% update life cycle state
        args = symbol_manager.resolve_symbols(self.life_cycle_updater.str_params)
        life_cycle_result = self.life_cycle_updater.fast_call(args)
        self.task_state.life_cycle_state = life_cycle_result[0]
        self.monitor_state.life_cycle_state = life_cycle_result[1]
        self.goal_state.life_cycle_state = life_cycle_result[2]

        self.trigger_update_triggers()

        self.log_states()
        if not done and exception is not None:
            raise exception
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

    def evaluate_payload_monitors(self) -> Tuple[np.ndarray, bool, Optional[Exception]]:
        done = False
        cancel_exception = None
        next_state = np.zeros(len(self.payload_monitors))
        if len(self.payload_monitor_filter) == 0:
            return next_state, False, None
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
        return next_state, done, cancel_exception

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
            if not isinstance(m, (CancelMotion, EndMotion)):
                return True
        return False

    @profile
    def get_constraints_from_tasks(self) \
            -> Tuple[List[EqualityConstraint],
            List[InequalityConstraint],
            List[DerivativeEqualityConstraint],
            List[DerivativeInequalityConstraint],
            List[QuadraticWeightGain],
            List[LinearWeightGain]]:
        eq_constraints = ImmutableDict()
        neq_constraints = ImmutableDict()
        eq_derivative_constraints = ImmutableDict()
        derivative_constraints = ImmutableDict()
        quadratic_weight_gains = ImmutableDict()
        linear_weight_gains = ImmutableDict()
        for task in self.task_state.nodes:
            try:
                new_eq_constraints = OrderedDict()
                new_neq_constraints = OrderedDict()
                new_derivative_constraints = OrderedDict()
                new_eq_derivative_constraints = OrderedDict()
                new_quadratic_weight_gains = OrderedDict()
                new_linear_weight_gains = OrderedDict()
                for constraint in task.get_eq_constraints():
                    new_eq_constraints[constraint.name] = constraint
                for constraint in task.get_neq_constraints():
                    new_neq_constraints[constraint.name] = constraint
                for constraint in task.get_eq_derivative_constraints():
                    new_eq_derivative_constraints[constraint.name] = constraint
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
            eq_derivative_constraints.update(new_eq_derivative_constraints)
            derivative_constraints.update(new_derivative_constraints)
            quadratic_weight_gains.update(new_quadratic_weight_gains)
            linear_weight_gains.update(new_linear_weight_gains)
            # logging.loginfo(f'{goal_name} added {len(_constraints)+len(_vel_constraints)} constraints.')
        return (list(eq_constraints.values()),
                list(neq_constraints.values()),
                list(eq_derivative_constraints.values()),
                list(derivative_constraints.values()),
                list(quadratic_weight_gains.values()),
                list(linear_weight_gains.values()))
