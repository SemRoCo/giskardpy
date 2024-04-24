from collections import OrderedDict
from typing import List, Dict, Tuple
import giskardpy.casadi_wrapper as cas
import numpy as np

from giskardpy.data_types.data_types import PrefixName, TaskState
from giskardpy.data_types.exceptions import GoalInitalizationException, \
    DuplicateNameException
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.tasks.task import Task
from giskardpy.utils.utils import get_all_classes_in_package, ImmutableDict
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain


class MotionGoalManager:
    motion_goals: Dict[str, Goal] = None
    tasks: Dict[PrefixName, Task]
    task_state: np.ndarray
    compiled_state_updater: cas.CompiledFunction
    state_history: List[Tuple[float, np.ndarray]]
    goal_package_paths = {'giskardpy.goals'}

    def __init__(self):
        self.motion_goals = {}
        self.tasks = OrderedDict()
        self.allowed_motion_goal_types = {}
        self.state_history = []
        for path in self.goal_package_paths:
            self.allowed_motion_goal_types.update(get_all_classes_in_package(path, Goal))

    def init_task_state(self):
        self.task_state = np.zeros(len(self.tasks))
        for task in self.tasks.values():
            if cas.is_true(task.start_condition):
                self.task_state[task.id] = TaskState.running
            else:
                self.task_state[task.id] = TaskState.not_started
        self.compile_task_state_updater()

    def add_motion_goal(self, goal: Goal) -> None:
        name = goal.name
        if name in self.motion_goals:
            raise DuplicateNameException(f'Motion goal with name {name} already exists.')
        self.motion_goals[name] = goal
        for task in goal.tasks:
            if task.name in self.tasks:
                raise DuplicateNameException(f'Task names {task.name} already exists.')
            self.tasks[task.name] = task
            task.set_id(len(self.tasks) - 1)

    @profile
    def compile_task_state_updater(self):
        symbols = []
        for task in sorted(self.tasks.values(), key=lambda x: x.id):
            symbols.append(task.get_state_expression())
        task_state = cas.Expression(symbols)
        symbols = []
        for i, monitor in enumerate(god_map.monitor_manager.monitors):
            symbols.append(monitor.get_state_expression())
        monitor_state = cas.Expression(symbols)

        state_updater = []
        for task in sorted(self.tasks.values(), key=lambda x: x.id):
            state_symbol = task_state[task.id]

            if not cas.is_true(task.start_condition):
                start_if = cas.if_else(task.start_condition,
                                       if_result=int(TaskState.running),
                                       else_result=state_symbol)
            else:
                start_if = state_symbol
            if not cas.is_true(task.hold_condition):
                hold_if = cas.if_else(task.hold_condition,
                                      if_result=int(TaskState.on_hold),
                                      else_result=int(TaskState.running))
            else:
                hold_if = state_symbol
            if not cas.is_true(task.end_condition):
                else_result = cas.if_else(task.end_condition,
                                          if_result=int(TaskState.succeeded),
                                          else_result=hold_if)
            else:
                else_result = hold_if

            state_f = cas.if_eq_cases(a=state_symbol,
                                      b_result_cases=[(int(TaskState.not_started), start_if),
                                                      (int(TaskState.succeeded), int(TaskState.succeeded))],
                                      else_result=else_result)  # running or on_hold
            state_updater.append(state_f)
        state_updater = cas.Expression(state_updater)
        symbols = task_state.free_symbols() + monitor_state.free_symbols()
        self.compiled_state_updater = state_updater.compile(symbols)

    @profile
    def update_task_state(self, new_state: np.ndarray) -> None:
        substitutions = np.concatenate((self.task_state, new_state))
        self.task_state = self.compiled_state_updater.fast_call(substitutions)
        self.state_history.append((god_map.time, self.task_state.copy()))

    @profile
    def get_constraints_from_goals(self) \
            -> Tuple[Dict[str, EqualityConstraint],
            Dict[str, InequalityConstraint],
            Dict[str, DerivativeInequalityConstraint],
            Dict[str, QuadraticWeightGain],
            Dict[str, LinearWeightGain]]:
        eq_constraints = ImmutableDict()
        neq_constraints = ImmutableDict()
        derivative_constraints = ImmutableDict()
        quadratic_weight_gains = ImmutableDict()
        linear_weight_gains = ImmutableDict()
        for goal_name, goal in list(self.motion_goals.items()):
            try:
                new_eq_constraints = OrderedDict()
                new_neq_constraints = OrderedDict()
                new_derivative_constraints = OrderedDict()
                new_quadratic_weight_gains = OrderedDict()
                new_linear_weight_gains = OrderedDict()
                for task in goal.tasks:
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
        god_map.eq_constraints = eq_constraints
        god_map.neq_constraints = neq_constraints
        god_map.derivative_constraints = derivative_constraints
        god_map.quadratic_weight_gains = quadratic_weight_gains
        god_map.linear_weight_gains = linear_weight_gains
        return eq_constraints, neq_constraints, derivative_constraints, quadratic_weight_gains, linear_weight_gains
