from collections import OrderedDict
from typing import List, Dict, Tuple
import giskardpy.casadi_wrapper as cas
import numpy as np

from giskardpy.data_types.data_types import PrefixName, TaskState
from giskardpy.data_types.exceptions import GoalInitalizationException, \
    DuplicateNameException
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.motion_graph.helpers import compile_graph_node_state_updater
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.motion_graph.tasks.task import Task
from giskardpy.utils.utils import get_all_classes_in_package, ImmutableDict
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from line_profiler import profile


class MotionGoalManager:
    motion_goals: Dict[str, Goal] = None
    tasks: Dict[PrefixName, Task]
    task_state: np.ndarray
    compiled_state_updater: cas.CompiledFunction
    state_history: List[Tuple[float, np.ndarray]]
    goal_package_paths = {'giskardpy.goals'}

    def __init__(self):
        self.allowed_motion_goal_types = {}
        for path in self.goal_package_paths:
            self.allowed_motion_goal_types.update(get_all_classes_in_package(path, Goal))
        self.reset()

    def reset(self):
        self.motion_goals = {}
        self.tasks = OrderedDict()
        self.state_history = []

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
            task.id = len(self.tasks) - 1

    @profile
    def compile_task_state_updater(self):
        self.compiled_state_updater = compile_graph_node_state_updater(self.tasks)

    @profile
    def update_task_state(self, new_state: np.ndarray) -> None:
        substitutions = np.concatenate((self.task_state, new_state))
        self.task_state = self.compiled_state_updater.fast_call(substitutions)
        self.state_history.append((god_map.time, self.task_state.copy()))

    @profile
    def get_constraints_from_goals(self) \
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
        return (list(eq_constraints.values()), list(neq_constraints.values()), list(derivative_constraints.values()),
                list(quadratic_weight_gains.values()), list(linear_weight_gains.values()))
