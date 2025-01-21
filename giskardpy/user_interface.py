import re
import traceback
from collections import defaultdict
from itertools import chain
from typing import Optional, List, Dict, Tuple, Union, Type
import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName
from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.data_types.exceptions import SetupException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.collision_avoidance_config import CollisionAvoidanceConfig, DisableCollisionAvoidanceConfig
from giskardpy.model.collision_world_syncer import CollisionEntry
from giskardpy.model.world_config import WorldConfig
from giskardpy.motion_statechart.goals.collision_avoidance import CollisionAvoidance
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.monitors.monitors import Monitor
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetSeedConfiguration, SetOdometry
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.task import Task
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.utils.utils import get_all_classes_in_package
from giskardpy.model.trajectory import Trajectory
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeEqualityConstraint, \
    DerivativeInequalityConstraint
from giskardpy.symbol_manager import symbol_manager
from giskardpy.motion_statechart.monitors.monitors import EndMotion


def quote_node_names(condition: str) -> str:
    operators = {'and', 'or', 'not', '(', ')'}
    pattern = r'(\b(?:and|or|not)\b|\(|\))'
    tokens = re.split(pattern, condition)
    result = []
    for token in tokens:
        if token in operators:
            result.append(token)
        elif token.strip() == '':
            result.append(token)
        else:
            # Check if token is already quoted
            stripped = token.strip()
            if (stripped.startswith('"') and stripped.endswith('"')) or \
                    (stripped.startswith("'") and stripped.endswith("'")):
                result.append(token)
            else:
                # Wrap stripped token in quotes, and preserve leading/trailing spaces
                leading_spaces = len(token) - len(token.lstrip())
                trailing_spaces = len(token) - len(token.rstrip())
                leading = token[:leading_spaces]
                trailing = token[len(token.rstrip()):]
                result.append(f'{leading}"{stripped}"{trailing}')
    return ''.join(result)


class MonitorWrapper:
    _name_prefix = 'M'
    _conditions: Dict[str, Tuple[str, str, str, str]]  # node name to start, pause, end, reset

    def __init__(self):
        self._conditions = {}

    def _generate_default_name(self, class_type: Type[Monitor], name: Optional[str]):
        if name is None:
            name = f'{class_type.__name__}{len(self._conditions)}'
        return name

    def add_monitor(self, *,
                    monitor: Monitor,
                    start_condition: str = '',
                    pause_condition: str = '',
                    end_condition: str = '',
                    reset_condition: str = '') -> str:
        god_map.motion_statechart_manager.add_monitor(monitor)
        self._conditions[monitor.name] = (quote_node_names(start_condition),
                                          quote_node_names(pause_condition),
                                          quote_node_names(end_condition),
                                          quote_node_names(reset_condition))
        return monitor.name

    def add_set_seed_configuration(self,
                                   seed_configuration: Dict[str, float],
                                   name: Optional[str] = None,
                                   group_name: Optional[str] = None,
                                   start_condition: str = '',
                                   reset_condition: str = '') -> str:
        """
        Only meant for use with projection. Changes the world state to seed_configuration before starting planning,
        without having to plan a motion to it like with add_joint_position
        """
        name = self._generate_default_name(SetSeedConfiguration, name)
        monitor = SetSeedConfiguration(seed_configuration=seed_configuration,
                                       group_name=group_name,
                                       name=name)
        return self.add_monitor(monitor=monitor,
                                start_condition=start_condition,
                                pause_condition='',
                                end_condition=monitor.name,
                                reset_condition=reset_condition)

    def add_set_seed_odometry(self,
                              base_pose: cas.TransMatrix,
                              name: Optional[str] = None,
                              group_name: Optional[str] = None,
                              start_condition: str = '',
                              reset_condition: str = '') -> str:
        """
        Only meant for use with projection. Overwrites the odometry transform with base_pose.
        """
        name = self._generate_default_name(SetOdometry, name)
        monitor = SetOdometry(base_pose=base_pose,
                              group_name=group_name,
                              name=name)
        return self.add_monitor(monitor=monitor,
                                start_condition=start_condition,
                                pause_condition='',
                                end_condition=monitor.name,
                                reset_condition=reset_condition)

    def add_end_motion(self,
                       start_condition: str,
                       name: str = 'Done') -> str:
        """
        Ends the motion execution/planning if all start_condition are True.
        Use this to describe when your motion should end.
        """
        return self.add_monitor(monitor=EndMotion(name=name),
                                start_condition=start_condition,
                                pause_condition='',
                                end_condition='',
                                reset_condition='')


class MotionGoalWrapper:
    _name_prefix = 'G'
    _collision_entries: Dict[Tuple[str, str, str], List[CollisionEntry]]
    _conditions: Dict[str, Tuple[str, str, str, str]]  # node name to start, pause, end, reset

    def __init__(self):
        self._conditions = {}
        self._collision_entries = {}

    def _generate_default_name(self, class_type: Union[Type[Task], Type[Goal]], name: Optional[str]):
        if name is None:
            name = f'{class_type.__name__}{len(self._conditions)}'
        return name

    def reset(self):
        self._collision_entries = defaultdict(list)

    def add_motion_goal(self, *,
                        goal: Union[Task, Goal],
                        start_condition: str = '',
                        pause_condition: str = '',
                        end_condition: str = '',
                        reset_condition: str = '') -> str:
        """
        Generic function to add a motion goal.
        :param class_name: Name of a class defined in src/giskardpy/goals
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param pause_condition: a logical expression. Goal will be on hold if it is True and active otherwise
        :param end_condition: a logical expression. Goal will become inactive when this becomes True.
        """
        god_map.motion_statechart_manager.add_task(goal)
        self._conditions[goal.name] = (quote_node_names(start_condition),
                                       quote_node_names(pause_condition),
                                       quote_node_names(end_condition),
                                       quote_node_names(reset_condition))
        return goal.name

    def add_joint_position(self,
                           goal_state: Dict[str, float],
                           name: Optional[str] = None,
                           weight: float = WEIGHT_BELOW_CA,
                           max_velocity: Optional[float] = None,
                           start_condition: str = '',
                           pause_condition: str = '',
                           end_condition: str = '',
                           reset_condition: str = '') -> str:
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param weight: None = use default weight
        :param max_velocity: will be applied to all joints
        """
        name = self._generate_default_name(JointPositionList, name)
        joint_task = JointPositionList(name=name,
                                       goal_state=goal_state,
                                       weight=weight,
                                       max_velocity=max_velocity)
        return self.add_motion_goal(goal=joint_task,
                                    start_condition=start_condition,
                                    pause_condition=pause_condition,
                                    end_condition=end_condition,
                                    reset_condition=reset_condition)

    def add_cartesian_pose(self,
                           goal_pose: cas.TransMatrix,
                           tip_link: Union[str, PrefixName],
                           root_link: Union[str, PrefixName],
                           name: Optional[str] = None,
                           reference_linear_velocity: Optional[float] = None,
                           reference_angular_velocity: Optional[float] = None,
                           absolute: bool = False,
                           weight: Optional[float] = None,
                           start_condition: str = '',
                           pause_condition: str = '',
                           end_condition: str = '',
                           reset_condition: str = '') -> str:
        """
        This goal will use the kinematic chain between root and tip link to move tip link to the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: None = use default weight
        """
        if isinstance(root_link, str):
            root_link = god_map.world.search_for_link_name(root_link)
        if isinstance(tip_link, str):
            tip_link = god_map.world.search_for_link_name(tip_link)
        name = self._generate_default_name(CartesianPose, name)
        goal = CartesianPose(root_link=root_link,
                             tip_link=tip_link,
                             goal_pose=goal_pose,
                             name=name,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             weight=weight,
                             absolute=absolute)
        return self.add_motion_goal(goal=goal,
                                    start_condition=start_condition,
                                    pause_condition=pause_condition,
                                    end_condition=end_condition,
                                    reset_condition=reset_condition)

    def _add_collision_avoidance(self,
                                 collisions: List[CollisionEntry],
                                 start_condition: str = '',
                                 pause_condition: str = '',
                                 end_condition: str = ''):
        key = (start_condition, pause_condition, end_condition)
        self._collision_entries[key].extend(collisions)

    def _add_collision_entries_as_goals(self):
        for (start_condition, pause_condition, end_condition), collision_entries in self._collision_entries.items():
            if (collision_entries[-1].type == CollisionEntry.ALLOW_COLLISION
                    and collision_entries[-1].group1 == CollisionEntry.ALL
                    and collision_entries[-1].group2 == CollisionEntry.ALL):
                continue
            name = 'collision avoidance'
            if start_condition or pause_condition or end_condition:
                name += f'{start_condition}, {pause_condition}, {end_condition}'
            self.add_motion_goal(class_name=CollisionAvoidance.__name__,
                                 name=name,
                                 collision_entries=collision_entries,
                                 start_condition=start_condition,
                                 pause_condition=pause_condition,
                                 end_condition=end_condition)

    def allow_all_collisions(self,
                             start_condition: str = '',
                             pause_condition: str = '',
                             end_condition: str = ''):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      pause_condition=pause_condition,
                                      end_condition=end_condition)


class GiskardWrapper:
    def __init__(self,
                 world_config: WorldConfig,
                 collision_avoidance_config: Optional[CollisionAvoidanceConfig] = None,
                 qp_controller_config: Optional[QPControllerConfig] = None,
                 additional_goal_package_paths: Optional[List[str]] = None,
                 additional_monitor_package_paths: Optional[List[str]] = None):
        # self.world = WorldWrapper(node_name)
        self.monitors = MonitorWrapper()
        self.motion_goals = MotionGoalWrapper()
        # self.clear_motion_goals_and_monitors()
        # self.clear_motion_goals_and_monitors()

        god_map.tmp_folder = 'tmp'
        self.world_config = world_config
        if collision_avoidance_config is None:
            collision_avoidance_config = DisableCollisionAvoidanceConfig()
        self.collision_avoidance_config = collision_avoidance_config
        if qp_controller_config is None:
            qp_controller_config = QPControllerConfig()
        self.qp_controller_config = qp_controller_config
        if additional_goal_package_paths is None:
            additional_goal_package_paths = set()
        for additional_path in additional_goal_package_paths:
            self.add_goal_package_name(additional_path)
        if additional_monitor_package_paths is None:
            additional_monitor_package_paths = set()
        for additional_path in additional_monitor_package_paths:
            self.add_monitor_package_name(additional_path)
        god_map.hack = 0

        with god_map.world.modify_world():
            self.world_config.setup()
        # god_map.world._notify_model_change()
        self.collision_avoidance_config.setup()
        self.collision_avoidance_config._sanity_check()
        god_map.collision_scene.sync()

        self.reset()

    def add_goal_package_name(self, package_name: str):
        new_goals = get_all_classes_in_package(package_name, Goal)
        if len(new_goals) == 0:
            raise SetupException(f'No classes of type \'{Goal.__name__}\' found in {package_name}.')
        get_middleware().loginfo(f'Made goal classes {new_goals} available.')
        god_map.motion_statechart_manager.add_goal_package_path(package_name)

    def add_task_package_name(self, package_name: str):
        new_goals = get_all_classes_in_package(package_name, Task)
        if len(new_goals) == 0:
            raise SetupException(f'No classes of type \'{Goal.__name__}\' found in {package_name}.')
        get_middleware().loginfo(f'Made task classes {new_goals} available.')
        god_map.motion_statechart_manager.add_task_package_path(package_name)

    def add_monitor_package_name(self, package_name: str) -> None:
        new_monitors = get_all_classes_in_package(package_name, Monitor)
        if len(new_monitors) == 0:
            raise SetupException(f'No classes of type \'{Monitor.__name__}\' found in \'{package_name}\'.')
        get_middleware().loginfo(f'Made Monitor classes \'{new_monitors}\' available.')
        god_map.motion_statechart_manager.add_monitor_package_path(package_name)

    def compile(self):
        task_state_symbols = god_map.motion_statechart_manager.task_state.get_observation_state_symbol_map()
        monitor_state_symbols = god_map.motion_statechart_manager.monitor_state.get_observation_state_symbol_map()
        goal_state_symbols = god_map.motion_statechart_manager.goal_state.get_observation_state_symbol_map()
        observation_state_symbols = {**task_state_symbols, **monitor_state_symbols, **goal_state_symbols}

        for (name, (start, reset, pause, end)) in chain(self.motion_goals._conditions.items(),
                                                        self.monitors._conditions.items()):
            node = god_map.motion_statechart_manager.get_node(name)
            start_condition = god_map.motion_statechart_manager.logic_str_to_expr(
                logic_str=start,
                default=cas.BinaryTrue,
                observation_state_symbols=observation_state_symbols)
            pause_condition = god_map.motion_statechart_manager.logic_str_to_expr(
                logic_str=pause,
                default=cas.BinaryFalse,
                observation_state_symbols=observation_state_symbols)
            end_condition = god_map.motion_statechart_manager.logic_str_to_expr(
                logic_str=end,
                default=cas.BinaryFalse,
                observation_state_symbols=observation_state_symbols)
            reset_condition = god_map.motion_statechart_manager.logic_str_to_expr(
                logic_str=reset,
                default=cas.BinaryFalse,
                observation_state_symbols=observation_state_symbols)
            node.set_conditions(start_condition=start_condition,
                                reset_condition=reset_condition,
                                pause_condition=pause_condition,
                                end_condition=end_condition)

        god_map.motion_statechart_manager.compile_node_state_updaters()
        god_map.motion_statechart_manager.initialize_states()

        eq, neq, eqd, neqd, lin_weight, quad_weight = god_map.motion_statechart_manager.get_constraints_from_tasks()
        god_map.qp_controller.init(free_variables=self.get_active_free_symbols(eq, neq, eqd, neqd),
                                   equality_constraints=eq,
                                   inequality_constraints=neq,
                                   eq_derivative_constraints=eqd,
                                   derivative_constraints=neqd)
        god_map.qp_controller.compile()
        god_map.debug_expression_manager.compile_debug_expressions()
        self.traj = Trajectory()

    def get_active_free_symbols(self,
                                eq_constraints: List[EqualityConstraint],
                                neq_constraints: List[InequalityConstraint],
                                eq_derivative_constraints: List[DerivativeEqualityConstraint],
                                derivative_constraints: List[DerivativeInequalityConstraint]):
        symbols = set()
        for c in chain(eq_constraints, neq_constraints, eq_derivative_constraints, derivative_constraints):
            symbols.update(str(s) for s in cas.free_symbols(c.expression))
        free_variables = list(sorted([v for v in god_map.world.free_variables.values() if v.position_name in symbols],
                                     key=lambda x: x.position_name))
        if len(free_variables) == 0:
            raise EmptyProblemException('Goal parsing resulted in no free variables.')
        god_map.free_variables = free_variables
        return free_variables

    def reset(self):
        self.goal_state = {}
        god_map.time = 0
        god_map.control_cycle_counter = 0
        god_map.motion_statechart_manager.reset()

    def step(self):
        import time

        done = god_map.motion_statechart_manager.evaluate_node_states()

        parameters = god_map.qp_controller.get_parameter_names()
        total_time_start = time.time()
        substitutions = symbol_manager.resolve_symbols(parameters)
        parameter_time = time.time() - total_time_start

        qp_time_start = time.time()
        next_cmd = god_map.qp_controller.get_cmd(substitutions)
        qp_time = time.time() - qp_time_start
        # god_map.debug_expression_manager.eval_debug_expressions()

        update_world_time = time.time()
        god_map.world.update_state(next_cmd, god_map.qp_controller.control_dt, god_map.qp_controller.max_derivative)
        god_map.world.notify_state_change()
        update_world_time = time.time() - update_world_time

        collision_time = time.time()
        collisions = god_map.collision_scene.check_collisions()
        god_map.closest_point = collisions
        collision_time = time.time() - collision_time

        total_time = time.time() - total_time_start
        self.traj.set(god_map.control_cycle_counter, god_map.world.state)
        god_map.time += god_map.qp_controller.control_dt
        god_map.control_cycle_counter += 1

        return total_time, parameter_time, qp_time, update_world_time, collision_time, done

    def execute(self, sim_time: float = 5, plot: bool = True, plot_kwargs: dict = {}, plot_legend: bool = True):
        self.compile()
        while god_map.time < sim_time:
            total_time, parameter_time, qp_time, update_world_time, collision_time, done = self.step()
            if done:
                break
            # self.apply_noise(pos_noise, vel_noise, acc_noise)

            # total_times.append(total_time)
            # parameter_times.append(parameter_time)
            # qp_times.append(qp_time)
            # update_world_times.append(update_world_time)
            # collision_times.append(collision_time)
            # for goal_name in self.goal_state:
            #     next_goal, next_weight = goal_function(goal_name, god_map.time)
            #     self.update_goal(goal_name, next_goal, next_weight)
        #
        # except Exception as e:
        #     traceback.print_exc()
        #     print(e)
        #     failed = True
        #     if not catch:
        #         raise e
        # finally:
        if plot:
            self.plot_traj(plot_kwargs, plot_legend)

    def plot_traj(self, plot_kwargs: dict, plot_legend: bool = True):
        self.traj.plot_trajectory('test', sample_period=god_map.qp_controller.control_dt, filter_0_vel=True,
                                  hspace=0.7, height_per_derivative=4)
        color_map = defaultdict(lambda: self.graph_styles[len(color_map)])
        god_map.debug_expression_manager.raw_traj_to_traj(control_dt=god_map.qp_controller.control_dt).plot_trajectory(
            '',
            sample_period=god_map.qp_controller.control_dt,
            filter_0_vel=False,
            hspace=0.7,
            height_per_derivative=4,
            color_map=color_map,
            plot0_lines=False,
            legend=plot_legend,
            sort=False,
            **plot_kwargs
        )
