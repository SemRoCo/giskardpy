from collections import defaultdict
from typing import Optional, List, Dict, Tuple

from giskardpy.data_types.exceptions import SetupException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.collision_avoidance_config import CollisionAvoidanceConfig, DisableCollisionAvoidanceConfig
from giskardpy.model.collision_world_syncer import CollisionEntry
from giskardpy.model.world_config import WorldConfig
from giskardpy.motion_statechart.goals.collision_avoidance import CollisionAvoidance
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.monitors.monitors import Monitor
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.task import Task
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.utils.utils import get_all_classes_in_package



class MotionStatechartNodeWrapper:
    _motion_graph_nodes: Dict[str, MotionStatechartNode]
    _name_prefix = ''
    giskard_wrapper: GiskardWrapper

    def __init__(self, giskard_wrapper: GiskardWrapper):
        self.giskard_wrapper = giskard_wrapper
        self.reset()

    @property
    def robot_name(self) -> str:
        return self.giskard_wrapper.robot_name

    @property
    def motion_graph_nodes(self) -> Dict[str, MotionStatechartNode]:
        return self._motion_graph_nodes

    def reset(self):
        self._motion_graph_nodes = ImmutableDict()

    def _add_motion_statechart_node(self, *,
                                    class_name: str,
                                    name: Optional[str] = None,
                                    start_condition: str = '',
                                    pause_condition: str = '',
                                    end_condition: str = '',
                                    reset_condition: str = '',
                                    **kwargs) -> str:
        """
        Generic function to add a motion goal.
        :param motion_goal_class: Name of a class defined in src/giskardpy/goals
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param pause_condition: a logical expression. Goal will be on hold if it is True and active otherwise
        :param end_condition: a logical expression. Goal will become inactive when this becomes True.
        :param kwargs: kwargs for __init__ function of motion_goal_class
        """
        if name is None:
            name = f'{self._name_prefix}{len(self._motion_graph_nodes)} [{class_name}]'
        motion_goal = MotionStatechartNode()
        motion_goal.name = name
        motion_goal.class_name = class_name
        self._motion_graph_nodes[name] = motion_goal
        motion_goal.kwargs = kwargs_to_json(kwargs)

        self.update_start_condition(node_name=name, condition=start_condition)
        self.update_pause_condition(node_name=name, condition=pause_condition)
        if end_condition is None:  # everything ends themselves by default
            motion_goal.end_condition = name
            self.update_end_condition(node_name=name, condition=name)
        else:
            self.update_end_condition(node_name=name, condition=end_condition)
        self.update_reset_condition(node_name=name, condition=reset_condition)
        return name

    def get_anded_nodes(self, add_nodes_without_end_condition: bool = True) -> str:
        nodes = []
        for node in self.motion_graph_nodes.values():
            if (node.class_name not in get_all_classes_in_package('giskardpy.motion_statechart.monitors',
                                                                  CancelMotion)
                    and (add_nodes_without_end_condition or node.end_condition != '')):
                nodes.append(node.name)
        return ' and '.join(nodes)

    def set_conditions(self, node_name: str,
                       start_condition: str,
                       pause_condition: str,
                       end_condition: str,
                       reset_condition: str):
        self.update_start_condition(node_name, start_condition)
        self.update_pause_condition(node_name, pause_condition)
        self.update_end_condition(node_name, end_condition)
        self.update_reset_condition(node_name, reset_condition)

    def update_start_condition(self, node_name: str, condition: str) -> None:
        self._motion_graph_nodes[node_name].start_condition = condition

    def update_reset_condition(self, node_name: str, condition: str) -> None:
        self._motion_graph_nodes[node_name].reset_condition = condition

    def update_pause_condition(self, node_name: str, condition: str) -> None:
        self._motion_graph_nodes[node_name].pause_condition = condition

    def update_end_condition(self, node_name: str, condition: str) -> None:
        self._motion_graph_nodes[node_name].end_condition = condition



class MotionGoalWrapper(MotionStatechartNodeWrapper):
    _name_prefix = 'G'
    _collision_entries: Dict[Tuple[str, str, str], List[CollisionEntry]]

    def reset(self):
        super().reset()
        self._collision_entries = defaultdict(list)

    def add_motion_goal(self, *,
                        class_name: str,
                        start_condition: str = '',
                        pause_condition: str = '',
                        name: Optional[str] = None,
                        end_condition: str = '',
                        **kwargs) -> str:
        """
        Generic function to add a motion goal.
        :param class_name: Name of a class defined in src/giskardpy/goals
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param pause_condition: a logical expression. Goal will be on hold if it is True and active otherwise
        :param end_condition: a logical expression. Goal will become inactive when this becomes True.
        :param kwargs: kwargs for __init__ function of motion_goal_class
        """
        return super()._add_motion_statechart_node(class_name=class_name,
                                                   name=name,
                                                   start_condition=start_condition,
                                                   pause_condition=pause_condition,
                                                   end_condition=end_condition,
                                                   **kwargs)

    def add_joint_position(self,
                           goal_state: Dict[str, float],
                           name: Optional[str] = None,
                           weight: Optional[float] = None,
                           max_velocity: Optional[float] = None,
                           start_condition: str = '',
                           pause_condition: str = '',
                           end_condition: str = '') -> str:
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param weight: None = use default weight
        :param max_velocity: will be applied to all joints
        """
        return self.add_motion_goal(class_name=JointPositionList.__name__,
                                    goal_state=goal_state,
                                    weight=weight,
                                    max_velocity=max_velocity,
                                    name=name,
                                    start_condition=start_condition,
                                    pause_condition=pause_condition,
                                    end_condition=end_condition)

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
        god_map.tmp_folder = 'tmp'
        self.world_config = world_config
        if collision_avoidance_config is None:
            collision_avoidance_config = DisableCollisionAvoidanceConfig()
        self.collision_avoidance_config = collision_avoidance_config
        if qp_controller_config is None:
            qp_controller_config = QPControllerConfig()
        self.qp_controller_config = qp_controller_config
        if additional_goal_package_paths is None:
            additional_goal_package_paths = {'giskardpy_ros.goals'}
        for additional_path in additional_goal_package_paths:
            self.add_goal_package_name(additional_path)
        if additional_monitor_package_paths is None:
            additional_monitor_package_paths = set()
        for additional_path in additional_monitor_package_paths:
            self.add_monitor_package_name(additional_path)
        god_map.hack = 0

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

    def execute(self):
        pass

