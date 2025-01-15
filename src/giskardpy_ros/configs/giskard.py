from __future__ import annotations

from typing import Optional, List

from giskardpy.model.world_config import WorldConfig
from giskardpy.motion_statechart.tasks.task import Task
from giskardpy_ros.configs.behavior_tree_config import BehaviorTreeConfig, OpenLoopBTConfig
from giskardpy.god_map import god_map
from giskardpy.model.collision_avoidance_config import CollisionAvoidanceConfig, DisableCollisionAvoidanceConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy_ros.configs.robot_interface_config import RobotInterfaceConfig
from giskardpy.data_types.exceptions import GiskardException, SetupException
from giskardpy.goals.goal import Goal
from giskardpy.motion_statechart.monitors.monitors import Monitor
from giskardpy.middleware import get_middleware, get_middleware
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy.utils.utils import get_all_classes_in_package


class Giskard:
    world_config: WorldConfig = None
    collision_avoidance_config: CollisionAvoidanceConfig = None
    behavior_tree_config: BehaviorTreeConfig = None
    robot_interface_config: RobotInterfaceConfig = None
    qp_controller_config: QPControllerConfig = None
    action_server_name: str = '~command'

    def __init__(self,
                 world_config: WorldConfig,
                 robot_interface_config: RobotInterfaceConfig,
                 collision_avoidance_config: Optional[CollisionAvoidanceConfig] = None,
                 behavior_tree_config: Optional[BehaviorTreeConfig] = None,
                 qp_controller_config: Optional[QPControllerConfig] = None,
                 additional_goal_package_paths: Optional[List[str]] = None,
                 additional_monitor_package_paths: Optional[List[str]] = None):
        """
        The main Class of Giskard.
        Instantiate it with appropriate configs for you setup and then call giskard.live()
        :param world_config: A world configuration. Use a predefined one or implement your own WorldConfig class.
        :param robot_interface_config: How Giskard talk to the robot. You probably have to implement your own RobotInterfaceConfig.
        :param collision_avoidance_config: default is no collision avoidance or implement your own collision_avoidance_config.
        :param behavior_tree_config: default is open loop mode
        :param qp_controller_config: default is good for almost all cases
        :param additional_goal_package_paths: specify paths that Giskard needs to import to find your custom Goals.
                                              Giskard will run 'from <additional path> import *' for each additional
                                              path in the list.
        :param additional_monitor_package_paths: specify paths that Giskard needs to import to find your custom Monitors.
                                              Giskard will run 'from <additional path> import *' for each additional
                                              path in the list.
        """
        god_map.tmp_folder = get_middleware().resolve_iri('package://giskardpy_ros/tmp/')
        GiskardBlackboard().giskard = self
        self.world_config = world_config
        self.robot_interface_config = robot_interface_config
        if collision_avoidance_config is None:
            collision_avoidance_config = DisableCollisionAvoidanceConfig()
        self.collision_avoidance_config = collision_avoidance_config
        if behavior_tree_config is None:
            behavior_tree_config = OpenLoopBTConfig()
        self.behavior_tree_config = behavior_tree_config
        # god_map.behavior_tree_config = behavior_tree_config
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

    def set_defaults(self) -> None:
        self.qp_controller_config.set_defaults()
        self.world_config.set_defaults()
        self.robot_interface_config.set_defaults()
        self.collision_avoidance_config.set_defaults()
        self.behavior_tree_config.set_defaults()

    def grow(self):
        """
        Initialize the behavior tree and world. You usually don't need to call this.
        """
        with god_map.world.modify_world():
            self.world_config.setup()
        self.behavior_tree_config._create_behavior_tree()
        self.behavior_tree_config.setup()
        self.robot_interface_config.setup()
        god_map.world._notify_model_change()
        self.collision_avoidance_config.setup()
        self.collision_avoidance_config._sanity_check()
        god_map.collision_scene.sync()
        self.sanity_check()
        GiskardBlackboard().tree.setup(30)

    def sanity_check(self):
        self._controlled_joints_sanity_check()

    def _controlled_joints_sanity_check(self):
        world = god_map.world
        non_controlled_joints = set(world.movable_joint_names).difference(set(world.controlled_joints))
        if len(world.controlled_joints) == 0 and len(world.joints) > 0:
            raise SetupException('No joints are flagged as controlled.')
        get_middleware().loginfo(f'The following joints are non-fixed according to the urdf, '
                        f'but not flagged as controlled: {non_controlled_joints}.')

    def add_goal_package_name(self, package_name: str):
        new_goals = get_all_classes_in_package(package_name, Goal)
        if len(new_goals) == 0:
            raise SetupException(f'No classes of type \'{Goal.__name__}\' found in {package_name}.')
        get_middleware().loginfo(f'Made goal classes {new_goals} available.')
        god_map.motion_graph_manager.add_goal_package_path(package_name)

    def add_task_package_name(self, package_name: str):
        new_goals = get_all_classes_in_package(package_name, Task)
        if len(new_goals) == 0:
            raise SetupException(f'No classes of type \'{Goal.__name__}\' found in {package_name}.')
        get_middleware().loginfo(f'Made task classes {new_goals} available.')
        god_map.motion_graph_manager.add_task_package_path(package_name)

    def add_monitor_package_name(self, package_name: str) -> None:
        new_monitors = get_all_classes_in_package(package_name, Monitor)
        if len(new_monitors) == 0:
            raise SetupException(f'No classes of type \'{Monitor.__name__}\' found in \'{package_name}\'.')
        get_middleware().loginfo(f'Made Monitor classes \'{new_monitors}\' available.')
        god_map.motion_graph_manager.add_monitor_package_path(package_name)

    def live(self):
        """
        Start Giskard.
        """
        self.grow()
        GiskardBlackboard().tree.live()
