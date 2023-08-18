from __future__ import annotations

from typing import Optional, List

from py_trees import Blackboard

from giskardpy import identifier
from giskardpy.configs.behavior_tree_config import BehaviorTreeConfig, OpenLoopBTConfig
from giskardpy.tree.control_modes import ControlModes
from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig, DisableCollisionAvoidanceConfig
from giskardpy.configs.qp_controller_config import QPControllerConfig
from giskardpy.configs.robot_interface_config import RobotInterfaceConfig
from giskardpy.configs.world_config import WorldConfig
from giskardpy.exceptions import GiskardException
from giskardpy.goals.goal import Goal
from giskardpy.god_map import GodMap
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.utils import logging
from giskardpy.utils.utils import resolve_ros_iris, get_all_classes_in_package


class Giskard(GodMapWorshipper):
    world_config: WorldConfig = None
    collision_avoidance_config: CollisionAvoidanceConfig = None
    behavior_tree_config: BehaviorTreeConfig = None
    robot_interface_config: RobotInterfaceConfig = None
    qp_controller_config: QPControllerConfig = None
    path_to_data_folder: str = resolve_ros_iris('package://giskardpy/tmp/')
    goal_package_paths = {'giskardpy.goals'}
    action_server_name: str = '~command'

    def __init__(self,
                 world_config: WorldConfig,
                 robot_interface_config: RobotInterfaceConfig,
                 collision_avoidance_config: Optional[CollisionAvoidanceConfig] = None,
                 behavior_tree_config: Optional[BehaviorTreeConfig] = None,
                 qp_controller_config: Optional[QPControllerConfig] = None,
                 additional_goal_package_paths: Optional[List[str]] = None):
        """
        :param world_config: A world configuration. Use a predefined one or implement your own WorldConfig class.
        :param robot_interface_config: How Giskard talk to the robot. You probably have to implement your own RobotInterfaceConfig.
        :param collision_avoidance_config: default is no collision avoidance or implement your own collision_avoidance_config.
        :param behavior_tree_config: default is open loop mode
        :param qp_controller_config: default is good for almost all cases
        :param additional_goal_package_paths: specify paths that Giskard needs to import to find your custom goals.
                                              Giskard will run 'from <additional path> import *' for each additional
                                              path in the list.
        """
        self.god_map.set_data(identifier.giskard, self)
        self.world_config = world_config
        self.robot_interface_config = robot_interface_config
        if collision_avoidance_config is None:
            collision_avoidance_config = DisableCollisionAvoidanceConfig()
        self.collision_avoidance_config = collision_avoidance_config
        if behavior_tree_config is None:
            behavior_tree_config = OpenLoopBTConfig()
        self.behavior_tree_config = behavior_tree_config
        if qp_controller_config is None:
            qp_controller_config = QPControllerConfig()
        self.qp_controller_config = qp_controller_config
        if additional_goal_package_paths is None:
            additional_goal_package_paths = set()
        for additional_path in additional_goal_package_paths:
            self.add_goal_package_name(additional_path)
        self.god_map.set_data(identifier.hack, 0)

    def set_defaults(self):
        self.world_config.set_defaults()
        self.robot_interface_config.set_defaults()
        self.qp_controller_config.set_defaults()
        self.collision_avoidance_config.set_defaults()
        self.behavior_tree_config.set_defaults()

    def grow(self):
        """
        Initialize the behavior tree and world. You usually don't need to call this.
        """
        with self.world.modify_world():
            self.world_config.setup()
        self.collision_avoidance_config.setup()
        self.collision_avoidance_config._sanity_check()
        self.behavior_tree_config._create_behavior_tree()
        self.behavior_tree_config.setup()
        self.robot_interface_config.setup()
        self._controlled_joints_sanity_check()
        self.world.notify_model_change()
        self.collision_scene.sync()

    def _controlled_joints_sanity_check(self):
        world = self.god_map.get_data(identifier.world)
        non_controlled_joints = set(world.movable_joint_names).difference(set(world.controlled_joints))
        if len(world.controlled_joints) == 0:
            raise GiskardException('No joints are flagged as controlled.')
        logging.loginfo(f'The following joints are non-fixed according to the urdf, '
                        f'but not flagged as controlled: {non_controlled_joints}.')
        if not self.tree_manager.base_tracking_enabled() \
                and not self.control_mode == ControlModes.standalone:
            logging.loginfo('No cmd_vel topic has been registered.')

    def add_goal_package_name(self, package_name: str):
        new_goals = get_all_classes_in_package(package_name, Goal)
        if len(new_goals) == 0:
            raise GiskardException(f'No classes of type \'{Goal.__name__}\' found in {package_name}.')
        logging.loginfo(f'Made goal classes {new_goals} available Giskard.')
        self.goal_package_paths.add(package_name)

    def live(self):
        """
        Start Giskard.
        """
        self.grow()
        self.god_map.get_data(identifier.tree_manager).live()
