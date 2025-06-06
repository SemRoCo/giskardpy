from __future__ import annotations

import abc
from copy import copy
from typing import Dict, Optional, List, Union, DefaultDict, Tuple

from giskardpy.exceptions import SetupException
from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer, CollisionAvoidanceGroupThresholds, \
    CollisionCheckerLib, CollisionAvoidanceThresholds
from giskardpy.model.world import WorldTree
from giskardpy.data_types import PrefixName
from giskardpy.utils import logging


class CollisionAvoidanceConfig(abc.ABC):
    def __init__(self, collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb):
        self._create_collision_checker(collision_checker)

    def set_defaults(self):
        pass

    @property
    def collision_scene(self) -> CollisionWorldSynchronizer:
        return god_map.collision_scene

    @property
    def collision_checker_id(self) -> CollisionCheckerLib:
        return self.collision_scene.collision_checker_id

    @property
    def world(self) -> WorldTree:
        return god_map.world

    @abc.abstractmethod
    def setup(self):
        """
        Implement this method to configure the collision avoidance using it's self. methods.
        """

    def _sanity_check(self):
        if god_map.is_collision_checking_enabled() and not self.collision_scene.has_self_collision_matrix():
            raise SetupException('You have to load a collision matrix, use: \n'
                                 'roslaunch giskardpy collision_matrix_tool.launch')

    def _create_collision_checker(self, collision_checker: CollisionCheckerLib):
        if collision_checker not in CollisionCheckerLib:
            raise KeyError(f'Unknown collision checker {collision_checker}. '
                           f'Collision avoidance is disabled')
        if collision_checker == CollisionCheckerLib.bpb:
            logging.loginfo('Using betterpybullet for collision checking.')
            try:
                from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
                god_map.collision_scene = BetterPyBulletSyncer()
                return
            except ImportError as e:
                logging.logerr(f'{e}; turning off collision avoidance.')
                self._collision_checker = CollisionCheckerLib.none
        logging.logwarn('Using no collision checking.')
        from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
        god_map.collision_scene = CollisionWorldSynchronizer()

    def set_default_self_collision_avoidance(self,
                                             number_of_repeller: int = 1,
                                             soft_threshold: float = 0.05,
                                             hard_threshold: float = 0.0,
                                             max_velocity: float = 0.2,
                                             group_name: Optional[str] = None):
        """
        Sets the default self collision configuration. The defaults of this function are set automatically.
        If they are fine, you don't need to use this function.
        :param number_of_repeller: how many constraints are added for a particular link pair
        :param soft_threshold: will try to stay out of this threshold, but can violate
        :param hard_threshold: distance threshold not allowed to be violated
        :param max_velocity: how fast it will move away from collisions
        :param group_name: name of the group this default will be applied to
        """
        if group_name is None:
            group_name = god_map.world.robot_name
        new_default = CollisionAvoidanceThresholds(
            number_of_repeller=number_of_repeller,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            max_velocity=max_velocity
        )
        self.collision_scene.collision_avoidance_configs[
            group_name].self_collision_avoidance.default_factory = lambda: new_default

    def set_default_external_collision_avoidance(self,
                                                 number_of_repeller: int = 1,
                                                 soft_threshold: float = 0.05,
                                                 hard_threshold: float = 0.0,
                                                 max_velocity: float = 0.2):
        """
        Sets the default external collision configuration. The default of this function are set automatically.
        If they are fine, you don't need to use this function.
        :param number_of_repeller: How many constraints are added for a joint to avoid collisions
        :param soft_threshold: will try to stay out of this threshold, but can violate
        :param hard_threshold: distance threshold not allowed to be violated
        :param max_velocity: how fast it will move away from collisions
        """
        new_default = CollisionAvoidanceThresholds(
            number_of_repeller=number_of_repeller,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            max_velocity=max_velocity
        )
        # overwrite the default of default
        old_default = self.collision_scene.collision_avoidance_configs.default_factory()
        old_default.external_collision_avoidance.default_factory = lambda: copy(new_default)
        self.collision_scene.collision_avoidance_configs.default_factory = lambda: copy(old_default)
        # overwrite the defaults of existing entries
        for config in self.collision_scene.collision_avoidance_configs.values():
            config.external_collision_avoidance.default_factory = lambda: copy(new_default)

    def overwrite_external_collision_avoidance(self,
                                               joint_name: str,
                                               group_name: Optional[str] = None,
                                               number_of_repeller: Optional[int] = None,
                                               soft_threshold: Optional[float] = None,
                                               hard_threshold: Optional[float] = None,
                                               max_velocity: Optional[float] = None):
        """
        :param joint_name:
        :param group_name: if there is only one robot, it will default to it
        :param number_of_repeller: How many constraints are added for a joint to avoid collisions
        :param soft_threshold: will try to stay out of this threshold, but can violate
        :param hard_threshold: distance threshold not allowed to be violated
        :param max_velocity: how fast it will move away from collisions
        """
        if group_name is None:
            group_name = god_map.world.robot_name
        config = self.collision_scene.collision_avoidance_configs[group_name]
        joint_name = PrefixName(joint_name, group_name)
        if number_of_repeller is not None:
            config.external_collision_avoidance[joint_name].number_of_repeller = number_of_repeller
        if soft_threshold is not None:
            config.external_collision_avoidance[joint_name].soft_threshold = soft_threshold
        if hard_threshold is not None:
            config.external_collision_avoidance[joint_name].hard_threshold = hard_threshold
        if max_velocity is not None:
            config.external_collision_avoidance[joint_name].max_velocity = max_velocity

    def overwrite_self_collision_avoidance(self,
                                           link_name: str,
                                           group_name: Optional[str] = None,
                                           number_of_repeller: Optional[int] = None,
                                           soft_threshold: Optional[float] = None,
                                           hard_threshold: Optional[float] = None,
                                           max_velocity: Optional[float] = None):
        """
        :param link_name:
        :param group_name: if there is only one robot, it will default to it
        :param number_of_repeller: How many constraints are added for a joint to avoid collisions
        :param soft_threshold: will try to stay out of this threshold, but can violate
        :param hard_threshold: distance threshold not allowed to be violated
        :param max_velocity: how fast it will move away from collisions
        """
        if group_name is None:
            group_name = god_map.world.robot_name
        config = self.collision_scene.collision_avoidance_configs[group_name]
        link_name = PrefixName(link_name, group_name)
        if number_of_repeller is not None:
            config.self_collision_avoidance[link_name].number_of_repeller = number_of_repeller
        if soft_threshold is not None:
            config.self_collision_avoidance[link_name].soft_threshold = soft_threshold
        if hard_threshold is not None:
            config.self_collision_avoidance[link_name].hard_threshold = hard_threshold
        if max_velocity is not None:
            config.self_collision_avoidance[link_name].max_velocity = max_velocity

    def load_self_collision_matrix(self, path_to_srdf: str, group_name: Optional[str] = None):
        """
        Load a self collision matrix. It can be created with roslaunch giskardpy collision_matrix_tool.launch.
        :param path_to_srdf: path to the srdf, can handle ros package paths
        :param group_name: name of the robot for which it will be applied, only needs to be set if there are multiple robots.
        """
        if group_name is None:
            group_name = god_map.world.robot_name
        if group_name not in self.collision_scene.self_collision_matrix_cache:
            self.collision_scene.load_self_collision_matrix_from_srdf(path_to_srdf, group_name)
        else:
            path_to_srdf, self_collision_matrix, disabled_links = self.collision_scene.self_collision_matrix_cache[
                group_name]
            self.collision_scene.self_collision_matrix = self_collision_matrix
            self.collision_scene.disabled_links = disabled_links

    def fix_joints_for_collision_avoidance(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Flag some joints as fixed for collision avoidance. These joints will not be moved to avoid self
        collisions.
        """
        if group_name is None:
            group_name = god_map.world.robot_name
        for joint_name in joint_names:
            world_joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            self.collision_scene.add_fixed_joint(world_joint_name)


class LoadSelfCollisionMatrixConfig(CollisionAvoidanceConfig):
    def __init__(self, path_to_self_collision_matrix: str,
                 collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb):
        super().__init__(collision_checker)
        self._path_to_self_collision_matrix = path_to_self_collision_matrix

    def setup(self):
        self.load_self_collision_matrix(self._path_to_self_collision_matrix)


class _BPBCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self, collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb):
        super().__init__(collision_checker)

    def setup(self):
        pass

    def _sanity_check(self):
        pass


class DisableCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self):
        super().__init__(CollisionCheckerLib.none)

    def setup(self):
        pass
