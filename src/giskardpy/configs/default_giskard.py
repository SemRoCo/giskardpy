from __future__ import annotations
import abc
from abc import ABC
from collections import defaultdict
from typing import Dict, Optional, List, Union, DefaultDict

import numpy as np
import rospy
import rosservice
from controller_manager_msgs.srv import ListControllers, ListControllersRequest
from numpy.typing import NDArray
from py_trees import Blackboard
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from tf2_py import LookupException

from giskardpy import identifier
from giskardpy.configs.data_types import CollisionCheckerLib, ControlModes, SupportedQPSolver, \
    CollisionAvoidanceGroupConfig, CollisionAvoidanceConfigEntry, TfPublishingModes
from giskardpy.exceptions import GiskardException
from giskardpy.goals.goal import Goal
from giskardpy.god_map import GodMap
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.joints import FixedJoint, OmniDrive, DiffDrive, Joint6DOF, OneDofJoint
from giskardpy.model.links import Link
from giskardpy.model.utils import robot_name_from_urdf_string
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string, PrefixName, Derivatives, derivative_map
from giskardpy.tree.garden import OpenLoop, ClosedLoop, StandAlone, TreeManager
from giskardpy.utils import logging
from giskardpy.utils.utils import resolve_ros_iris, get_all_classes_in_package


class Config:
    god_map = GodMap()

    @property
    def _behavior_tree_config(self) -> BehaviorTreeConfig:
        return self.god_map.get_data(identifier.giskard).behavior_tree

    @property
    def _execution_config(self) -> ExecutionConfig:
        return self.god_map.get_data(identifier.execution_config)

    @property
    def _behavior_tree(self) -> TreeManager:
        return self.god_map.get_data(identifier.tree_manager)

    @property
    def _world(self) -> WorldTree:
        return self.god_map.get_data(identifier.world)

    @property
    def _collision_scene(self) -> CollisionWorldSynchronizer:
        return self.god_map.get_data(identifier.collision_scene)

    def get_default_group_name(self):
        return list(self._world.groups.keys())[0]

    @abc.abstractmethod
    def set_defaults(self):
        ...


class ExecutionConfig(Config):
    qp_solver: SupportedQPSolver
    prediction_horizon: int = 9
    sample_period: float = 0.05
    goal_package_paths = {'giskardpy.goals'}
    control_mode: ControlModes
    max_derivative: Derivatives = Derivatives.jerk
    action_server_name: str = '~command'
    max_trajectory_length: float = 30
    qp_solver: SupportedQPSolver = None,
    retries_with_relaxed_constraints: int = 5,
    added_slack: float = 100,
    weight_factor: float = 100

    def __init__(self):
        self.set_defaults()

    def set_defaults(self):
        self.qp_solver = None
        self.prediction_horizon = 9
        self.retries_with_relaxed_constraints = 5
        self.added_slack = 100
        self.sample_period = 0.05
        self.weight_factor = 100
        self.endless_mode = False
        self.default_weights = {d: defaultdict(float) for d in Derivatives}

    def set_prediction_horizon(self, new_prediction_horizon: int):
        """
        Set the prediction horizon for the MPC. If set to 1, it will turn off acceleration and jerk limits.
        :param new_prediction_horizon: should be >= 7
        """
        if new_prediction_horizon < 7:
            raise ValueError('prediction horizon must be >= 7.')
        self.prediction_horizon = new_prediction_horizon

    def set_qp_solver(self, new_solver: SupportedQPSolver):
        self.qp_solver = new_solver

    def set_max_trajectory_length(self, length: float = 30):
        self.max_trajectory_length = length

    def set_control_mode(self, mode: ControlModes):
        self.control_mode = mode

    def add_goal_package_name(self, package_name: str):
        new_goals = get_all_classes_in_package(package_name, Goal)
        if len(new_goals) == 0:
            raise GiskardException(f'No classes of type \'{Goal.__name__}\' found in {package_name}.')
        logging.loginfo(f'Made goal classes {new_goals} available Giskard.')
        self.goal_package_paths.add(package_name)


class WorldConfig(Config):
    def __init__(self):
        self.god_map.set_data(identifier.world, WorldTree())
        self.set_default_weights()

    def set_defaults(self):
        pass

    def set_default_weights(self,
                            velocity_weight: float = 0.01,
                            acceleration_weight: float = 0,
                            jerk_weight: float = 0.01):
        """
        The default values are set automatically, even if this function is not called.
        A typical goal has a weight of 1, so the values in here should be sufficiently below that.
        """
        self._world.update_default_weights({Derivatives.velocity: velocity_weight,
                                            Derivatives.acceleration: acceleration_weight,
                                            Derivatives.jerk: jerk_weight})

    def set_weight(self, weight_map: derivative_map, joint_name: str, group_name: Optional[str] = None):
        joint_name = self._world.search_for_joint_name(joint_name, group_name)
        joint = self._world.joints[joint_name]
        if not isinstance(joint, OneDofJoint):
            raise ValueError(f'Can\'t change weight because {joint_name} is not of type {str(OneDofJoint)}.')
        free_variable = self._world.free_variables[joint.free_variable.name]
        for derivative, weight in weight_map.items():
            free_variable.quadratic_weights[derivative] = weight

    def get_root_link_of_group(self, group_name: str) -> PrefixName:
        return self._world.groups[group_name].root_link_name

    def set_joint_limits(self, limit_map: derivative_map, joint_name: my_string, group_name: Optional[str] = None):
        joint_name = self._world.search_for_joint_name(joint_name, group_name)
        joint = self._world.joints[joint_name]
        if not isinstance(joint, OneDofJoint):
            raise ValueError(f'Can\'t change limits because {joint_name} is not of type {str(OneDofJoint)}.')
        free_variable = self._world.free_variables[joint.free_variable.name]
        for derivative, limit in limit_map.items():
            free_variable.set_lower_limit(Derivatives.velocity, -limit)
            free_variable.set_upper_limit(Derivatives.velocity, limit)

    def set_default_color(self, r: float, g: float, b: float, a: float):
        """
        :param r: 0-1
        :param g: 0-1
        :param b: 0-1
        :param a: 0-1
        """
        self._world.default_link_color = ColorRGBA(r, g, b, a)

    def set_default_limits(self, new_limits: Dict[Derivatives, float]):
        """
        The default values will be set automatically, even if this function is not called.
        velocity_limit: in m/s or rad/s
        acceleration_limit: in m/s**2 or rad/s**2
        jerk_limit: in m/s**3 or rad/s**3
        """
        self._world.update_default_limits(new_limits)

    def add_robot_urdf(self,
                       urdf: str,
                       group_name: str) -> str:
        """
        Add a robot urdf to the world.
        :param urdf: robot urdf as string, not the path
        :param group_name:
        """
        if group_name is None:
            group_name = robot_name_from_urdf_string(urdf)
        self._world.add_urdf(urdf=urdf, group_name=group_name, actuated=True)
        return group_name

    def add_robot_from_parameter_server(self,
                                        parameter_name: str = 'robot_description',
                                        group_name: Optional[str] = None) -> str:
        """
        Add a robot urdf from parameter server to Giskard.
        :param parameter_name:
        :param group_name: How to call the robot. If nothing is specified it will get the name it has in the urdf
        """
        urdf = rospy.get_param(parameter_name)
        return self.add_robot_urdf(urdf=urdf, group_name=group_name)

    def add_fixed_joint(self, parent_link: my_string, child_link: my_string,
                        homogenous_transform: Optional[NDArray] = None):
        """
        Add a fixed joint to Giskard's world. Can be used to connect a non-mobile robot to the world frame.
        :param parent_link:
        :param child_link:
        :param homogenous_transform: a 4x4 transformation matrix.
        """
        if homogenous_transform is None:
            homogenous_transform = np.eye(4)
        parent_link = self._world.search_for_link_name(parent_link)

        child_link = PrefixName.from_string(child_link, set_none_if_no_slash=True)
        joint_name = PrefixName(f'{parent_link}_{child_link}_fixed_joint', None)
        joint = FixedJoint(name=joint_name, parent_link_name=parent_link, child_link_name=child_link,
                           parent_T_child=homogenous_transform)
        self._world._add_joint(joint)

    def add_diff_drive_joint(self,
                             name: str,
                             parent_link_name: my_string,
                             child_link_name: my_string,
                             robot_group_name: Optional[str] = None,
                             translation_limits: Optional[derivative_map] = None,
                             rotation_limits: Optional[derivative_map] = None):
        """
        Same as add_omni_drive_joint, but for a differential drive.
        """
        joint_name = PrefixName(name, robot_group_name)
        parent_link_name = PrefixName.from_string(parent_link_name, set_none_if_no_slash=True)
        child_link_name = PrefixName.from_string(child_link_name, set_none_if_no_slash=True)
        brumbrum_joint = DiffDrive(parent_link_name=parent_link_name,
                                   child_link_name=child_link_name,
                                   name=joint_name,
                                   translation_limits=translation_limits,
                                   rotation_limits=rotation_limits)
        self._world._add_joint(brumbrum_joint)
        self._world.deregister_group(robot_group_name)
        self._world.register_group(robot_group_name, root_link_name=parent_link_name, actuated=True)

    def add_6dof_joint(self, parent_link: my_string, child_link: my_string, joint_name: my_string):
        """
        Add a fixed joint to Giskard's world. Can be used to connect a non-mobile robot to the world frame.
        :param parent_link:
        :param child_link:
        """
        parent_link = self._world.search_for_link_name(parent_link)
        child_link = PrefixName.from_string(child_link, set_none_if_no_slash=True)
        joint_name = PrefixName.from_string(joint_name, set_none_if_no_slash=True)
        joint = Joint6DOF(name=joint_name, parent_link_name=parent_link, child_link_name=child_link)
        self._world._add_joint(joint)

    def add_empty_link(self, link_name: my_string):
        link = Link(link_name)
        self._world._add_link(link)

    def add_joint_group_position_controller(self, namespace: str, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        self.hardware_config.joint_group_position_controllers_kwargs.append({'namespace': namespace,
                                                                             'group_name': group_name})

    def add_joint_position_controller(self, namespaces: List[str], group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        self.hardware_config.joint_position_controllers_kwargs.append({'namespaces': namespaces,
                                                                       'group_name': group_name})

    def add_joint_velocity_controller(self, namespaces: List[str], group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        self.hardware_config.joint_velocity_controllers_kwargs.append({'namespaces': namespaces,
                                                                       'group_name': group_name})

    def add_omni_drive_joint(self,
                             name: str,
                             parent_link_name: Union[str, PrefixName],
                             child_link_name: Union[str, PrefixName],
                             robot_group_name: Optional[str] = None,
                             translation_limits: Optional[derivative_map] = None,
                             rotation_limits: Optional[derivative_map] = None,
                             x_name: Optional[PrefixName] = None,
                             y_name: Optional[PrefixName] = None,
                             yaw_vel_name: Optional[PrefixName] = None):
        """
        Use this to connect a robot urdf of a mobile robot to the world if it has an omni-directional drive.
        :param parent_link_name:
        :param child_link_name:
        :param robot_group_name: set if there are multiple robots
        :param name: Name of the new link. Has to be unique and may be required in other functions.
        :param translation_limits: in m/s**3
        :param rotation_limits: in rad/s**3
        """
        joint_name = PrefixName(name, robot_group_name)
        parent_link_name = PrefixName.from_string(parent_link_name, set_none_if_no_slash=True)
        child_link_name = PrefixName.from_string(child_link_name, set_none_if_no_slash=True)
        brumbrum_joint = OmniDrive(parent_link_name=parent_link_name,
                                   child_link_name=child_link_name,
                                   name=joint_name,
                                   translation_limits=translation_limits,
                                   rotation_limits=rotation_limits,
                                   x_name=x_name,
                                   y_name=y_name,
                                   yaw_name=yaw_vel_name)
        self._world._add_joint(brumbrum_joint)
        self._world.deregister_group(robot_group_name)
        self._world.register_group(robot_group_name, root_link_name=parent_link_name, actuated=True)


class RobotInterfaceConfig(Config):

    def set_defaults(self):
        pass

    def sync_odometry_topic(self, odometry_topic: str, joint_name: str):
        joint_name = self._world.search_for_joint_name(joint_name)
        self._behavior_tree.sync_odometry_topic(odometry_topic, joint_name)

    def sync_6dof_joint_with_tf_frame(self, joint_name: str, tf_parent_frame: str, tf_child_frame: str):
        """
        Tell Giskard to keep track of tf frames, e.g., for robot localization.
        """
        joint_name = self._world.search_for_joint_name(joint_name)
        self._behavior_tree.sync_6dof_joint_with_tf_frame(joint_name, tf_parent_frame, tf_child_frame)

    def sync_joint_state_topic(self, topic_name: str, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        self._behavior_tree.sync_joint_state_topic(group_name=group_name, topic_name=topic_name)

    def add_base_cmd_velocity(self,
                              cmd_vel_topic: str,
                              joint_name: my_string,
                              track_only_velocity: bool = False):
        """
        Used if the robot's base can be controlled with a Twist topic.
        :param cmd_vel_topic:
        :param track_only_velocity: The tracking mode. If true, any position error is not considered which makes
                                    the tracking smoother but less accurate.
        :param joint_name: name of the omni or diff drive joint. Doesn't need to be specified if there is only one.
        """
        joint_name = self._world.search_for_joint_name(joint_name)
        self._behavior_tree.add_base_traj_action_server(cmd_vel_topic, track_only_velocity, joint_name)

    def register_controlled_joints(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Tell Giskard which joints can be controlled. Only used in standalone mode.
        :param joint_names:
        :param group_name: Only needs to be specified, if there are more than two robots.
        """
        if self._execution_config.control_mode != ControlModes.stand_alone:
            raise GiskardException(f'Joints only need to be registered in {ControlModes.stand_alone.name} mode.')
        joint_names = [self._world.search_for_joint_name(j, group_name) for j in joint_names]
        self._world.register_controlled_joints(joint_names)

    def add_follow_joint_trajectory_server(self,
                                           namespace: str,
                                           state_topic: str,
                                           group_name: Optional[str] = None,
                                           fill_velocity_values: bool = False):
        """
        Connect Giskard to a follow joint trajectory server. It will automatically figure out which joints are offered
        and can be controlled.
        :param namespace: namespace of the action server
        :param state_topic: name of the state topic of the action server
        :param group_name: set if there are multiple robots
        :param fill_velocity_values: whether to fill the velocity entries in the message send to the robot
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        self._behavior_tree.add_follow_joint_traj_action_server(namespace=namespace, state_topic=state_topic,
                                                                group_name=group_name,
                                                                fill_velocity_values=fill_velocity_values)


    def add_joint_velocity_controller(self, namespaces: List[str]):
        self._behavior_tree.add_joint_velocity_controllers(namespaces)

class BehaviorTreeConfig(Config):
    tree_tick_rate: float = 0.05

    def set_defaults(self):
        pass

    def set_tree_tick_rate(self, rate: float = 0.05):
        self.tree_tick_rate = rate

    def add_visualization_marker_publisher(self,
                                           add_to_sync: Optional[bool] = None,
                                           add_to_planning: Optional[bool] = None,
                                           add_to_control_loop: Optional[bool] = None):
        self._behavior_tree.configure_visualization_marker(add_to_sync=add_to_sync, add_to_planning=add_to_planning,
                                                           add_to_control_loop=add_to_control_loop)

    def add_qp_data_publisher(self, publish_lb: bool = False, publish_ub: bool = False,
                              publish_lbA: bool = False, publish_ubA: bool = False,
                              publish_bE: bool = False, publish_Ax: bool = False,
                              publish_Ex: bool = False, publish_xdot: bool = False,
                              publish_weights: bool = False, publish_g: bool = False,
                              publish_debug: bool = False, add_to_base: bool = False):
        self._behavior_tree.add_qp_data_publisher(publish_lb=publish_lb,
                                                  publish_ub=publish_ub,
                                                  publish_lbA=publish_lbA,
                                                  publish_ubA=publish_ubA,
                                                  publish_bE=publish_bE,
                                                  publish_Ax=publish_Ax,
                                                  publish_Ex=publish_Ex,
                                                  publish_xdot=publish_xdot,
                                                  publish_weights=publish_weights,
                                                  publish_g=publish_g,
                                                  publish_debug=publish_debug,
                                                  add_to_base=add_to_base)

    def add_trajectory_plotter(self, normalize_position: bool = False, wait: bool = False):
        self._behavior_tree.add_plot_trajectory(normalize_position, wait)

    def add_debug_trajectory_plotter(self, normalize_position: bool = False, wait: bool = False):
        self._behavior_tree.add_plot_debug_trajectory(normalize_position=normalize_position, wait=wait)

    def add_debug_marker_publisher(self):
        self._behavior_tree.add_debug_marker_publisher()

    def add_tf_publisher(self, include_prefix: bool = True, tf_topic: str = 'tf',
                         mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects):
        self._behavior_tree.add_tf_publisher(include_prefix=include_prefix, tf_topic=tf_topic, mode=mode)


class CollisionAvoidanceConfig(Config):
    _collision_avoidance_configs: DefaultDict[str, CollisionAvoidanceGroupConfig]
    collision_checker_id: CollisionCheckerLib = None

    def __init__(self):
        self._collision_avoidance_configs = defaultdict(CollisionAvoidanceGroupConfig)

    def set_defaults(self):
        pass

    def set_collision_checker(self, new_collision_checker: CollisionCheckerLib):
        self.collision_checker_id = new_collision_checker
        collision_scene = self._create_collision_checker(self._world, new_collision_checker)
        self.god_map.set_data(identifier.collision_scene, collision_scene)

    def _create_collision_checker(self, world: WorldTree, collision_checker: CollisionCheckerLib) \
            -> CollisionWorldSynchronizer:
        if collision_checker not in CollisionCheckerLib:
            raise KeyError(f'Unknown collision checker {collision_checker}. '
                           f'Collision avoidance is disabled')
        if collision_checker == CollisionCheckerLib.bpb:
            logging.loginfo('Using betterpybullet for collision checking.')
            try:
                from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
                return BetterPyBulletSyncer(world)
            except ImportError as e:
                logging.logerr(f'{e}; turning off collision avoidance.')
                self._collision_checker = CollisionCheckerLib.none
        logging.logwarn('Using no collision checking.')
        from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
        return CollisionWorldSynchronizer(world)

    def set_default_self_collision_avoidance(self,
                                             number_of_repeller: int = 1,
                                             soft_threshold: float = 0.05,
                                             hard_threshold: float = 0.0,
                                             max_velocity: float = 0.2,
                                             group_name: Optional[str] = None):
        """
        Sets the default self collision configuration. The default of this function are set automatically.
        If they are fine, you don't need to use this function.
        :param number_of_repeller: how many constraints are added for a particular link pair
        :param soft_threshold: will try to stay out of this threshold, but can violate
        :param hard_threshold: distance threshold not allowed to be violated
        :param max_velocity: how fast it will move away from collisions
        :param group_name: name of the group this default will be applied to
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        new_default = CollisionAvoidanceConfigEntry(
            number_of_repeller=number_of_repeller,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            max_velocity=max_velocity
        )
        self._collision_avoidance_configs[group_name].self_collision_avoidance.default_factory = lambda: new_default

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
        for config in self._collision_avoidance_configs.values():
            config.external_collision_avoidance.default_factory = lambda: CollisionAvoidanceConfigEntry(
                number_of_repeller=number_of_repeller,
                soft_threshold=soft_threshold,
                hard_threshold=hard_threshold,
                max_velocity=max_velocity
            )

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
            group_name = self.get_default_group_name()
        config = self._collision_avoidance_configs[group_name]
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
            group_name = self.get_default_group_name()
        config = self._collision_avoidance_configs[group_name]
        link_name = PrefixName(link_name, group_name)
        if number_of_repeller is not None:
            config.self_collision_avoidance[link_name].number_of_repeller = number_of_repeller
        if soft_threshold is not None:
            config.self_collision_avoidance[link_name].soft_threshold = soft_threshold
        if hard_threshold is not None:
            config.self_collision_avoidance[link_name].hard_threshold = hard_threshold
        if max_velocity is not None:
            config.self_collision_avoidance[link_name].max_velocity = max_velocity

    def load_moveit_self_collision_matrix(self, path_to_srdf: str, group_name: Optional[str] = None):
        """
        Giskard only has a limited ability to compute a self collision matrix. With this function you can load one
        from Moveit.
        :param path_to_srdf: path to the srdf, can handle ros package paths
        :param group_name: name of the robot for which it will be applied, only needs to be set if there are multiple robots.
        """
        import lxml.etree as ET
        path_to_srdf = resolve_ros_iris(path_to_srdf)
        srdf = ET.parse(path_to_srdf)
        srdf_root = srdf.getroot()
        for child in srdf_root:
            if hasattr(child, 'tag') and child.tag == 'disable_collisions':
                link1 = child.attrib['link1']
                link2 = child.attrib['link2']
                reason = child.attrib['reason']
                if reason in ['Never', 'Adjacent', 'Default']:
                    self.ignore_self_collisions_of_pair(link1, link2, group_name)
        logging.loginfo(f'loaded {path_to_srdf} for self collision avoidance matrix')

    def ignore_all_self_collisions_of_link(self, link_name: str, group_name: Optional[str] = None):
        """
        Completely turn off self collision avoidance for this link.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self._collision_avoidance_configs[group_name]
        link_name = PrefixName(link_name, group_name)
        config.ignored_self_collisions.append(link_name)

    def fix_joints_for_self_collision_avoidance(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Flag some joints as fixed for self collision avoidance. These joints will not be moved to avoid self
        collisions.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self._collision_avoidance_configs[group_name]
        joint_names = [PrefixName(joint_name, group_name) for joint_name in joint_names]
        config.fixed_joints_for_self_collision_avoidance.extend(joint_names)

    def fix_joints_for_external_collision_avoidance(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Flag some joints as fixed for external collision avoidance. These joints will not be moved to avoid
        external collisions.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self._collision_avoidance_configs[group_name]
        joint_names = [PrefixName(joint_name, group_name) for joint_name in joint_names]
        config.fixed_joints_for_external_collision_avoidance.extend(joint_names)

    def ignore_self_collisions_of_pair(self, link_name1: str, link_name2: str, group_name: Optional[str] = None):
        """
        Ignore a certain pair of links for self collision avoidance.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self._collision_avoidance_configs[group_name]
        link_name1 = PrefixName(link_name1, group_name)
        link_name2 = PrefixName(link_name2, group_name)
        config.ignored_self_collisions.append((link_name1, link_name2))

    def add_self_collision(self, link_name1: str, link_name2: str, group_name: Optional[str] = None):
        """
        Specifically add a link pair for self collision avoidance.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self._collision_avoidance_configs[group_name]
        link_name1 = PrefixName(link_name1, group_name)
        link_name2 = PrefixName(link_name2, group_name)
        config.add_self_collisions.append((link_name1, link_name2))

    def ignore_all_collisions_of_links(self, link_names: List[str], group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        link_names = [PrefixName(link_name, group_name) for link_name in link_names]
        self._collision_avoidance_configs[group_name].ignored_collisions.extend(link_names)


class Giskard(ABC, Config):
    world: WorldConfig
    collision_avoidance: CollisionAvoidanceConfig
    behavior_tree: BehaviorTreeConfig
    robot_interface: RobotInterfaceConfig
    execution: ExecutionConfig
    path_to_data_folder: str = resolve_ros_iris('package://giskardpy/tmp/')

    def __init__(self):
        self._god_map = GodMap()
        self._god_map.set_data(identifier.giskard, self)
        self.world = WorldConfig()
        self.robot_interface = RobotInterfaceConfig()
        self.execution = ExecutionConfig()
        self.collision_avoidance = CollisionAvoidanceConfig()
        self.behavior_tree = BehaviorTreeConfig()
        self._god_map.set_data(identifier.hack, 0)
        blackboard = Blackboard
        blackboard.god_map = self._god_map

        self._backup = {}

    def set_defaults(self):
        self.world.set_defaults()
        self.robot_interface.set_defaults()
        self.execution.set_defaults()
        self.collision_avoidance.set_defaults()
        self.behavior_tree.set_defaults()

    @abc.abstractmethod
    def configure_world(self):
        ...

    @abc.abstractmethod
    def configure_execution(self):
        ...

    @abc.abstractmethod
    def configure_collision_avoidance(self):
        ...

    @abc.abstractmethod
    def configure_behavior_tree(self):
        ...

    @abc.abstractmethod
    def configure_robot_interface(self):
        ...

    def grow(self):
        """
        Initialize the behavior tree and world. You usually don't need to call this.
        """
        with self._world.modify_world():
            self.configure_world()
        self.configure_collision_avoidance()
        if self.collision_avoidance.collision_checker_id is None:
            self.collision_avoidance.set_collision_checker(CollisionCheckerLib.bpb)
        self.configure_execution()
        if self.execution.control_mode == ControlModes.open_loop:
            behavior_tree = OpenLoop()
        elif self.execution.control_mode == ControlModes.close_loop:
            behavior_tree = ClosedLoop()
        elif self.execution.control_mode == ControlModes.stand_alone:
            behavior_tree = StandAlone()
        else:
            raise KeyError(f'Robot interface mode \'{self.execution.control_mode}\' is not supported.')
        self.god_map.set_data(identifier.tree_manager, behavior_tree)
        self.configure_robot_interface()
        self.configure_behavior_tree()
        self._controlled_joints_sanity_check()

    def _controlled_joints_sanity_check(self):
        world = self._god_map.get_data(identifier.world)
        non_controlled_joints = set(world.movable_joint_names).difference(set(world.controlled_joints))
        if len(world.controlled_joints) == 0:
            raise GiskardException('No joints are flagged as controlled.')
        logging.loginfo(f'The following joints are non-fixed according to the urdf, '
                        f'but not flagged as controlled: {non_controlled_joints}.')
        if not self._behavior_tree.base_tracking_enabled() \
                and not self.execution.control_mode == ControlModes.stand_alone:
            logging.loginfo('No cmd_vel topic has been registered.')

    def live(self):
        """
        Start Giskard.
        """
        self.grow()
        self._god_map.get_data(identifier.tree_manager).live()
