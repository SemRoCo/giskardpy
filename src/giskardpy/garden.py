from collections import defaultdict

import rospy
from py_trees import Blackboard

import giskardpy.identifier as identifier
from giskardpy.data_types import order_map, KeyDefaultDict
from giskardpy.god_map import GodMap
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.world import WorldTree
from giskardpy.tree.closed_loop_tree import ClosedLoopTree
from giskardpy.tree.open_loop_tree import OpenLoopTree
from giskardpy.utils import logging
from giskardpy.utils.config_loader import ros_load_robot_config
from giskardpy.utils.math import max_velocity_from_horizon_and_jerk


def upload_config_file_to_paramserver():
    old_params = rospy.get_param('~')
    if rospy.has_param('~test'):
        test = rospy.get_param('~test')
    else:
        test = False
    config_file_name = rospy.get_param('~{}'.format('config'))
    ros_load_robot_config(config_file_name, old_data=old_params, test=test)


def initialize_god_map():
    upload_config_file_to_paramserver()
    god_map = GodMap.init_from_paramserver(rospy.get_name())
    blackboard = Blackboard
    blackboard.god_map = god_map

    world = WorldTree(god_map)
    world.delete_all_but_robot()

    collision_checker = god_map.get_data(identifier.collision_checker)
    if collision_checker == 'bpb':
        logging.loginfo('Using bpb for collision checking.')
        from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
        collision_scene = BetterPyBulletSyncer(world)
    elif collision_checker == 'pybullet':
        logging.loginfo('Using pybullet for collision checking.')
        from giskardpy.model.pybullet_syncer import PyBulletSyncer
        collision_scene = PyBulletSyncer(world)
    else:
        logging.logwarn('Unknown collision checker {}. Collision avoidance is disabled'.format(collision_checker))
        collision_scene = CollisionWorldSynchronizer(world)
        god_map.set_data(identifier.collision_checker, None)
    god_map.set_data(identifier.collision_scene, collision_scene)

    # sanity_check_derivatives(god_map)
    # sanity_check(god_map)
    return god_map


def sanity_check(god_map):
    check_velocity_limits_reachable(god_map)


def sanity_check_derivatives(god_map):
    weights = god_map.get_data(identifier.joint_weights)
    limits = god_map.get_data(identifier.joint_limits)
    check_derivatives(weights, 'Weights')
    check_derivatives(limits, 'Limits')
    if len(weights) != len(limits):
        raise AttributeError('Weights and limits are not defined for the same number of derivatives')


def check_derivatives(entries, name):
    """
    :type entries: dict
    """
    allowed_derivates = list(order_map.values())[1:]
    for weight in entries:
        if weight not in allowed_derivates:
            raise AttributeError(
                '{} set for unknown derivative: {} not in {}'.format(name, weight, list(allowed_derivates)))
    weight_ids = [order_map.inverse[x] for x in entries]
    if max(weight_ids) != len(weight_ids):
        raise AttributeError(
            '{} for {} set, but some of the previous derivatives are missing'.format(name, order_map[max(weight_ids)]))


def check_velocity_limits_reachable(god_map):
    # TODO a more general version of this
    robot = god_map.get_data(identifier.robot)
    sample_period = god_map.get_data(identifier.sample_period)
    prediction_horizon = god_map.get_data(identifier.prediction_horizon)
    print_help = False
    for joint_name in robot.get_joint_names():
        velocity_limit = robot.get_joint_limit_expr_evaluated(joint_name, 1, god_map)
        jerk_limit = robot.get_joint_limit_expr_evaluated(joint_name, 3, god_map)
        velocity_limit_horizon = max_velocity_from_horizon_and_jerk(prediction_horizon, jerk_limit, sample_period)
        if velocity_limit_horizon < velocity_limit:
            logging.logwarn('Joint \'{}\' '
                            'can reach at most \'{:.4}\' '
                            'with to prediction horizon of \'{}\' '
                            'and jerk limit of \'{}\', '
                            'but limit in urdf/config is \'{}\''.format(joint_name,
                                                                        velocity_limit_horizon,
                                                                        prediction_horizon,
                                                                        jerk_limit,
                                                                        velocity_limit
                                                                        ))
            print_help = True
    if print_help:
        logging.logwarn('Check utils.py/max_velocity_from_horizon_and_jerk for help.')


def process_joint_specific_params(identifier_, default, override, god_map):
    default_value = god_map.unsafe_get_data(default)
    d = defaultdict(lambda: default_value)
    override = god_map.get_data(override)
    if isinstance(override, dict):
        d.update(override)
    god_map.set_data(identifier_, d)
    return KeyDefaultDict(lambda key: god_map.to_symbol(identifier_ + [key]))


def let_there_be_motions():
    god_map = initialize_god_map()
    mode = god_map.get_data(identifier.robot_interface_mode)
    del god_map.get_data(identifier.robot_interface)['mode']
    if mode == 'open_loop':
        tree = OpenLoopTree(god_map)
    elif mode == 'closed_loop':
        tree = ClosedLoopTree(god_map)
    else:
        raise KeyError('Robot interface mode \'{}\' is not supported.'.format(mode))

    return tree
