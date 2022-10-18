import keyword
from collections import defaultdict
from multiprocessing import Queue
from time import time
from typing import Tuple, Optional

import actionlib
import control_msgs
import hypothesis.strategies as st
import numpy as np
import rospy
from angles import shortest_angular_distance
from control_msgs.msg import FollowJointTrajectoryActionGoal
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Vector3Stamped, PointStamped, QuaternionStamped
from hypothesis import assume
from hypothesis.strategies import composite
from numpy import pi
from rospy import Timer
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from tf.transformations import rotation_from_matrix, quaternion_matrix
from tf2_py import LookupException
from visualization_msgs.msg import Marker

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import CollisionEntry, MoveResult, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, DyeGroupResponse
from giskardpy import identifier, RobotPrefix
from giskardpy.configs.boxy import Boxy
from giskardpy.configs.data_types import CollisionAvoidanceConfig
from giskardpy.configs.default_config import ControlModes
from giskardpy.data_types import KeyDefaultDict, JointStates, PrefixName
from giskardpy.exceptions import UnknownGroupException
from giskardpy.goals.goal import WEIGHT_ABOVE_CA
from giskardpy.god_map import GodMap
from giskardpy.model.joints import OneDofJoint
from giskardpy.model.world import WorldTree, SubWorldTree
from giskardpy.my_types import goal_parameter
from giskardpy.python_interface import GiskardWrapper
from giskardpy.utils import logging, utils
from giskardpy.utils.math import compare_poses
from giskardpy.utils.utils import msg_to_list, position_dict_to_joint_states
from iai_naive_kinematics_sim.srv import SetJointState, SetJointStateRequest, UpdateTransform, UpdateTransformRequest

BIG_NUMBER = 1e100
SMALL_NUMBER = 1e-100

folder_name = 'tmp_data/'


def vector(x):
    return st.lists(float_no_nan_no_inf(), min_size=x, max_size=x)


update_world_error_codes = {value: name for name, value in vars(UpdateWorldResponse).items() if
                            isinstance(value, int) and name[0].isupper()}


def update_world_error_code(code):
    return update_world_error_codes[code]


move_result_error_codes = {value: name for name, value in vars(MoveResult).items() if
                           isinstance(value, int) and name[0].isupper()}


def move_result_error_code(code):
    return move_result_error_codes[code]


def robot_urdfs():
    return st.sampled_from(['urdfs/pr2.urdfs', 'urdfs/boxy.urdfs', 'urdfs/iai_donbot.urdfs'])


def angle_positive():
    return st.floats(0, 2 * np.pi)


def angle():
    return st.floats(-np.pi, np.pi)


def keys_values(max_length=10, value_type=st.floats(allow_nan=False)):
    return lists_of_same_length([variable_name(), value_type], max_length=max_length, unique=True)


def compare_axis_angle(actual_angle, actual_axis, expected_angle, expected_axis, decimal=3):
    try:
        np.testing.assert_array_almost_equal(actual_axis, expected_axis, decimal=decimal)
        np.testing.assert_almost_equal(shortest_angular_distance(actual_angle, expected_angle), 0, decimal=decimal)
    except AssertionError:
        try:
            np.testing.assert_array_almost_equal(actual_axis, -expected_axis, decimal=decimal)
            np.testing.assert_almost_equal(shortest_angular_distance(actual_angle, abs(expected_angle - 2 * pi)), 0,
                                           decimal=decimal)
        except AssertionError:
            np.testing.assert_almost_equal(shortest_angular_distance(actual_angle, 0), 0, decimal=decimal)
            np.testing.assert_almost_equal(shortest_angular_distance(0, expected_angle), 0, decimal=decimal)
            assert not np.any(np.isnan(actual_axis))
            assert not np.any(np.isnan(expected_axis))


@composite
def variable_name(draw):
    variable = draw(st.text('qwertyuiopasdfghjklzxcvbnm', min_size=1))
    assume(variable not in keyword.kwlist)
    return variable


@composite
def lists_of_same_length(draw, data_types=(), min_length=1, max_length=10, unique=False):
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    lists = []
    for elements in data_types:
        lists.append(draw(st.lists(elements, min_size=length, max_size=length, unique=unique)))
    return lists


@composite
def rnd_joint_state(draw, joint_limits):
    return {jn: draw(st.floats(ll, ul, allow_nan=False, allow_infinity=False)) for jn, (ll, ul) in joint_limits.items()}


@composite
def rnd_joint_state2(draw, joint_limits):
    muh = draw(joint_limits)
    muh = {jn: ((ll if ll is not None else pi * 2), (ul if ul is not None else pi * 2))
           for (jn, (ll, ul)) in muh.items()}
    return {jn: draw(st.floats(ll, ul, allow_nan=False, allow_infinity=False)) for jn, (ll, ul) in muh.items()}


@composite
def pr2_joint_state(draw):
    pass


def pr2_urdf():
    with open('urdfs/pr2_with_base.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def pr2_without_base_urdf():
    with open('urdfs/pr2.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def base_bot_urdf():
    with open('urdfs/2d_base_bot.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def donbot_urdf():
    with open('urdfs/iai_donbot.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def boxy_urdf():
    with open('urdfs/boxy.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def hsr_urdf():
    with open('urdfs/hsr.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def float_no_nan_no_inf(outer_limit=None, min_dist_to_zero=None):
    if outer_limit is not None:
        return st.floats(allow_nan=False, allow_infinity=False, max_value=outer_limit, min_value=-outer_limit)
    else:
        return st.floats(allow_nan=False, allow_infinity=False)
    # f = st.floats(allow_nan=False, allow_infinity=False, max_value=outer_limit, min_value=-outer_limit)
    # # f = st.floats(allow_nan=False, allow_infinity=False)
    # if min_dist_to_zero is not None:
    #     f = f.filter(lambda x: (outer_limit > abs(x) and abs(x) > min_dist_to_zero) or x == 0)
    # else:
    #     f = f.filter(lambda x: abs(x) < outer_limit)
    # return f


@composite
def sq_matrix(draw):
    i = draw(st.integers(min_value=1, max_value=10))
    i_sq = i ** 2
    l = draw(st.lists(float_no_nan_no_inf(outer_limit=1000), min_size=i_sq, max_size=i_sq))
    return np.array(l).reshape((i, i))


def unit_vector(length, elements=None):
    if elements is None:
        elements = float_no_nan_no_inf(min_dist_to_zero=1e-10)
    vector = st.lists(elements,
                      min_size=length,
                      max_size=length).filter(lambda x: SMALL_NUMBER < np.linalg.norm(x) < BIG_NUMBER)

    def normalize(v):
        v = [round(x, 4) for x in v]
        l = np.linalg.norm(v)
        if l == 0:
            return np.array([0] * (length - 1) + [1])
        return np.array([x / l for x in v])

    return st.builds(normalize, vector)


def quaternion(elements=None):
    return unit_vector(4, elements)


def pykdl_frame_to_numpy(pykdl_frame):
    return np.array([[pykdl_frame.M[0, 0], pykdl_frame.M[0, 1], pykdl_frame.M[0, 2], pykdl_frame.p[0]],
                     [pykdl_frame.M[1, 0], pykdl_frame.M[1, 1], pykdl_frame.M[1, 2], pykdl_frame.p[1]],
                     [pykdl_frame.M[2, 0], pykdl_frame.M[2, 1], pykdl_frame.M[2, 2], pykdl_frame.p[2]],
                     [0, 0, 0, 1]])


class GoalChecker(object):
    def __init__(self, god_map: GodMap):
        self.god_map = god_map
        self.world: WorldTree = self.god_map.unsafe_get_data(identifier.world)
        self.robot: SubWorldTree = self.world.groups[self.god_map.unsafe_get_data(identifier.robot_group_name)]

    def transform_msg(self, target_frame, msg, timeout=1):
        try:
            return tf.transform_msg(target_frame, msg, timeout=timeout)
        except LookupException as e:
            return self.world.transform_msg(target_frame, msg)


class JointGoalChecker(GoalChecker):
    def __init__(self, god_map, goal_state, decimal=2):
        super().__init__(god_map)
        self.goal_state = goal_state
        self.decimal = decimal

    def get_current_joint_state(self) -> JointStates:
        try:
            current_joint_state = rospy.wait_for_message('joint_states', JointState, timeout=rospy.Duration(0.5))
            return JointStates.from_msg(current_joint_state)
        except:
            return self.world.state

    def __call__(self):
        current_joint_state = self.get_current_joint_state()
        self.compare_joint_state(current_joint_state, self.goal_state, decimal=self.decimal)

    def compare_joint_state(self, current_js, goal_js, decimal=2):
        """
        :type current_js: dict
        :type goal_js: dict
        :type decimal: int
        """
        for joint_name in goal_js:
            goal = goal_js[joint_name]
            current = current_js[PrefixName(joint_name, None)].position
            if self.world.is_joint_continuous(PrefixName(joint_name, RobotPrefix)):
                np.testing.assert_almost_equal(shortest_angular_distance(goal, current), 0, decimal=decimal,
                                               err_msg=f'{joint_name}: actual: {current} desired: {goal}')
            else:
                np.testing.assert_almost_equal(current, goal, decimal,
                                               err_msg=f'{joint_name}: actual: {current} desired: {goal}')


class TranslationGoalChecker(GoalChecker):
    def __init__(self, god_map, tip_link, root_link, expected):
        super().__init__(god_map)
        self.expected = expected
        self.tip_link = tip_link
        self.root_link = root_link
        self.expected = self.transform_msg(self.root_link, self.expected)

    def __call__(self):
        expected = self.expected
        current_pose = self.world.compute_fk_pose(self.root_link, self.tip_link)
        np.testing.assert_array_almost_equal(msg_to_list(expected.point),
                                             msg_to_list(current_pose.pose.position), decimal=2)


class AlignPlanesGoalChecker(GoalChecker):
    def __init__(self, god_map, tip_link, tip_normal, root_link, root_normal):
        super(AlignPlanesGoalChecker, self).__init__(god_map)
        self.tip_normal = tip_normal
        self.tip_link = tip_link
        self.root_link = root_link
        self.expected = self.transform_msg(self.root_link, root_normal)

    def __call__(self):
        expected = self.expected
        current = self.transform_msg(self.root_link, self.tip_normal)
        np.testing.assert_array_almost_equal(msg_to_list(expected.vector), msg_to_list(current.vector), decimal=2)


class PointingGoalChecker(GoalChecker):
    def __init__(self, god_map, tip_link, goal_point, root_link=None, pointing_axis=None):
        super(PointingGoalChecker, self).__init__(god_map)
        self.tip_link = tip_link
        self.root_link = root_link
        if pointing_axis:
            self.tip_V_pointer = self.transform_msg(self.tip_link, pointing_axis)
        else:
            self.tip_V_pointer = Vector3Stamped()
            self.tip_V_pointer.vector.z = 1
            self.tip_V_pointer.header.frame_id = tip_link
        self.tip_V_pointer = tf.msg_to_homogeneous_matrix(self.tip_V_pointer)
        self.goal_point = self.transform_msg(self.root_link, goal_point)
        self.goal_point.header.stamp = rospy.Time()

    def __call__(self):
        tip_P_goal = tf.msg_to_homogeneous_matrix(self.transform_msg(self.tip_link, self.goal_point))
        tip_P_goal[-1] = 0
        tip_P_goal = tip_P_goal / np.linalg.norm(tip_P_goal)
        np.testing.assert_array_almost_equal(tip_P_goal, self.tip_V_pointer, decimal=2)


class RotationGoalChecker(GoalChecker):
    def __init__(self, god_map, tip_link, root_link, expected):
        super().__init__(god_map)
        self.expected = expected
        self.tip_link = tip_link
        self.root_link = root_link
        self.expected = self.transform_msg(self.root_link, self.expected)

    def __call__(self):
        expected = self.expected
        current_pose = self.world.compute_fk_pose(self.root_link, self.tip_link)

        try:
            np.testing.assert_array_almost_equal(msg_to_list(expected.quaternion),
                                                 msg_to_list(current_pose.pose.orientation), decimal=2)
        except AssertionError:
            np.testing.assert_array_almost_equal(msg_to_list(expected.quaternion),
                                                 -np.array(msg_to_list(current_pose.pose.orientation)), decimal=2)


class GiskardTestWrapper(GiskardWrapper):
    god_map: GodMap
    default_pose = {}
    better_pose = {}
    odom_root = 'odom'

    def __init__(self, config_file):
        self.total_time_spend_giskarding = 0
        self.total_time_spend_moving = 0

        self.set_localization_srv = rospy.ServiceProxy('/map_odom_transform_publisher/update_map_odom_transform',
                                                       UpdateTransform)

        self.giskard = config_file()
        self.giskard.grow()
        self.tree = self.giskard._tree
        # self.tree = TreeManager.from_param_server()
        self.god_map = self.tree.god_map
        self.tick_rate = self.god_map.unsafe_get_data(identifier.tree_tick_rate)
        self.heart = Timer(rospy.Duration(self.tick_rate), self.heart_beat)
        super().__init__(node_name='tests')
        self.results = Queue(100)
        self.default_root = str(self.world.root_link_name)
        self.goal_checks = defaultdict(list)

        def create_publisher(topic):
            p = rospy.Publisher(topic, JointState, queue_size=10)
            rospy.sleep(.2)
            return p

        self.joint_state_publisher = KeyDefaultDict(create_publisher)
        # rospy.sleep(1)
        self.original_number_of_links = len(self.world.links)

    def set_seed_odometry(self, base_pose):
        self.set_json_goal('SetOdometry',
                           group_name=self.get_robot_name(),
                           base_pose=base_pose)

    def set_seed_configuration(self, seed_configuration):
        self.set_json_goal('SetSeedConfiguration',
                           seed_configuration=seed_configuration)

    def set_localization(self, map_T_odom: PoseStamped):
        map_T_odom.pose.position.z = 0
        req = UpdateTransformRequest()
        req.transform.translation = map_T_odom.pose.position
        req.transform.rotation = map_T_odom.pose.orientation
        assert self.set_localization_srv(req).success
        self.wait_heartbeats(15)
        p2 = self.world.compute_fk_pose(self.world.root_link_name, self.odom_root)
        compare_poses(p2.pose, map_T_odom.pose)

    @profile
    def transform_msg(self, target_frame, msg, timeout=1):
        try:
            return tf.transform_msg(target_frame, msg, timeout=timeout)
        except LookupException as e:
            return self.world.transform_msg(target_frame, msg)

    def wait_heartbeats(self, number=2):
        behavior_tree = self.tree.tree
        c = behavior_tree.count
        while behavior_tree.count < c + number:
            rospy.sleep(0.001)

    @property
    def collision_scene(self):
        """
        :rtype: giskardpy.model.collision_world_syncer.CollisionWorldSynchronizer
        """
        return self.god_map.unsafe_get_data(identifier.collision_scene)

    @property
    def robot(self):
        """
        :rtype: giskardpy.model.world.SubWorldTree
        """
        return self.world.groups[self.god_map.unsafe_get_data(identifier.robot_group_name)]

    def dye_group(self, group_name: str, rgba: Tuple[float, float, float, float],
                  expected_error_codes=(DyeGroupResponse.SUCCESS,)):
        res = super().dye_group(group_name, rgba)
        assert res.error_codes in expected_error_codes

    def heart_beat(self, timer_thing):
        self.tree.tick()

    def tear_down(self):
        self.god_map.unsafe_get_data(identifier.timer_collector).print()
        rospy.sleep(1)
        self.heart.shutdown()
        # TODO it is strange that I need to kill the services... should be investigated. (:
        self.tree.kill_all_services()
        giskarding_time = self.total_time_spend_giskarding
        if self.god_map.get_data(identifier.control_mode) != ControlModes.stand_alone:
            giskarding_time -= self.total_time_spend_moving
        logging.loginfo(f'total time spend giskarding: {giskarding_time}')
        logging.loginfo(f'total time spend moving: {self.total_time_spend_moving}')
        logging.loginfo('stopping tree')

    def set_object_joint_state(self, object_name, joint_state):
        super().set_object_joint_state(object_name, joint_state)
        self.wait_heartbeats(2)
        current_js = self.world.groups[object_name].state
        joint_names_without_prefix = set(j.short_name for j in current_js)
        assert set(joint_state.keys()).difference(joint_names_without_prefix) == set()
        for joint_name, state in current_js.items():
            if joint_name.short_name in joint_state:
                np.testing.assert_almost_equal(state.position, joint_state[joint_name.short_name], 2)

    def set_kitchen_js(self, joint_state, object_name='kitchen'):
        self.set_object_joint_state(object_name, joint_state)

    def set_apartment_js(self, joint_state, object_name='apartment'):
        self.set_object_joint_state(object_name, joint_state)

    def compare_joint_state(self, current_js, goal_js, decimal=2):
        """
        :type current_js: dict
        :type goal_js: dict
        :type decimal: int
        """
        for joint_name in goal_js:
            goal = goal_js[joint_name]
            current = current_js[joint_name]
            if self.world.is_joint_continuous(joint_name):
                np.testing.assert_almost_equal(shortest_angular_distance(goal, current), 0, decimal=decimal,
                                               err_msg='{}: actual: {} desired: {}'.format(joint_name, current,
                                                                                           goal))
            else:
                np.testing.assert_almost_equal(current, goal, decimal,
                                               err_msg='{}: actual: {} desired: {}'.format(joint_name, current,
                                                                                           goal))

    #
    # GOAL STUFF #################################################################################################
    #

    def set_joint_goal(self, goal, weight=None, hard=False, decimal=2, expected_error_codes=(MoveResult.SUCCESS,),
                       check=True):
        """
        :type goal: dict
        """
        super().set_joint_goal(goal, weight=weight, hard=hard)
        if check:
            self.add_goal_check(JointGoalChecker(self.god_map, goal, decimal))

    def teleport_base(self, goal_pose):
        goal_pose = tf.transform_pose(self.default_root, goal_pose)
        js = {'odom_x_joint': goal_pose.pose.position.x,
              'odom_y_joint': goal_pose.pose.position.y,
              'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                      goal_pose.pose.orientation.y,
                                                                      goal_pose.pose.orientation.z,
                                                                      goal_pose.pose.orientation.w]))[0]}
        goal = SetJointStateRequest()
        goal.state = position_dict_to_joint_states(js)
        # self.set_base.call(goal)
        rospy.sleep(0.5)

    def set_rotation_goal(self, goal_orientation, tip_link, root_link=None, weight=None, max_velocity=None, check=True,
                          **kwargs):
        if not root_link:
            root_link = self.default_root
        super().set_rotation_goal(goal_orientation=goal_orientation,
                                  tip_link=tip_link,
                                  root_link=root_link,
                                  max_velocity=max_velocity,
                                  weight=weight, **kwargs)
        if check:
            self.add_goal_check(RotationGoalChecker(self.god_map, tip_link, root_link, goal_orientation))

    def set_translation_goal(self, goal_point, tip_link, root_link=None, weight=None, max_velocity=None, check=True,
                             **kwargs):
        if root_link is None:
            root_link = self.default_root
        super().set_translation_goal(goal_point=goal_point,
                                     tip_link=tip_link,
                                     root_link=root_link,
                                     max_velocity=max_velocity,
                                     weight=weight,
                                     **kwargs)
        if check:
            self.add_goal_check(TranslationGoalChecker(self.god_map, tip_link, root_link, goal_point))

    def set_straight_translation_goal(self, goal_pose, tip_link, root_link=None, weight=None, max_velocity=None,
                                      **kwargs):
        if not root_link:
            root_link = self.default_root
        super().set_straight_translation_goal(goal_pose, tip_link, root_link,
                                              max_velocity=max_velocity, weight=weight,
                                              **kwargs)

    def set_cart_goal(self, goal_pose, tip_link, root_link=None, weight=None, linear_velocity=None,
                      angular_velocity=None, check=True, **kwargs):
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        self.set_translation_goal(goal_point=goal_point,
                                  tip_link=tip_link,
                                  root_link=root_link,
                                  weight=weight,
                                  max_velocity=linear_velocity,
                                  check=check,
                                  **kwargs)
        goal_orientation = QuaternionStamped()
        goal_orientation.header = goal_pose.header
        goal_orientation.quaternion = goal_pose.pose.orientation
        self.set_rotation_goal(goal_orientation=goal_orientation,
                               tip_link=tip_link,
                               root_link=root_link,
                               weight=weight,
                               max_velocity=angular_velocity,
                               check=check,
                               **kwargs)

    def set_diff_drive_base_goal(self, goal_pose, tip_link=None, root_link=None, weight=None, linear_velocity=None,
                                 angular_velocity=None, check=True, **kwargs):
        if tip_link is None:
            tip_link = 'base_footprint'
        if root_link is None:
            root_link = self.default_root
        self.set_json_goal(constraint_type='DiffDriveBaseGoal',
                           tip_link=tip_link,
                           root_link=root_link,
                           goal_pose=goal_pose,
                           max_linear_velocity=linear_velocity,
                           max_angular_velocity=angular_velocity,
                           weight=weight)
        if check:
            goal_point = PointStamped()
            goal_point.header = goal_pose.header
            goal_point.point = goal_pose.pose.position
            self.add_goal_check(TranslationGoalChecker(self.god_map, tip_link, root_link, goal_point))
            goal_orientation = QuaternionStamped()
            goal_orientation.header = goal_pose.header
            goal_orientation.quaternion = goal_pose.pose.orientation
            self.add_goal_check(RotationGoalChecker(self.god_map, tip_link, root_link, goal_orientation))

    def set_keep_hand_in_workspace(self, tip_link, map_frame=None, base_footprint=None):
        self.set_json_goal('KeepHandInWorkspace',
                           tip_link=tip_link,
                           map_frame=map_frame,
                           base_footprint=base_footprint)

    def set_diff_drive_tangential_to_point(self, goal_point: PointStamped, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        self.set_json_goal('DiffDriveTangentialToPoint',
                           goal_point=goal_point,
                           weight=weight,
                           **kwargs)

    def set_pointing_goal(self, tip_link, goal_point, root_link=None, pointing_axis=None, weight=None, check=True,
                          **kwargs: goal_parameter):
        root_link = root_link if root_link else self.default_root
        super().set_pointing_goal(tip_link=tip_link,
                                  goal_point=goal_point,
                                  root_link=root_link,
                                  pointing_axis=pointing_axis,
                                  weight=weight)
        if check:
            self.add_goal_check(PointingGoalChecker(self.god_map,
                                                    tip_link=tip_link,
                                                    goal_point=goal_point,
                                                    root_link=root_link,
                                                    pointing_axis=pointing_axis))

    def set_align_planes_goal(self, tip_link, tip_normal, root_link=None, root_normal=None, max_angular_velocity=None,
                              weight=None, check=True):
        if root_link is None:
            root_link = self.robot.root_link_name
        super().set_align_planes_goal(tip_link=str(tip_link),
                                      tip_normal=tip_normal,
                                      root_link=str(root_link),
                                      root_normal=root_normal,
                                      max_angular_velocity=max_angular_velocity,
                                      weight=weight)
        if check:
            self.add_goal_check(AlignPlanesGoalChecker(self.god_map, tip_link, tip_normal, root_link, root_normal))

    def add_goal_check(self, goal_checker):
        self.goal_checks[self.number_of_cmds - 1].append(goal_checker)

    def set_straight_cart_goal(self, goal_pose, tip_link, root_link=None, weight=None, linear_velocity=None,
                               angular_velocity=None, check=True):
        if not root_link:
            root_link = self.default_root
        super().set_straight_cart_goal(goal_pose, tip_link, root_link, weight=weight,
                                       max_linear_velocity=linear_velocity,
                                       max_angular_velocity=angular_velocity)

        if check:
            goal_point = PointStamped()
            goal_point.header = goal_pose.header
            goal_point.point = goal_pose.pose.position
            self.add_goal_check(TranslationGoalChecker(self.god_map, tip_link, root_link, goal_point))
            goal_orientation = QuaternionStamped()
            goal_orientation.header = goal_pose.header
            goal_orientation.quaternion = goal_pose.pose.orientation
            self.add_goal_check(RotationGoalChecker(self.god_map, tip_link, root_link, goal_orientation))

    #
    # GENERAL GOAL STUFF ###############################################################################################
    #

    def plan_and_execute(self, expected_error_codes=None, stop_after=None, wait=True):
        return self.send_goal(expected_error_codes=expected_error_codes, stop_after=stop_after, wait=wait)

    def plan(self, expected_error_codes=None, wait: bool = True) -> MoveResult:
        return self.send_goal(expected_error_codes, MoveGoal.PLAN_ONLY, wait)

    def send_goal(self, expected_error_codes=None, goal_type=MoveGoal.PLAN_AND_EXECUTE, goal=None, stop_after=None,
                  wait=True):
        try:
            time_spend_giskarding = time()
            if stop_after is not None:
                super().send_goal(goal_type, wait=False)
                rospy.sleep(stop_after)
                self.interrupt()
                rospy.sleep(1)
                r = self.get_result(rospy.Duration(3))
            elif not wait:
                super().send_goal(goal_type, wait=wait)
                return
            else:
                r = super().send_goal(goal_type, wait=wait)
            self.wait_heartbeats()
            diff = time() - time_spend_giskarding
            self.total_time_spend_giskarding += diff
            self.total_time_spend_moving += len(self.god_map.get_data(identifier.trajectory).keys()) * \
                                            self.god_map.get_data(identifier.sample_period)
            logging.logwarn(f'Goal processing took {diff}')
            for cmd_id in range(len(r.error_codes)):
                error_code = r.error_codes[cmd_id]
                error_message = r.error_messages[cmd_id]
                if expected_error_codes is None:
                    expected_error_code = MoveResult.SUCCESS
                else:
                    expected_error_code = expected_error_codes[cmd_id]
                assert error_code == expected_error_code, \
                    f'in goal {cmd_id}; ' \
                    f'got: {move_result_error_code(error_code)}, ' \
                    f'expected: {move_result_error_code(expected_error_code)} | error_massage: {error_message}'
            if error_code == MoveResult.SUCCESS:
                try:
                    self.wait_heartbeats(30)
                    for goal_checker in self.goal_checks[len(r.error_codes) - 1]:
                        goal_checker()
                except:
                    logging.logerr(f'Goal #{cmd_id} did\'t pass test.')
                    raise
            # self.are_joint_limits_violated()
        finally:
            self.goal_checks = defaultdict(list)
            self.sync_world_with_trajectory()
        return r

    def sync_world_with_trajectory(self):
        t = self.god_map.get_data(identifier.trajectory)
        whole_last_joint_state = t.get_last().to_position_dict()
        for group_name in self._object_js_topics:
            group_joints = self.get_group_info(group_name).joint_state.name
            group_last_joint_state = {str(k): v for k, v in whole_last_joint_state.items() if k in group_joints}
            self.set_object_joint_state(group_name, group_last_joint_state)

    def get_result_trajectory_position(self):
        trajectory = self.god_map.unsafe_get_data(identifier.trajectory)
        trajectory2 = {}
        for joint_name in trajectory.get_exact(0).keys():
            trajectory2[joint_name] = np.array([p[joint_name].position for t, p in trajectory.items()])
        return trajectory2

    def get_result_trajectory_velocity(self):
        trajectory = self.god_map.get_data(identifier.trajectory)
        trajectory2 = {}
        for joint_name in trajectory.get_exact(0).keys():
            trajectory2[joint_name] = np.array([p[joint_name].velocity for t, p in trajectory.items()])
        return trajectory2

    def are_joint_limits_violated(self):
        joints = list(self.world.controlled_joints)
        for joint in joints:
            try:
                lower_limit, upper_limit = self.world.joints[joint].get_limit_expressions(0)
            except:
                continue
            assert lower_limit < self.world.state[joint].position < upper_limit, \
                f'joint limit of {joint} is violated {lower_limit} < {self.world.state[joint].position} < {upper_limit}'

    def are_joint_limits_in_traj_violated(self):
        trajectory_vel = self.get_result_trajectory_velocity()
        trajectory_pos = self.get_result_trajectory_position()
        controlled_joints = self.god_map.get_data(identifier.controlled_joints)
        for joint_name in controlled_joints:
            if isinstance(self.world.joints[joint_name], OneDofJoint):
                if not self.world.is_joint_continuous(joint_name):
                    joint_limits = self.world.get_joint_position_limits(joint_name)
                    error_msg = f'{joint_name} has violated joint position limit'
                    eps = 0.0001
                    np.testing.assert_array_less(trajectory_pos[joint_name], joint_limits[1] + eps, error_msg)
                    np.testing.assert_array_less(-trajectory_pos[joint_name], -joint_limits[0] + eps, error_msg)
                vel_limit = self.world.get_joint_velocity_limits(joint_name)[1] * 1.001
                vel = trajectory_vel[joint_name]
                error_msg = f'{joint_name} has violated joint velocity limit {vel} > {vel_limit}'
                assert np.all(np.less_equal(vel, vel_limit)), error_msg
                assert np.all(np.greater_equal(vel, -vel_limit)), error_msg

    #
    # BULLET WORLD #####################################################################################################
    #

    TimeOut = 5000

    def register_group(self, group_name: str, parent_group_name: str, root_link_name: str):
        super().register_group(group_name=group_name,
                               parent_group_name=parent_group_name,
                               root_link_name=root_link_name)
        assert group_name in self.get_group_names()

    @property
    def world(self):
        """
        :rtype: giskardpy.model.world.WorldTree
        """
        return self.god_map.get_data(identifier.world)

    def clear_world(self, timeout: float = TimeOut) -> UpdateWorldResponse:
        respone = super().clear_world(timeout=timeout)
        assert respone.error_codes == UpdateWorldResponse.SUCCESS
        assert len(self.world.groups) == 1
        assert len(self.get_group_names()) == 1
        assert self.original_number_of_links == len(self.world.links)
        return respone

    def remove_group(self,
                     name: str,
                     timeout: float = TimeOut,
                     expected_response: int = UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        if expected_response == UpdateWorldResponse.SUCCESS:
            old_link_names = self.world.groups[name].link_names
            old_joint_names = self.world.groups[name].joint_names
        r = super(GiskardTestWrapper, self).remove_group(name, timeout=timeout)
        assert r.error_codes == expected_response, \
            f'Got: \'{update_world_error_code(r.error_codes)}\', ' \
            f'expected: \'{update_world_error_code(expected_response)}.\''
        assert name not in self.world.groups
        assert name not in self.get_group_names()
        if expected_response == UpdateWorldResponse.SUCCESS:
            for old_link_name in old_link_names:
                assert old_link_name not in self.world.link_names
            for old_joint_name in old_joint_names:
                assert old_joint_name not in self.world.joint_names
        return r

    def detach_group(self, name, timeout: float = TimeOut, expected_response=UpdateWorldResponse.SUCCESS):
        if expected_response == UpdateWorldResponse.SUCCESS:
            response = super().detach_group(name, timeout=timeout)
            self.check_add_object_result(response=response,
                                         name=name,
                                         size=None,
                                         pose=None,
                                         parent_link=self.world.root_link_name,
                                         parent_link_group='',
                                         expected_error_code=expected_response)

    def check_add_object_result(self,
                                response: UpdateWorldResponse,
                                name: str,
                                size: Optional,
                                pose: Optional[PoseStamped],
                                parent_link: str,
                                parent_link_group: str,
                                expected_error_code: int):
        assert response.error_codes == expected_error_code, \
            f'Got: \'{update_world_error_code(response.error_codes)}\', ' \
            f'expected: \'{update_world_error_code(expected_error_code)}.\''
        if expected_error_code == UpdateWorldResponse.SUCCESS:
            assert name in self.get_group_names()
            response2 = self.get_group_info(name)
            if pose is not None:
                p = self.transform_msg(self.world.root_link_name, pose)
                o_p = self.world.groups[name].base_pose
                compare_poses(p.pose, o_p)
                compare_poses(o_p, response2.root_link_pose.pose)
            if parent_link_group != '':
                robot = self.get_group_info(parent_link_group)
                assert name in robot.child_groups
                short_parent_link = self.world.groups[parent_link_group].get_link_short_name_match(parent_link)
                assert short_parent_link == self.world.get_parent_link_of_link(self.world.groups[name].root_link_name)
            else:
                if parent_link == '':
                    parent_link = self.world.root_link_name
                assert parent_link == self.world.get_parent_link_of_link(self.world.groups[name].root_link_name)
        else:
            if expected_error_code != UpdateWorldResponse.DUPLICATE_GROUP_ERROR:
                assert name not in self.world.groups
                assert name not in self.get_group_names()

    def add_box(self,
                name: str,
                size: Tuple[float, float, float],
                pose: PoseStamped,
                parent_link: str = '',
                parent_link_group: str = '',
                timeout: float = TimeOut,
                expected_error_code: int = UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_box(name=name,
                                   size=size,
                                   pose=pose,
                                   parent_link=parent_link,
                                   parent_link_group=parent_link_group,
                                   timeout=timeout)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=size,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def update_group_pose(self, group_name: str, new_pose: PoseStamped, timeout: float = TimeOut,
                          expected_error_code=UpdateWorldResponse.SUCCESS):
        try:
            res = super().update_group_pose(group_name, new_pose, timeout)
        except UnknownGroupException as e:
            assert expected_error_code == UpdateWorldResponse.UNKNOWN_GROUP_ERROR
        if expected_error_code == UpdateWorldResponse.SUCCESS:
            info = self.get_group_info(group_name)
            map_T_group = tf.transform_pose(self.world.root_link_name, new_pose)
            compare_poses(info.root_link_pose.pose, map_T_group.pose)

    def add_sphere(self,
                   name: str,
                   radius: float = 1,
                   pose: PoseStamped = None,
                   parent_link: str = '',
                   parent_link_group: str = '',
                   timeout: float = TimeOut,
                   expected_error_code=UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_sphere(name=name,
                                      radius=radius,
                                      pose=pose,
                                      parent_link=parent_link,
                                      parent_link_group=parent_link_group,
                                      timeout=timeout)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def add_cylinder(self,
                     name: str,
                     height: float,
                     radius: float,
                     pose: PoseStamped = None,
                     parent_link: str = '',
                     parent_link_group: str = '',
                     timeout: float = TimeOut,
                     expected_error_code=UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_cylinder(name=name,
                                        height=height,
                                        radius=radius,
                                        pose=pose,
                                        parent_link=parent_link,
                                        parent_link_group=parent_link_group,
                                        timeout=timeout)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def add_mesh(self,
                 name: str = 'meshy',
                 mesh: str = '',
                 pose: PoseStamped = None,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 scale: Tuple[float, float, float] = (1, 1, 1),
                 timeout: float = TimeOut,
                 expected_error_code=UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_mesh(name=name,
                                    mesh=mesh,
                                    pose=pose,
                                    parent_link=parent_link,
                                    parent_link_group=parent_link_group,
                                    scale=scale,
                                    timeout=timeout)
        pose = utils.make_pose_from_parts(pose=pose, frame_id=pose.header.frame_id,
                                          position=pose.pose.position, orientation=pose.pose.orientation)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def add_urdf(self,
                 name: str,
                 urdf: str,
                 pose: PoseStamped,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 js_topic: str = '',
                 set_js_topic: Optional[str] = None,
                 timeout: float = TimeOut,
                 expected_error_code=UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_urdf(name=name,
                                    urdf=urdf,
                                    pose=pose,
                                    parent_link=parent_link,
                                    parent_link_group=parent_link_group,
                                    js_topic=js_topic,
                                    set_js_topic=set_js_topic,
                                    timeout=timeout)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def update_parent_link_of_group(self,
                                    name: str,
                                    parent_link: str = '',
                                    parent_link_group: str = '',
                                    timeout: float = TimeOut,
                                    expected_response: int = UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        r = super(GiskardTestWrapper, self).update_parent_link_of_group(name=name,
                                                                        parent_link=parent_link,
                                                                        parent_link_group=parent_link_group,
                                                                        timeout=timeout)
        self.wait_heartbeats()
        assert r.error_codes == expected_response, \
            f'Got: \'{update_world_error_code(r.error_codes)}\', ' \
            f'expected: \'{update_world_error_code(expected_response)}.\''
        if r.error_codes == r.SUCCESS:
            self.check_add_object_result(response=r,
                                         name=name,
                                         size=None,
                                         pose=None,
                                         parent_link=parent_link,
                                         parent_link_group=parent_link_group,
                                         expected_error_code=expected_response)
        return r

    @property
    def collision_avoidance_config(self) -> CollisionAvoidanceConfig:
        return self.god_map.unsafe_get_data(identifier.collision_avoidance_config)

    def get_external_collisions(self, link, distance_threshold):
        """
        :param distance_threshold:
        :rtype: list
        """
        self.collision_scene.reset_cache()
        collision_goals = [CollisionEntry(type=CollisionEntry.AVOID_COLLISION,
                                          distance=distance_threshold,
                                          group1=self.get_robot_name()),
                           CollisionEntry(type=CollisionEntry.ALLOW_COLLISION,
                                          distance=distance_threshold,
                                          group1=self.get_robot_name(),
                                          group2=self.get_robot_name())
                           ]
        collision_matrix = self.collision_scene.collision_goals_to_collision_matrix(collision_goals,
                                                                                    defaultdict(lambda: 0.3),
                                                                                    set(self.collision_avoidance_config.ignored_collisions))
        collisions = self.collision_scene.check_collisions(collision_matrix)
        controlled_parent_joint = self.robot.get_controlled_parent_joint_of_link(link)
        controlled_parent_link = self.robot.joints[controlled_parent_joint].child_link_name
        collision_list = collisions.get_external_collisions(controlled_parent_link)
        for key, self_collisions in collisions.self_collisions.items():
            if controlled_parent_link in key:
                collision_list.update(self_collisions)
        return collision_list

    def check_cpi_geq(self, links, distance_threshold):
        for link in links:
            collisions = self.get_external_collisions(link, distance_threshold)
            assert collisions[0].contact_distance >= distance_threshold, \
                f'distance for {link}: {collisions[0].contact_distance} < {distance_threshold} ' \
                f'({collisions[0].original_link_a} with {collisions[0].original_link_b})'

    def check_cpi_leq(self, links, distance_threshold):
        for link in links:
            collisions = self.get_external_collisions(link, distance_threshold)
            assert collisions[0].contact_distance <= distance_threshold, \
                f'distance for {link}: {collisions[0].contact_distance} > {distance_threshold} ' \
                f'({collisions[0].original_link_a} with {collisions[0].original_link_b})'

    def move_base(self, goal_pose):
        """
        :type goal_pose: PoseStamped
        """
        pass

    def reset(self):
        pass

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = tf.get_tf_root()
        p.pose.orientation.w = 1
        self.set_localization(p)
        self.wait_heartbeats()
        self.teleport_base(p)


class BaseBot(GiskardTestWrapper):
    default_pose = {
        'joint_x': 0.0,
        'joint_y': 0.0,
        'rot_z': 0.0,
    }

    def __init__(self):
        super().__init__('package://giskardpy/config/base_bot.yaml')

    def reset(self):
        self.clear_world()
        # self.set_joint_goal(self.default_pose)
        # self.plan_and_execute()


class BoxyTestWrapper(GiskardTestWrapper):
    default_pose = {
        'neck_shoulder_pan_joint': 0.0,
        'neck_shoulder_lift_joint': 0.0,
        'neck_elbow_joint': 0.0,
        'neck_wrist_1_joint': 0.0,
        'neck_wrist_2_joint': 0.0,
        'neck_wrist_3_joint': 0.0,
        'triangle_base_joint': 0.0,
        'left_arm_0_joint': 0.0,
        'left_arm_1_joint': 0.0,
        'left_arm_2_joint': 0.0,
        'left_arm_3_joint': 0.0,
        'left_arm_4_joint': 0.0,
        'left_arm_5_joint': 0.0,
        'left_arm_6_joint': 0.0,
        'right_arm_0_joint': 0.0,
        'right_arm_1_joint': 0.0,
        'right_arm_2_joint': 0.0,
        'right_arm_3_joint': 0.0,
        'right_arm_4_joint': 0.0,
        'right_arm_5_joint': 0.0,
        'right_arm_6_joint': 0.0,
    }

    better_pose = {
        'neck_shoulder_pan_joint': -1.57,
        'neck_shoulder_lift_joint': -1.88,
        'neck_elbow_joint': -2.0,
        'neck_wrist_1_joint': 0.139999387693,
        'neck_wrist_2_joint': 1.56999999998,
        'neck_wrist_3_joint': 0,
        'triangle_base_joint': -0.24,
        'left_arm_0_joint': -0.68,
        'left_arm_1_joint': 1.08,
        'left_arm_2_joint': -0.13,
        'left_arm_3_joint': -1.35,
        'left_arm_4_joint': 0.3,
        'left_arm_5_joint': 0.7,
        'left_arm_6_joint': -0.01,
        'right_arm_0_joint': 0.68,
        'right_arm_1_joint': -1.08,
        'right_arm_2_joint': 0.13,
        'right_arm_3_joint': 1.35,
        'right_arm_4_joint': -0.3,
        'right_arm_5_joint': -0.7,
        'right_arm_6_joint': 0.01,
    }

    def __init__(self, config=None):
        self.camera_tip = 'camera_link'
        self.r_tip = 'right_gripper_tool_frame'
        self.l_tip = 'left_gripper_tool_frame'
        self.r_gripper_group = 'r_gripper'
        super().__init__(Boxy)

    def move_base(self, goal_pose):
        goal_pose = tf.transform_pose(self.default_root, goal_pose)
        js = {'odom_x_joint': goal_pose.pose.position.x,
              'odom_y_joint': goal_pose.pose.position.y,
              'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                      goal_pose.pose.orientation.y,
                                                                      goal_pose.pose.orientation.z,
                                                                      goal_pose.pose.orientation.w]))[0]}
        self.allow_all_collisions()
        self.set_joint_goal(js)
        self.plan_and_execute()

    def reset(self):
        self.clear_world()
        self.reset_base()
        self.register_group(self.r_gripper_group, self.get_robot_name(), 'right_arm_7_link')


class BoxyCloseLoop(BoxyTestWrapper):

    def __init__(self, config=None):
        super().__init__('package://giskardpy/config/boxy_closed_loop.yaml')

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        self.move_base(p)

    def reset(self):
        self.clear_world()
        self.reset_base()


class HSR(GiskardTestWrapper):
    default_pose = {
        'arm_flex_joint': 0.0,
        'arm_lift_joint': 0.0,
        'arm_roll_joint': 0.0,
        'head_pan_joint': 0.0,
        'head_tilt_joint': 0.0,
        'odom_t': 0.0,
        'odom_x': 0.0,
        'odom_y': 0.0,
        'wrist_flex_joint': 0.0,
        'wrist_roll_joint': 0.0,
        'hand_l_spring_proximal_joint': 0,
        'hand_r_spring_proximal_joint': 0
    }
    better_pose = default_pose

    def __init__(self):
        self.tip = 'hand_gripper_tool_frame'
        super(HSR, self).__init__('package://giskardpy/config/hsr.yaml')

    def move_base(self, goal_pose):
        goal_pose = tf.transform_pose(self.default_root, goal_pose)
        js = {'odom_x': goal_pose.pose.position.x,
              'odom_y': goal_pose.pose.position.y,
              'odom_t': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                goal_pose.pose.orientation.y,
                                                                goal_pose.pose.orientation.z,
                                                                goal_pose.pose.orientation.w]))[0]}
        self.allow_all_collisions()
        self.set_joint_goal(js)
        self.plan_and_execute()

    def open_gripper(self):
        self.command_gripper(1.24)

    def close_gripper(self):
        self.command_gripper(0)

    def command_gripper(self, width):
        js = {'hand_motor_joint': width}
        self.set_joint_goal(js)
        self.plan_and_execute()

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        self.move_base(p)

    def reset(self):
        self.clear_world()
        self.close_gripper()
        self.reset_base()


def publish_marker_sphere(position, frame_id='map', radius=0.05, id_=0):
    m = Marker()
    m.action = m.ADD
    m.ns = 'debug'
    m.id = id_
    m.type = m.SPHERE
    m.header.frame_id = frame_id
    m.pose.position.x = position[0]
    m.pose.position.y = position[1]
    m.pose.position.z = position[2]
    m.color = ColorRGBA(1, 0, 0, 1)
    m.scale.x = radius
    m.scale.y = radius
    m.scale.z = radius

    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    start = rospy.get_rostime()
    while pub.get_num_connections() < 1 and (rospy.get_rostime() - start).to_sec() < 2:
        # wait for a connection to publisher
        # you can do whatever you like here or simply do nothing
        pass

    pub.publish(m)


def publish_marker_vector(start, end, diameter_shaft=0.01, diameter_head=0.02, id_=0):
    """
    assumes points to be in frame map
    :type start: Point
    :type end: Point
    :type diameter_shaft: float
    :type diameter_head: float
    :type id_: int
    """
    m = Marker()
    m.action = m.ADD
    m.ns = 'debug'
    m.id = id_
    m.type = m.ARROW
    m.points.append(start)
    m.points.append(end)
    m.color = ColorRGBA(1, 0, 0, 1)
    m.scale.x = diameter_shaft
    m.scale.y = diameter_head
    m.scale.z = 0
    m.header.frame_id = 'map'

    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    start = rospy.get_rostime()
    while pub.get_num_connections() < 1 and (rospy.get_rostime() - start).to_sec() < 2:
        # wait for a connection to publisher
        # you can do whatever you like here or simply do nothing
        pass
    rospy.sleep(0.3)

    pub.publish(m)


class SuccessfulActionServer(object):
    def __init__(self):
        self.name_space = rospy.get_param('~name_space')
        self.joint_names = rospy.get_param('~joint_names')
        self.state = {j: 0 for j in self.joint_names}
        self.pub = rospy.Publisher('{}/state'.format(self.name_space), control_msgs.msg.JointTrajectoryControllerState,
                                   queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.state_cb)
        self._as = actionlib.SimpleActionServer(self.name_space, control_msgs.msg.FollowJointTrajectoryAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()
        self._as.register_preempt_callback(self.preempt_requested)

    def state_cb(self, timer_event):
        msg = control_msgs.msg.JointTrajectoryControllerState()
        msg.header.stamp = timer_event.current_real
        msg.joint_names = self.joint_names
        self.pub.publish(msg)

    def preempt_requested(self):
        print('cancel called')
        self._as.set_preempted()

    def execute_cb(self, goal):
        rospy.sleep(goal.trajectory.points[-1].time_from_start)
        if self._as.is_active():
            self._as.set_succeeded()
