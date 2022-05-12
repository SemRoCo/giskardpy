from time import time
import keyword
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Queue
from threading import Thread

import hypothesis.strategies as st
import numpy as np
import rospy
from actionlib_msgs.msg import GoalID
from angles import shortest_angular_distance
from control_msgs.msg import FollowJointTrajectoryActionGoal, FollowJointTrajectoryActionResult
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from giskard_msgs.msg import MoveActionResult, CollisionEntry, MoveActionGoal, MoveResult, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse
from hypothesis import assume
from hypothesis.strategies import composite
from iai_naive_kinematics_sim.srv import SetJointState, SetJointStateRequest
from iai_wsg_50_msgs.msg import PositionCmd
from numpy import pi
from py_trees import Blackboard
from sensor_msgs.msg import JointState
from tf.transformations import rotation_from_matrix, quaternion_matrix

from giskardpy import logging, identifier
from giskardpy.garden import grow_tree
from giskardpy.identifier import robot, world
from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.python_interface import GiskardWrapper
from giskardpy.robot import Robot
from giskardpy.tfwrapper import transform_pose, lookup_pose
from giskardpy.utils import msg_to_list, KeyDefaultDict, position_dict_to_joint_states, get_ros_pkg_path, \
    to_joint_state_position_dict

BIG_NUMBER = 1e100
SMALL_NUMBER = 1e-100

vector = lambda x: st.lists(float_no_nan_no_inf(), min_size=x, max_size=x)

update_world_error_codes = {value: name for name, value in vars(UpdateWorldResponse).items() if
                            isinstance(value, int) and name[0].isupper()}


def update_world_error_code(code):
    return update_world_error_codes[code]


move_result_error_codes = {value: name for name, value in vars(MoveResult).items() if
                           isinstance(value, int) and name[0].isupper()}


def move_result_error_code(code):
    return move_result_error_codes[code]


def robot_urdfs():
    return st.sampled_from([u'urdfs/pr2.urdfs', u'urdfs/boxy.urdfs', u'urdfs/iai_donbot.urdfs'])
    # return st.sampled_from([u'pr2.urdfs'])


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


def compare_poses(pose1, pose2, decimal=2):
    """
    :type pose1: Pose
    :type pose2: Pose
    """
    np.testing.assert_almost_equal(pose1.position.x, pose2.position.x, decimal=decimal)
    np.testing.assert_almost_equal(pose1.position.y, pose2.position.y, decimal=decimal)
    np.testing.assert_almost_equal(pose1.position.z, pose2.position.z, decimal=decimal)
    try:
        np.testing.assert_almost_equal(pose1.orientation.x, pose2.orientation.x, decimal=decimal)
        np.testing.assert_almost_equal(pose1.orientation.y, pose2.orientation.y, decimal=decimal)
        np.testing.assert_almost_equal(pose1.orientation.z, pose2.orientation.z, decimal=decimal)
        np.testing.assert_almost_equal(pose1.orientation.w, pose2.orientation.w, decimal=decimal)
    except:
        np.testing.assert_almost_equal(pose1.orientation.x, -pose2.orientation.x, decimal=decimal)
        np.testing.assert_almost_equal(pose1.orientation.y, -pose2.orientation.y, decimal=decimal)
        np.testing.assert_almost_equal(pose1.orientation.z, -pose2.orientation.z, decimal=decimal)
        np.testing.assert_almost_equal(pose1.orientation.w, -pose2.orientation.w, decimal=decimal)


@composite
def variable_name(draw):
    variable = draw(st.text(u'qwertyuiopasdfghjklzxcvbnm', min_size=1))
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
    pr2 = Robot.from_urdf_file(pr2_urdf())
    return draw(rnd_joint_state(*pr2.get_joint_limits()))


def pr2_urdf():
    with open(u'urdfs/pr2_with_base.urdf', u'r') as f:
        urdf_string = f.read()
    return urdf_string


def pr2_without_base_urdf():
    with open(u'urdfs/pr2.urdf', u'r') as f:
        urdf_string = f.read()
    return urdf_string


def base_bot_urdf():
    with open(u'urdfs/2d_base_bot.urdf', u'r') as f:
        urdf_string = f.read()
    return urdf_string


def donbot_urdf():
    with open(u'urdfs/iai_donbot.urdf', u'r') as f:
        urdf_string = f.read()
    return urdf_string


def boxy_urdf():
    with open(u'urdfs/boxy.urdf', u'r') as f:
        urdf_string = f.read()
    return urdf_string


def float_no_nan_no_inf(outer_limit=None, min_dist_to_zero=None):
    return st.floats(allow_nan=False, allow_infinity=False, max_value=outer_limit, min_value=outer_limit)
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
    l = draw(st.lists(float_no_nan_no_inf(), min_size=i_sq, max_size=i_sq))
    return np.array(l).reshape((i, i))


def unit_vector(length, elements=None):
    if elements is None:
        elements = float_no_nan_no_inf(min_dist_to_zero=1e-10)
    vector = st.lists(elements,
                      min_size=length,
                      max_size=length).filter(lambda x: np.linalg.norm(x) > SMALL_NUMBER and
                                                        np.linalg.norm(x) < BIG_NUMBER)

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


class GiskardTestWrapper(GiskardWrapper):
    def __init__(self, config_file):
        self.total_time_spend_giskarding = 0
        self.total_time_spend_moving = 0

        rospy.set_param('~config', config_file)
        self.sub_result = rospy.Subscriber('~command/result', MoveActionResult, self.cb, queue_size=100)
        self.cancel_goal = rospy.Publisher('~command/cancel', GoalID, queue_size=100)
        self.start_motion_sub = rospy.Subscriber('/whole_body_controller/follow_joint_trajectory/goal',
                                                FollowJointTrajectoryActionGoal, self.start_motion_cb,
                                                 queue_size=100)
        self.stop_motion_sub = rospy.Subscriber('/whole_body_controller/follow_joint_trajectory/result',
                                               FollowJointTrajectoryActionResult, self.stop_motion_cb,
                                                queue_size=100)

        self.tree = grow_tree()
        self.loop_once()
        # rospy.sleep(1)
        super(GiskardTestWrapper, self).__init__(node_name=u'tests')
        self.results = Queue(100)
        self.default_root = self.get_robot().get_root()
        self.map = u'map'
        self.simple_base_pose_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.set_base = rospy.ServiceProxy('/base_simulator/set_joint_states', SetJointState)
        self.tick_rate = 10

        def create_publisher(topic):
            p = rospy.Publisher(topic, JointState, queue_size=10)
            rospy.sleep(.2)
            return p

        self.joint_state_publisher = KeyDefaultDict(create_publisher)
        # rospy.sleep(1)

    def start_motion_cb(self, msg):
        self.time = time()

    def stop_motion_cb(self, msg):
        self.total_time_spend_moving += time() - self.time

    def wait_for_synced(self):
        sleeper = rospy.Rate(self.tick_rate)
        self.loop_once()
        # while self.tree.tip().name != u'has goal':
        #     self.loop_once()
        #     sleeper.sleep()
        # self.loop_once()
        # while self.tree.tip().name != u'has goal':
        #     self.loop_once()
        #     sleeper.sleep()

    def get_robot(self):
        """
        :rtype: Robot
        """
        return self.get_god_map().get_data(robot)

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return Blackboard().god_map

    def cb(self, msg):
        """
        :type msg: MoveActionResult
        """
        self.results.put(msg.result)

    def loop_once(self):
        self.tree.tick()

    def get_controlled_joint_names(self):
        """
        :rtype: dict
        """
        return self.get_robot().controlled_joints

    def get_controllable_links(self):
        return self.get_robot().get_controlled_links()

    def get_current_joint_state(self):
        """
        :rtype: JointState
        """
        return rospy.wait_for_message('joint_states', JointState)

    def tear_down(self):
        rospy.sleep(1)
        logging.loginfo(u'total time spend giskarding: {}'.format(self.total_time_spend_giskarding-self.total_time_spend_moving))
        logging.loginfo(u'total time spend moving: {}'.format(self.total_time_spend_moving))
        logging.loginfo(u'stopping plugins')

    def set_object_joint_state(self, object_name, joint_state):
        super(GiskardTestWrapper, self).set_object_joint_state(object_name, joint_state)
        rospy.sleep(0.5)
        self.wait_for_synced()
        current_js = self.get_world().get_object(object_name).joint_state
        for joint_name, state in joint_state.items():
            np.testing.assert_almost_equal(current_js[joint_name].position, state, 2)

    def set_kitchen_js(self, joint_state, object_name=u'kitchen'):
        self.set_object_joint_state(object_name, joint_state)

    #
    # JOINT GOAL STUFF #################################################################################################
    #

    def compare_joint_state(self, current_js, goal_js, decimal=2):
        """
        :type current_js: dict
        :type goal_js: dict
        :type decimal: int
        """
        joint_names = set(current_js.keys()).intersection(set(goal_js.keys()))
        for joint_name in joint_names:
            goal = goal_js[joint_name]
            current = current_js[joint_name]
            if self.get_robot().is_joint_continuous(joint_name):
                np.testing.assert_almost_equal(shortest_angular_distance(goal, current), 0, decimal=decimal)
            else:
                np.testing.assert_almost_equal(current, goal, decimal,
                                               err_msg='{} at {} insteand of {}'.format(joint_name, current, goal))

    def check_current_joint_state(self, expected, decimal=2):
        current_joint_state = to_joint_state_position_dict(self.get_current_joint_state())
        self.compare_joint_state(current_joint_state, expected, decimal=decimal)

    def send_and_check_joint_goal(self, goal, weight=None, decimal=2, expected_error_codes=None):
        """
        :type goal: dict
        """
        self.set_joint_goal(goal, weight=weight)
        self.send_and_check_goal(expected_error_codes=expected_error_codes)
        if expected_error_codes is None:
            self.check_current_joint_state(goal, decimal=decimal)

    #
    # CART GOAL STUFF ##################################################################################################
    #
    def teleport_base(self, goal_pose):
        goal_pose = transform_pose(self.default_root, goal_pose)
        js = {u'odom_x_joint': goal_pose.pose.position.x,
              u'odom_y_joint': goal_pose.pose.position.y,
              u'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                       goal_pose.pose.orientation.y,
                                                                       goal_pose.pose.orientation.z,
                                                                       goal_pose.pose.orientation.w]))[0]}
        goal = SetJointStateRequest()
        goal.state = position_dict_to_joint_states(js)
        self.set_base.call(goal)
        self.loop_once()
        rospy.sleep(0.5)
        self.loop_once()

    def keep_position(self, tip, root=None):
        if root is None:
            root = self.default_root
        goal = PoseStamped()
        goal.header.frame_id = tip
        goal.pose.orientation.w = 1
        self.set_cart_goal(goal, tip, root)

    def keep_orientation(self, tip, root=None):
        goal = PoseStamped()
        goal.header.frame_id = tip
        goal.pose.orientation.w = 1
        self.set_rotation_goal(goal, tip, root)

    def set_rotation_goal(self, goal_pose, tip_link, root_link=None, weight=None, max_velocity=None):
        if not root_link:
            root_link = self.default_root
        super(GiskardTestWrapper, self).set_rotation_goal(goal_pose, tip_link, root_link, max_velocity=max_velocity)

    def set_translation_goal(self, goal_pose, tip_link, root_link=None, weight=None, max_velocity=None):
        if not root_link:
            root_link = self.default_root
        super(GiskardTestWrapper, self).set_translation_goal(goal_pose, tip_link, root_link, max_velocity=max_velocity)


    def set_straight_translation_goal(self, goal_pose, tip_link, root_link=None, weight=None, max_velocity=None):
        if not root_link:
            root_link = self.default_root
        super(GiskardTestWrapper, self).set_straight_translation_goal(goal_pose, tip_link, root_link, max_velocity=max_velocity)

    def set_cart_goal(self, goal_pose, tip_link, root_link=None, weight=None, linear_velocity=None, angular_velocity=None):
        if not root_link:
            root_link = self.default_root
        if weight is not None:
            super(GiskardTestWrapper, self).set_cart_goal(goal_pose, tip_link, root_link, weight=weight, max_linear_velocity=linear_velocity,
                                       max_angular_velocity=angular_velocity)
        else:
            super(GiskardTestWrapper, self).set_cart_goal(goal_pose, tip_link, root_link, max_linear_velocity=linear_velocity,
                                       max_angular_velocity=angular_velocity)


    def set_straight_cart_goal(self, goal_pose, tip_link, root_link=None, weight=None, linear_velocity=None, angular_velocity=None):
        if not root_link:
            root_link = self.default_root
        if weight is not None:
            super(GiskardTestWrapper, self).set_straight_cart_goal(goal_pose, tip_link, root_link, weight=weight, trans_max_velocity=linear_velocity,
                                       rot_max_velocity=angular_velocity)
        else:
            super(GiskardTestWrapper, self).set_straight_cart_goal(goal_pose, tip_link, root_link, trans_max_velocity=linear_velocity,
                                       rot_max_velocity=angular_velocity)


    def set_and_check_cart_goal(self, goal_pose, tip_link, root_link=None, weight=None, linear_velocity=None, angular_velocity=None,
                                expected_error_codes=None):
        goal_pose_in_map = transform_pose(u'map', goal_pose)
        self.set_cart_goal(goal_pose, tip_link, root_link, weight, linear_velocity=linear_velocity, angular_velocity=angular_velocity)
        self.loop_once()
        self.send_and_check_goal(expected_error_codes)
        self.loop_once()
        if expected_error_codes is None:
            self.check_cart_goal(tip_link, goal_pose_in_map)


    def set_and_check_straight_cart_goal(self, goal_pose, tip_link, root_link=None, weight=None, linear_velocity=None, angular_velocity=None,
                                expected_error_codes=None):
        goal_pose_in_map = transform_pose(u'map', goal_pose)
        self.set_straight_cart_goal(goal_pose, tip_link, root_link, weight, linear_velocity=linear_velocity, angular_velocity=angular_velocity)
        self.loop_once()
        self.send_and_check_goal(expected_error_codes)
        self.loop_once()
        if expected_error_codes is None:
            self.check_cart_goal(tip_link, goal_pose_in_map)


    def check_cart_goal(self, tip_link, goal_pose):
        goal_in_base = transform_pose(u'map', goal_pose)
        current_pose = lookup_pose(u'map', tip_link)
        np.testing.assert_array_almost_equal(msg_to_list(goal_in_base.pose.position),
                                             msg_to_list(current_pose.pose.position), decimal=2)

        try:
            np.testing.assert_array_almost_equal(msg_to_list(goal_in_base.pose.orientation),
                                                 msg_to_list(current_pose.pose.orientation), decimal=2)
        except AssertionError:
            np.testing.assert_array_almost_equal(msg_to_list(goal_in_base.pose.orientation),
                                                 -np.array(msg_to_list(current_pose.pose.orientation)), decimal=2)

    #
    # GENERAL GOAL STUFF ###############################################################################################
    #

    def interrupt(self):
        self.cancel_goal.publish(GoalID())

    def check_reachability(self, expected_error_codes=None):
        self.send_and_check_goal(expected_error_codes=expected_error_codes,
                                 goal_type=MoveGoal.PLAN_AND_CHECK_REACHABILITY)

    def get_as(self):
        return Blackboard().get(u'~command')

    def send_goal(self, goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE, wait=True):
        """
        :rtype: MoveResult
        """
        if goal is None:
            goal = MoveActionGoal()
            goal.goal = self._get_goal()
            goal.goal.type = goal_type
        i = 0
        self.loop_once()
        t = time()
        t1 = Thread(target=self.get_as()._as.action_server.internal_goal_callback, args=(goal,))
        self.loop_once()
        t1.start()
        sleeper = rospy.Rate(self.tick_rate)
        while self.results.empty():
            self.loop_once()
            sleeper.sleep()
            i += 1
        t1.join()
        t = time() - t
        self.total_time_spend_giskarding += t
        self.loop_once()
        result = self.results.get()
        return result

    def send_goal_and_dont_wait(self, goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE, stop_after=20):
        if goal is None:
            goal = MoveActionGoal()
            goal.goal = self._get_goal()
            goal.goal.type = goal_type
        i = 0
        self.loop_once()
        t1 = Thread(target=self.get_as()._as.action_server.internal_goal_callback, args=(goal,))
        self.loop_once()
        t1.start()
        sleeper = rospy.Rate(self.tick_rate)
        while self.results.empty():
            self.loop_once()
            sleeper.sleep()
            i += 1
            if i > stop_after:
                self.interrupt()
        t1.join()
        self.loop_once()
        result = self.results.get()
        return result

    def send_and_check_goal(self, expected_error_codes=None, goal_type=MoveGoal.PLAN_AND_EXECUTE, goal=None):
        r = self.send_goal(goal=goal, goal_type=goal_type)
        for i in range(len(r.error_codes)):
            error_code = r.error_codes[i]
            error_message = r.error_messages[i]
            if expected_error_codes is None:
                expected_error_code = MoveResult.SUCCESS
            else:
                expected_error_code = expected_error_codes[i]
            assert error_code == expected_error_code, \
                u'in goal {}; got: {}, expected: {} | error_massage: {}'.format(i, move_result_error_code(error_code),
                                                                                move_result_error_code(
                                                                                    expected_error_code),
                                                                                error_message)
        return r.trajectory

    def get_result_trajectory_position(self):
        trajectory = self.get_god_map().unsafe_get_data(identifier.trajectory)
        trajectory2 = []
        for t, p in trajectory._points.items():
            trajectory2.append({joint_name: js.position for joint_name, js in p.items()})
        return trajectory2

    def get_result_trajectory_velocity(self):
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        trajectory2 = []
        for t, p in trajectory._points.items():
            trajectory2.append({joint_name: js.velocity for joint_name, js in p.items()})
        return trajectory2

    def are_joint_limits_violated(self):
        controllable_joints = self.get_robot().get_movable_joints()
        trajectory_pos = self.get_result_trajectory_position()
        trajectory_vel = self.get_result_trajectory_velocity()

        for joint in controllable_joints:
            joint_limits = self.get_robot().get_joint_limits(joint)
            vel_limit = self.get_robot().get_joint_velocity_limit(joint)
            trajectory_pos_joint = [p[joint] for p in trajectory_pos]
            trajectory_vel_joint = [p[joint] for p in trajectory_vel]
            if any(round(p, 7) < joint_limits[0] and round(p, 7) > joint_limits[1] for p in trajectory_pos_joint):
                return True
            if any(round(p, 7) < vel_limit and round(p, 7) > vel_limit for p in trajectory_vel_joint):
                return True

        return False

    #
    # BULLET WORLD #####################################################################################################
    #
    def get_world(self):
        """
        :rtype: PyBulletWorld
        """
        return self.get_god_map().get_data(world)

    def clear_world(self):
        assert super(GiskardTestWrapper, self).clear_world().error_codes == UpdateWorldResponse.SUCCESS
        assert len(self.get_world().get_object_names()) == 0
        assert len(self.get_object_names().object_names) == 0
        # assert len(self.get_robot().get_attached_objects()) == 0
        # assert self.get_world().has_object(u'plane')

    def remove_object(self, name, expected_response=UpdateWorldResponse.SUCCESS):
        r = super(GiskardTestWrapper, self).remove_object(name)
        assert r.error_codes == expected_response, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(expected_response))
        assert not self.get_world().has_object(name)
        assert not name in self.get_object_names().object_names

    def detach_object(self, name, expected_response=UpdateWorldResponse.SUCCESS):
        if expected_response == UpdateWorldResponse.SUCCESS:
            p = self.get_robot().get_fk_pose(self.get_robot().get_root(), name)
            p = transform_pose(u'map', p)
            assert name in self.get_attached_objects().object_names, \
                'there is no attached object named {}'.format(name)
        r = super(GiskardTestWrapper, self).detach_object(name)
        assert r.error_codes == expected_response, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(expected_response))
        if expected_response == UpdateWorldResponse.SUCCESS:
            assert self.get_world().has_object(name)
            assert not name in self.get_attached_objects().object_names, 'the object was not detached'
            compare_poses(self.get_world().get_object(name).base_pose, p.pose, decimal=2)

    def add_box(self, name=u'box', size=(1, 1, 1), pose=None, expected_response=UpdateWorldResponse.SUCCESS):
        r = super(GiskardTestWrapper, self).add_box(name, size, pose=pose)
        assert r.error_codes == expected_response, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(expected_response))
        p = transform_pose(u'map', pose)
        o_p = self.get_world().get_object(name).base_pose
        assert self.get_world().has_object(name)
        compare_poses(p.pose, o_p)
        assert name in self.get_object_names().object_names
        compare_poses(o_p, self.get_object_info(name).pose.pose)

    def add_sphere(self, name=u'sphere', size=1, pose=None):
        r = super(GiskardTestWrapper, self).add_sphere(name=name, size=size, pose=pose)
        assert r.error_codes == UpdateWorldResponse.SUCCESS, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(UpdateWorldResponse.SUCCESS))
        assert self.get_world().has_object(name)
        assert name in self.get_object_names().object_names
        o_p = self.get_world().get_object(name).base_pose
        compare_poses(o_p, self.get_object_info(name).pose.pose)

    def add_cylinder(self, name=u'cylinder', size=[1, 1], pose=None):
        r = super(GiskardTestWrapper, self).add_cylinder(name=name, height=size[0], radius=size[1], pose=pose)
        assert r.error_codes == UpdateWorldResponse.SUCCESS, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(UpdateWorldResponse.SUCCESS))
        assert self.get_world().has_object(name)
        assert name in self.get_object_names().object_names

    def add_mesh(self, name=u'cylinder', path=u'', pose=None, expected_error=UpdateWorldResponse.SUCCESS):
        r = super(GiskardTestWrapper, self).add_mesh(name=name, mesh=path, pose=pose)
        assert r.error_codes == expected_error, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(expected_error))
        if expected_error == UpdateWorldResponse.SUCCESS:
            assert self.get_world().has_object(name)
            assert name in self.get_object_names().object_names
            o_p = self.get_world().get_object(name).base_pose
            compare_poses(o_p, self.get_object_info(name).pose.pose)
        else:
            assert not self.get_world().has_object(name)
            assert name not in self.get_object_names().object_names

    def add_urdf(self, name, urdf, pose, js_topic=u'', set_js_topic=None):
        r = super(GiskardTestWrapper, self).add_urdf(name, urdf, pose, js_topic, set_js_topic=set_js_topic)
        assert r.error_codes == UpdateWorldResponse.SUCCESS, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(UpdateWorldResponse.SUCCESS))
        assert self.get_world().has_object(name)
        assert name in self.get_object_names().object_names

    def avoid_all_collisions(self, distance=0.5):
        super(GiskardTestWrapper, self).avoid_all_collisions(distance)
        self.loop_once()

    def attach_box(self, name=u'box', size=None, frame_id=None, position=None, orientation=None, pose=None,
                   expected_response=UpdateWorldResponse.SUCCESS):
        scm = self.get_robot().get_self_collision_matrix()
        if pose is None:
            expected_pose = PoseStamped()
            expected_pose.header.frame_id = frame_id
            expected_pose.pose.position = Point(*position)
            if orientation:
                expected_pose.pose.orientation = Quaternion(*orientation)
            else:
                expected_pose.pose.orientation = Quaternion(0, 0, 0, 1)
        else:
            expected_pose = deepcopy(pose)
        r = super(GiskardTestWrapper, self).attach_box(name, size, frame_id, position, orientation, pose)
        assert r.error_codes == expected_response, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(expected_response))
        if expected_response == UpdateWorldResponse.SUCCESS:
            self.wait_for_synced()
            assert name in self.get_controllable_links()
            assert not self.get_world().has_object(name)
            assert not name in self.get_object_names().object_names
            assert name in self.get_attached_objects().object_names, 'object {} was not attached'
            assert scm.difference(self.get_robot().get_self_collision_matrix()) == set()
            assert len(scm) < len(self.get_robot().get_self_collision_matrix())
            compare_poses(expected_pose.pose, lookup_pose(frame_id, name).pose)
        self.loop_once()

    def attach_cylinder(self, name=u'cylinder', height=1, radius=1, frame_id=None, position=None, orientation=None,
                        expected_response=UpdateWorldResponse.SUCCESS):
        scm = self.get_robot().get_self_collision_matrix()
        expected_pose = PoseStamped()
        expected_pose.header.frame_id = frame_id
        expected_pose.pose.position = Point(*position)
        if orientation:
            expected_pose.pose.orientation = Quaternion(*orientation)
        else:
            expected_pose.pose.orientation = Quaternion(0, 0, 0, 1)
        r = super(GiskardTestWrapper, self).attach_cylinder(name, height, radius, frame_id, position, orientation)
        assert r.error_codes == expected_response, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(expected_response))
        if expected_response == UpdateWorldResponse.SUCCESS:
            self.wait_for_synced()
            assert name in self.get_controllable_links()
            assert not self.get_world().has_object(name)
            assert not name in self.get_object_names().object_names
            assert name in self.get_attached_objects().object_names
            assert scm.difference(self.get_robot().get_self_collision_matrix()) == set()
            assert len(scm) < len(self.get_robot().get_self_collision_matrix())
            compare_poses(expected_pose.pose, lookup_pose(frame_id, name).pose)
        self.loop_once()

    def attach_object(self, name=u'box', frame_id=None, expected_response=UpdateWorldResponse.SUCCESS):
        scm = self.get_robot().get_self_collision_matrix()
        r = super(GiskardTestWrapper, self).attach_object(name, frame_id)
        assert r.error_codes == expected_response, \
            u'got: {}, expected: {}'.format(update_world_error_code(r.error_codes),
                                            update_world_error_code(expected_response))
        self.wait_for_synced()
        assert name in self.get_controllable_links()
        assert not self.get_world().has_object(name)
        assert not name in self.get_object_names().object_names
        assert name in self.get_attached_objects().object_names
        assert scm.difference(self.get_robot().get_self_collision_matrix()) == set()
        assert len(scm) < len(self.get_robot().get_self_collision_matrix())
        self.loop_once()

    def get_external_collisions(self, link, distance_threshold):
        """
        :param distance_threshold:
        :rtype: list
        """
        collision_goals = [CollisionEntry(type=CollisionEntry.AVOID_ALL_COLLISIONS, min_dist=distance_threshold)]
        collision_matrix = self.get_world().collision_goals_to_collision_matrix(collision_goals,
                                                                                defaultdict(lambda: 0.3))
        collisions = self.get_world().check_collisions(collision_matrix)
        controlled_parent_joint = self.get_robot().get_controlled_parent_joint(link)
        controlled_parent_link = self.get_robot().get_child_link_of_joint(controlled_parent_joint)
        collision_list = collisions.get_external_collisions(controlled_parent_link)
        for key, self_collisions in collisions.self_collisions.items():
            if controlled_parent_link in key:
                collision_list.update(self_collisions)
        return collision_list

    def check_cpi_geq(self, links, distance_threshold):
        for link in links:
            collisions = self.get_external_collisions(link, distance_threshold)
            assert collisions[0].get_contact_distance() >= distance_threshold, \
                u'distance for {}: {} >= {}'.format(link,
                                                    collisions[0].get_contact_distance(),
                                                    distance_threshold)

    def check_cpi_leq(self, links, distance_threshold):
        for link in links:
            collisions = self.get_external_collisions(link, distance_threshold)
            assert collisions[0].get_contact_distance() <= distance_threshold, \
                u'distance for {}: {} <= {}'.format(link,
                                                    collisions[0].get_contact_distance(),
                                                    distance_threshold)

    def move_base(self, goal_pose):
        """
        :type goal_pose: PoseStamped
        """
        self.simple_base_pose_pub.publish(goal_pose)
        rospy.sleep(.07)
        self.wait_for_synced()
        current_pose = self.get_robot().get_base_pose()
        goal_pose = transform_pose(u'map', goal_pose)
        compare_poses(goal_pose.pose, current_pose.pose, decimal=1)

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = self.map
        p.pose.orientation.w = 1
        self.teleport_base(p)


class PR2(GiskardTestWrapper):
    def __init__(self):
        self.r_tip = u'r_gripper_tool_frame'
        self.l_tip = u'l_gripper_tool_frame'
        self.r_gripper = rospy.ServiceProxy(u'r_gripper_simulator/set_joint_states', SetJointState)
        self.l_gripper = rospy.ServiceProxy(u'l_gripper_simulator/set_joint_states', SetJointState)
        super(PR2, self).__init__(u'package://giskardpy/config/pr2.yaml')
        self.default_root = self.get_robot().get_root()

    def move_base(self, goal_pose):
        goal_pose = transform_pose(self.default_root, goal_pose)
        js = {u'odom_x_joint': goal_pose.pose.position.x,
              u'odom_y_joint': goal_pose.pose.position.y,
              u'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                       goal_pose.pose.orientation.y,
                                                                       goal_pose.pose.orientation.z,
                                                                       goal_pose.pose.orientation.w]))[0]}
        self.send_and_check_joint_goal(js)

    def get_l_gripper_links(self):
        return [u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link', u'l_gripper_l_finger_link',
                u'l_gripper_r_finger_link', u'l_gripper_r_finger_link', u'l_gripper_palm_link']

    def get_r_gripper_links(self):
        return [u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link', u'r_gripper_l_finger_link',
                u'r_gripper_r_finger_link', u'r_gripper_r_finger_link', u'r_gripper_palm_link']

    def get_r_upper_arm(self):
        return [u'r_shoulder_lift_link', u'r_upper_arm_roll_link', u'r_upper_arm_link']

    def get_r_forearm_links(self):
        return [u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_forearm_roll_link', u'r_forearm_link',
                u'r_forearm_link']

    def get_allow_l_gripper(self, body_b=u'box'):
        links = self.get_l_gripper_links()
        return [CollisionEntry(CollisionEntry.ALLOW_COLLISION, 0, [link], body_b, []) for link in links]

    def get_l_gripper_collision_entries(self, body_b=u'box', distance=0, action=CollisionEntry.ALLOW_COLLISION):
        links = self.get_l_gripper_links()
        return [CollisionEntry(action, distance, [link], body_b, []) for link in links]

    def open_r_gripper(self):
        sjs = SetJointStateRequest()
        sjs.state.name = [u'r_gripper_l_finger_joint', u'r_gripper_r_finger_joint', u'r_gripper_l_finger_tip_joint',
                          u'r_gripper_r_finger_tip_joint']
        sjs.state.position = [0.54, 0.54, 0.54, 0.54]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.r_gripper.call(sjs)

    def close_r_gripper(self):
        sjs = SetJointStateRequest()
        sjs.state.name = [u'r_gripper_l_finger_joint', u'r_gripper_r_finger_joint', u'r_gripper_l_finger_tip_joint',
                          u'r_gripper_r_finger_tip_joint']
        sjs.state.position = [0, 0, 0, 0]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.r_gripper.call(sjs)

    def open_l_gripper(self):
        sjs = SetJointStateRequest()
        sjs.state.name = [u'l_gripper_l_finger_joint', u'l_gripper_r_finger_joint', u'l_gripper_l_finger_tip_joint',
                          u'l_gripper_r_finger_tip_joint']
        sjs.state.position = [0.54, 0.54, 0.54, 0.54]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.l_gripper.call(sjs)

    def close_l_gripper(self):
        sjs = SetJointStateRequest()
        sjs.state.name = [u'l_gripper_l_finger_joint', u'l_gripper_r_finger_joint', u'l_gripper_l_finger_tip_joint',
                          u'l_gripper_r_finger_tip_joint']
        sjs.state.position = [0, 0, 0, 0]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.l_gripper.call(sjs)

    def move_pr2_base(self, goal_pose):
        """
        :type goal_pose: PoseStamped
        """
        self.simple_base_pose_pub.publish(goal_pose)

    def reset_pr2_base(self):
        p = PoseStamped()
        p.header.frame_id = self.map
        p.pose.orientation.w = 1
        self.move_pr2_base(p)


class Donbot(GiskardTestWrapper):
    def __init__(self):
        self.camera_tip = u'camera_link'
        self.gripper_tip = u'gripper_tool_frame'
        self.gripper_pub = rospy.Publisher(u'/wsg_50_driver/goal_position', PositionCmd, queue_size=10)
        super(Donbot, self).__init__(u'package://giskardpy/config/donbot.yaml')

    def move_base(self, goal_pose):
        goal_pose = transform_pose(self.default_root, goal_pose)
        js = {u'odom_x_joint': goal_pose.pose.position.x,
              u'odom_y_joint': goal_pose.pose.position.y,
              u'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                       goal_pose.pose.orientation.y,
                                                                       goal_pose.pose.orientation.z,
                                                                       goal_pose.pose.orientation.w]))[0]}
        self.allow_all_collisions()
        self.send_and_check_joint_goal(js)

    def open_gripper(self):
        self.set_gripper(0.109)

    def close_gripper(self):
        self.set_gripper(0)

    def set_gripper(self, width, gripper_joint=u'gripper_joint'):
        """
        :param width: goal width in m
        :type width: float
        """
        width = max(0.0065, min(0.109, width))
        goal = PositionCmd()
        goal.pos = width * 1000
        self.gripper_pub.publish(goal)
        rospy.sleep(0.5)
        js = self.get_current_joint_state()
        index = js.name.index(gripper_joint)
        np.testing.assert_almost_equal(js.position[index], width, decimal=3)


class KMR_IIWA(GiskardTestWrapper):
    def __init__(self):
        self.camera_tip = u'camera_link'
        self.gripper_tip = u'gripper_tool_frame'
        super(KMR_IIWA, self).__init__(u'package://giskardpy/config/kmr_iiwa.yaml')

    def move_base(self, goal_pose):
        goal_pose = transform_pose(self.default_root, goal_pose)
        js = {u'odom_x_joint': goal_pose.pose.position.x,
              u'odom_y_joint': goal_pose.pose.position.y,
              u'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                       goal_pose.pose.orientation.y,
                                                                       goal_pose.pose.orientation.z,
                                                                       goal_pose.pose.orientation.w]))[0]}
        self.allow_all_collisions()
        self.send_and_check_joint_goal(js)


class Boxy(GiskardTestWrapper):
    def __init__(self):
        self.camera_tip = u'camera_link'
        self.r_tip = u'right_gripper_tool_frame'
        self.l_tip = u'left_gripper_tool_frame'
        super(Boxy, self).__init__(u'package://giskardpy/config/boxy_sim.yaml')

    def move_base(self, goal_pose):
        goal_pose = transform_pose(self.default_root, goal_pose)
        js = {u'odom_x_joint': goal_pose.pose.position.x,
              u'odom_y_joint': goal_pose.pose.position.y,
              u'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                       goal_pose.pose.orientation.y,
                                                                       goal_pose.pose.orientation.z,
                                                                       goal_pose.pose.orientation.w]))[0]}
        self.allow_all_collisions()
        self.send_and_check_joint_goal(js)


class HSR(GiskardTestWrapper):
    def __init__(self):
        self.tip = u'hand_palm_link'
        super(HSR, self).__init__(u'package://giskardpy/config/hsr.yaml')

    def move_base(self, goal_pose):
        goal_pose = transform_pose(self.default_root, goal_pose)
        js = {u'odom_x': goal_pose.pose.position.x,
              u'odom_y': goal_pose.pose.position.y,
              u'odom_t': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                 goal_pose.pose.orientation.y,
                                                                 goal_pose.pose.orientation.z,
                                                                 goal_pose.pose.orientation.w]))[0]}
        self.allow_all_collisions()
        self.send_and_check_joint_goal(js)

    def open_gripper(self):
        js = {u'hand_l_spring_proximal_joint': 0.7,
              u'hand_r_spring_proximal_joint': 0.7}
        self.send_and_check_joint_goal(js)

    def close_gripper(self):
        js = {u'hand_l_spring_proximal_joint': 0,
              u'hand_r_spring_proximal_joint': 0}
        self.send_and_check_joint_goal(js)

    # def command_gripper(self, width):
    #     js = {u'hand_motor_joint': width}
    #     self.send_and_check_joint_goal(js)
