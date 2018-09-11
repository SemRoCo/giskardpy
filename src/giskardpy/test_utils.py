from multiprocessing import Queue
from threading import Thread

import rospy
from angles import normalize_angle, shortest_angular_distance
from geometry_msgs.msg import PoseStamped
from giskard_msgs.msg import MoveActionResult, CollisionEntry, MoveActionGoal, MoveResult
from giskard_msgs.srv import UpdateWorldResponse
from hypothesis import given, reproduce_failure, assume
import hypothesis.strategies as st
from hypothesis.strategies import composite
import keyword
import numpy as np
from numpy import pi

from py_trees import Blackboard
from sensor_msgs.msg import JointState

from giskard_trees import grow_tree
from giskardpy.data_types import ClosestPointInfo
from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.python_interface import GiskardWrapper
from giskardpy.symengine_robot import Robot
from giskardpy.tfwrapper import transform_pose, lookup_transform
from giskardpy.utils import msg_to_list
from ros_trajectory_controller_main import giskard_pm

BIG_NUMBER = 1e100
SMALL_NUMBER = 1e-100

vector = lambda x: st.lists(limited_float(), min_size=x, max_size=x)


def robot_urdfs():
    return st.sampled_from([u'urdfs/pr2.urdf', u'urdfs/boxy.urdf', u'urdfs/iai_donbot.urdf'])
    # return st.sampled_from([u'pr2.urdf'])


def angle(*args, **kwargs):
    return st.builds(normalize_angle, limited_float(*args, **kwargs))


def keys_values(max_length=10, value_type=st.floats(allow_nan=False)):
    return lists_of_same_length([variable_name(), value_type], max_length=max_length, unique=True)

def compare_axis_angle(angle1, axis1, angle2, axis2):
    if np.isclose(axis1, axis2).all():
        assert np.isclose(angle1, angle2), '{} != {}'.format(angle, angle2)
    elif np.isclose(axis1, -axis2).all():
        assert np.isclose(angle1, abs(angle2-2*pi)), '{} != {}'.format(angle, angle2)


@composite
def variable_name(draw):
    variable = draw(st.text(u'qwertyuiopasdfghjklzxcvbnm', min_size=1))
    assume(variable not in keyword.kwlist)
    return variable


@composite
def lists_of_same_length(draw, data_types=(), max_length=10, unique=False):
    length = draw(st.integers(min_value=1, max_value=max_length))
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
    pr2 = Robot.from_urdf_file(u'../test/urdfs/pr2.urdf')
    return draw(rnd_joint_state(*pr2.get_joint_limits()))


def limited_float(outer_limit=BIG_NUMBER, min_dist_to_zero=None):
    # f = st.floats(allow_nan=False, allow_infinity=False, max_value=outer_limit, min_value=-outer_limit)
    f = st.floats(allow_nan=False, allow_infinity=False)
    if min_dist_to_zero is not None:
        f = f.filter(lambda x: (outer_limit > abs(x) and abs(x) > min_dist_to_zero) or x == 0)
    else:
        f = f.filter(lambda x: abs(x) < outer_limit)
    return f


def unit_vector(length, elements=None):
    if elements is None:
        elements = limited_float(min_dist_to_zero=1e-20)
    vector = st.lists(elements,
                      min_size=length,
                      max_size=length).filter(lambda x: np.linalg.norm(x) > SMALL_NUMBER and
                                                        np.linalg.norm(x) < BIG_NUMBER)

    def normalize(v):
        l = np.linalg.norm(v)
        return [round(x / l, 10) for x in v]

    return st.builds(normalize, vector)


def quaternion(elements=None):
    return unit_vector(4, elements)


def pykdl_frame_to_numpy(pykdl_frame):
    return np.array([[pykdl_frame.M[0, 0], pykdl_frame.M[0, 1], pykdl_frame.M[0, 2], pykdl_frame.p[0]],
                     [pykdl_frame.M[1, 0], pykdl_frame.M[1, 1], pykdl_frame.M[1, 2], pykdl_frame.p[1]],
                     [pykdl_frame.M[2, 0], pykdl_frame.M[2, 1], pykdl_frame.M[2, 2], pykdl_frame.p[2]],
                     [0, 0, 0, 1]])


class GiskardTestWrapper(object):
    def __init__(self):
        rospy.set_param(u'~enable_gui', False)
        rospy.set_param(u'~interactive_marker_chains', [])
        rospy.set_param(u'~map_frame', u'map')
        rospy.set_param(u'~joint_convergence_threshold', 0.002)
        rospy.set_param(u'~wiggle_precision_threshold', 4)
        rospy.set_param(u'~sample_period', 0.1)
        rospy.set_param(u'~default_joint_vel_limit', 10)
        rospy.set_param(u'~default_collision_avoidance_distance', 0.05)
        rospy.set_param(u'~fill_velocity_values', False)
        rospy.set_param(u'~nWSR', u'None')
        rospy.set_param(u'~root_link', u'base_footprint')
        rospy.set_param(u'~enable_collision_marker', True)
        rospy.set_param(u'~enable_self_collision', False)
        rospy.set_param(u'~path_to_data_folder', u'../data/pr2/')
        rospy.set_param(u'~collision_time_threshold', 15)
        rospy.set_param(u'~max_traj_length', 30)
        self.sub_result = rospy.Subscriber(u'/giskardpy/command/result', MoveActionResult, self.cb, queue_size=100)

        self.tree = grow_tree()
        self.tree.tick()
        rospy.sleep(1)
        self.wrapper = GiskardWrapper(ns=u'tests')
        self.results = Queue(100)
        self.robot = self.tree.root.children[0].children[0].children[1]._plugins[u'fk'].robot
        self.controlled_joints = Blackboard().god_map.safe_get_data([u'controlled_joints'])
        self.joint_limits = {joint_name: self.robot.get_joint_lower_upper_limit(joint_name) for joint_name in
                             self.controlled_joints if self.robot.is_joint_controllable(joint_name)}
        self.world = Blackboard().god_map.safe_get_data([u'pybullet_world'])  # type: PyBulletWorld
        self.default_root = u'base_link'
        self.r_tip = u'r_gripper_tool_frame'
        self.l_tip = u'l_gripper_tool_frame'
        self.map = u'map'
        self.simple_base_pose_pub = rospy.Publisher(u'/move_base_simple/goal', PoseStamped, queue_size=10)
        rospy.sleep(1)

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
        return self.controlled_joints

    def get_l_gripper_links(self):
        return [u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link', u'l_gripper_l_finger_link',
                u'l_gripper_r_finger_link', u'l_gripper_r_finger_link', u'l_gripper_palm_link']

    def get_r_gripper_links(self):
        return [u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link', u'r_gripper_l_finger_link',
                u'r_gripper_r_finger_link', u'r_gripper_r_finger_link', u'r_gripper_palm_link']

    def get_allow_l_gripper(self, body_b=u'box'):
        links = self.get_l_gripper_links()
        return [CollisionEntry(CollisionEntry.ALLOW_COLLISION, 0, link, body_b, '') for link in links]

    def get_l_gripper_collision_entries(self, body_b=u'box', distance=0, action=CollisionEntry.ALLOW_COLLISION):
        links = self.get_l_gripper_links()
        return [CollisionEntry(action, distance, link, body_b, '') for link in links]

    def get_current_joint_state(self):
        """
        :rtype: JointState
        """
        return rospy.wait_for_message(u'joint_states', JointState)

    def tear_down(self):
        rospy.sleep(1)
        print(u'stopping plugins')
        # self.tree.destroy(

    #
    # JOINT GOAL STUFF #################################################################################################
    #
    def set_joint_goal(self, js):
        """
        :rtype js: dict
        """
        self.wrapper.set_joint_goal(js)

    def check_joint_state(self, expected):
        current_joint_state = self.get_current_joint_state()
        for i, joint_name in enumerate(current_joint_state.name):
            if joint_name in expected:
                goal = expected[joint_name]
                current = current_joint_state.position[i]
                if self.robot.is_joint_continuous(joint_name):
                    np.testing.assert_almost_equal(shortest_angular_distance(goal, current), 0)
                else:
                    np.testing.assert_almost_equal(goal, current, 2)

    def send_and_check_joint_goal(self, goal):
        """
        :type goal: dict
        """
        self.set_joint_goal(goal)
        self.send_and_check_goal()
        self.check_joint_state(goal)

    #
    # CART GOAL STUFF ##################################################################################################
    #
    def set_cart_goal(self, root, tip, goal_pose):
        self.wrapper.set_cart_goal(root, tip, goal_pose)

    def set_and_check_cart_goal(self, root, tip, goal_pose):
        self.set_cart_goal(root, tip, goal_pose)
        self.send_and_check_goal()
        self.check_cart_goal(tip, goal_pose)

    def check_cart_goal(self, tip, goal_pose):
        goal_in_base = transform_pose(u'base_footprint', goal_pose)
        current_pose = lookup_transform(u'base_footprint', tip)
        np.testing.assert_array_almost_equal(msg_to_list(goal_in_base.pose.position),
                                             msg_to_list(current_pose.pose.position), decimal=3)

        np.testing.assert_array_almost_equal(msg_to_list(goal_in_base.pose.orientation),
                                             msg_to_list(current_pose.pose.orientation), decimal=2)

    #
    # GENERAL GOAL STUFF ###############################################################################################
    #
    def get_as(self):
        return Blackboard().get('giskardpy/command')

    def send_goal(self):
        """
        :rtype: MoveResult
        """
        goal = MoveActionGoal()
        goal.goal = self.wrapper._get_goal()
        i = 0
        t1 = Thread(target=self.get_as()._as.action_server.internal_goal_callback, args=(goal,))
        t1.start()
        while self.results.empty():
            self.loop_once()
            # if i >= 100:
            #     assert False, 'planning took too long'
            rospy.sleep(.5)
            i += 1
        t1.join()
        self.loop_once()
        result = self.results.get()
        return result

    def send_and_check_goal(self, expected_error_code=MoveResult.SUCCESS):
        assert self.send_goal().error_code == expected_error_code

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

    def add_waypoint(self):
        self.wrapper.add_cmd()

    #
    # BULLET WORLD #####################################################################################################
    #
    def clear_world(self):
        assert self.wrapper.clear_world().error_codes == UpdateWorldResponse.SUCCESS
        assert len(self.world.get_object_names()) == 1
        assert len(self.world.get_robot().get_attached_objects()) == 0
        assert self.world.has_object(u'plane')

    def remove_object(self, name, expected_response=UpdateWorldResponse.SUCCESS):
        assert self.wrapper.remove_object(name).error_codes == expected_response
        assert not self.world.has_object(name)

    def add_box(self, name=u'box', position=(1.2, 0, 0.5)):
        r = self.wrapper.add_box(name=name, position=position)
        assert r.error_codes == UpdateWorldResponse.SUCCESS
        assert self.world.has_object(name)

    def add_sphere(self, name=u'sphere', position=(1.2, 0, 0.5)):
        r = self.wrapper.add_sphere(name=name, position=position)
        assert r.error_codes == UpdateWorldResponse.SUCCESS
        assert self.world.has_object(name)

    def add_cylinder(self, name=u'cylinder', position=(1.2, 0, 0.5)):
        r = self.wrapper.add_cylinder(name=name, position=position)
        assert r.error_codes == UpdateWorldResponse.SUCCESS
        assert self.world.has_object(name)

    def add_urdf(self, name, urdf, js_topic, pose):
        r = self.wrapper.add_urdf(name, urdf, js_topic, pose)
        assert r.error_codes == UpdateWorldResponse.SUCCESS
        assert self.world.has_object(name)

    def allow_all_collisions(self):
        self.wrapper.allow_all_collisions()

    def add_collision_entries(self, collisions_entries):
        self.wrapper.set_collision_entries(collisions_entries)

    def attach_box(self, name=u'box', size=(1, 1, 1), frame_id=u'map', position=(0, 0, 0), orientation=(0, 0, 0, 1),
                   expected_response=UpdateWorldResponse.SUCCESS):
        assert self.wrapper.attach_box(name, size, frame_id, position, orientation).error_codes == expected_response

    def check_cpi_geq(self, links, distance_threshold):
        cpi_identifier = u'cpi'
        cpi = Blackboard().god_map.safe_get_data([cpi_identifier])
        if cpi == 0 or cpi == None:
            return False
        for link in links:
            assert cpi[link].contact_distance >= distance_threshold, u'{} -- {}\n {} < {}'.format(link,
                                                                                                  cpi[link].link_b,
                                                                                                  cpi[
                                                                                                      link].contact_distance,
                                                                                                  distance_threshold)

    def check_cpi_leq(self, links, distance_threshold):
        cpi_identifier = u'cpi'
        cpi = Blackboard().god_map.safe_get_data([cpi_identifier])
        if cpi == 0 or cpi == None:
            return False
        for link in links:
            assert cpi[link].contact_distance <= distance_threshold, u'{} -- {}\n {} > {}'.format(link,
                                                                                                  cpi[link].link_b,
                                                                                                  cpi[
                                                                                                      link].contact_distance,
                                                                                                  distance_threshold)
