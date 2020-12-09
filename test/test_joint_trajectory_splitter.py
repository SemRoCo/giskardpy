#! /usr/bin/env python

import copy

import actionlib
import control_msgs.msg
import pytest
import roslaunch
import rospy
import trajectory_msgs.msg
from actionlib_msgs.msg import GoalStatusArray

from giskardpy import logging


class Clients(object):
    class cb(object):
        def __init__(self, clients, statuses, i):
            self.clients = clients
            self.statuses = statuses
            self.i = i

        def __call__(self, data):
            if data.status_list != []:
                self.statuses[self.i] = data

    def __init__(self, splitter, others, state_topics):
        self.splitter = splitter
        self.others = others
        self.state_topics = []
        self.statuses = [None for _ in state_topics]
        for i, state_topic in enumerate(state_topics):
            self.state_topics.append(rospy.Subscriber(state_topic,
                                                      GoalStatusArray,
                                                      self.cb(self.others, self.statuses, i),
                                                      queue_size=10))

    def get_other_state(self, i):
        return self.statuses[i].status_list[0].status


@pytest.fixture(scope=u'module')
def init_ros():
    rospy.init_node('JointTrajectorySplitterTest')


def get_simple_trajectory_goal():
    goal = control_msgs.msg.FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

    point1 = trajectory_msgs.msg.JointTrajectoryPoint()
    point1.positions.extend([1.0, 1.0, 1, 0])
    point1.time_from_start = rospy.Time(0)
    goal.trajectory.points.append(point1)

    point2 = copy.copy(point1)
    point2.time_from_start = rospy.Time(1)
    goal.trajectory.points.append(point2)

    point3 = copy.copy(point1)
    point3.time_from_start = rospy.Time(2)
    goal.trajectory.points.append(point3)

    return goal


def get_missing_joint_goal():
    goal = control_msgs.msg.FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['joint1', 'joint3', 'joint4']

    point1 = trajectory_msgs.msg.JointTrajectoryPoint()
    point1.positions.extend([1.0, 1.0, 1, 0])
    point1.time_from_start = rospy.Time(0)
    goal.trajectory.points.append(point1)

    point2 = copy.copy(point1)
    point2.time_from_start = rospy.Time(1)
    goal.trajectory.points.append(point2)

    point3 = copy.copy(point1)
    point3.time_from_start = rospy.Time(2)
    goal.trajectory.points.append(point3)

    return goal


def get_long_trajectory_goal():
    goal = control_msgs.msg.FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

    point1 = trajectory_msgs.msg.JointTrajectoryPoint()
    point1.positions.extend([1.0, 1.0, 1, 0])
    point1.time_from_start = rospy.Time(0)
    goal.trajectory.points.append(point1)

    point2 = copy.copy(point1)
    point2.time_from_start = rospy.Time(10)
    goal.trajectory.points.append(point2)

    point3 = copy.copy(point1)
    point3.time_from_start = rospy.Time(20)
    goal.trajectory.points.append(point3)

    return goal


@pytest.fixture(scope=u'module')
def ros_launch():
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()
    return launch


@pytest.fixture(scope=u'function')
def launch_state_publisher1(request, init_ros, ros_launch):
    rospy.set_param('/state_publisher1/joint_names', ['joint1', 'joint2'])
    node = roslaunch.core.Node('giskardpy', 'state_publisher.py', name='state_publisher1', output='screen')
    state_publisher1 = ros_launch.launch(node)

    def fin():
        state_publisher1.stop()
        rospy.delete_param('/state_publisher1/joint_names')
        while (state_publisher1.is_alive()):
            logging.loginfo('waiting for nodes to finish')
            rospy.sleep(0.2)

    request.addfinalizer(fin)


@pytest.fixture(scope=u'function')
def launch_state_publisher2(request, init_ros, ros_launch):
    rospy.set_param('/state_publisher2/joint_names', ['joint3', 'joint4'])
    node = roslaunch.core.Node('giskardpy', 'state_publisher.py', name='state_publisher2', output='screen')
    state_publisher2 = ros_launch.launch(node)

    def fin():
        state_publisher2.stop()
        rospy.delete_param('/state_publisher2/joint_names')
        while (state_publisher2.is_alive()):
            logging.loginfo('waiting for nodes to finish')
            rospy.sleep(0.2)

    request.addfinalizer(fin)


@pytest.fixture(scope=u'function')
def launch_timeout_test_nodes(request, init_ros, ros_launch, launch_state_publisher1, launch_state_publisher2):
    node = roslaunch.core.Node('giskardpy', 'successful_action_server.py', name='successful_action_server1',
                               output='screen')
    process1 = ros_launch.launch(node)

    node = roslaunch.core.Node('giskardpy', 'timeout_action_server.py', name='timeout_action_server1', output='screen')
    process2 = ros_launch.launch(node)

    rospy.set_param('/joint_trajectory_splitter/state_topics', ['/state_publisher1', '/state_publisher2'])
    rospy.set_param('/joint_trajectory_splitter/client_topics',
                    ['/successful_action_server1', '/timeout_action_server1'])
    node = roslaunch.core.Node('giskardpy', 'joint_trajectory_splitter.py', name='joint_trajectory_splitter',
                               output='screen')
    process3 = ros_launch.launch(node)

    def fin():
        process1.stop()
        process2.stop()
        process3.stop()
        rospy.delete_param('/joint_trajectory_splitter/state_topics')
        rospy.delete_param('/joint_trajectory_splitter/client_topics')
        while (process1.is_alive() or process2.is_alive() or process3.is_alive()):
            logging.loginfo('waiting for nodes to finish')
            rospy.sleep(0.2)

    request.addfinalizer(fin)
    splitter = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/',
                                            control_msgs.msg.FollowJointTrajectoryAction)
    others = [
        actionlib.SimpleActionClient('/successful_action_server1/',
                                     control_msgs.msg.FollowJointTrajectoryAction),
        actionlib.SimpleActionClient('/successful_action_server1/',
                                     control_msgs.msg.FollowJointTrajectoryAction)
    ]
    r = Clients(splitter, others, ['/successful_action_server1/status',
                                   '/timeout_action_server1/status'])
    rospy.sleep(2)
    return r


@pytest.fixture(scope=u'function')
def launch_successful_test_nodes(request, init_ros, ros_launch, launch_state_publisher1, launch_state_publisher2):
    node = roslaunch.core.Node('giskardpy', 'successful_action_server.py', name='successful_action_server1',
                               output='screen')
    process1 = ros_launch.launch(node)

    node = roslaunch.core.Node('giskardpy', 'successful_action_server.py', name='successful_action_server2',
                               output='screen')
    process2 = ros_launch.launch(node)

    rospy.set_param('/joint_trajectory_splitter/state_topics', ['/state_publisher1', '/state_publisher2'])
    rospy.set_param('/joint_trajectory_splitter/client_topics',
                    ['/successful_action_server1', '/successful_action_server2'])
    node = roslaunch.core.Node('giskardpy', 'joint_trajectory_splitter.py', name='joint_trajectory_splitter',
                               output='screen')
    process3 = ros_launch.launch(node)

    def fin():
        process1.stop()
        process2.stop()
        process3.stop()
        rospy.delete_param('/joint_trajectory_splitter/state_topics')
        rospy.delete_param('/joint_trajectory_splitter/client_topics')
        while (process1.is_alive() or process2.is_alive() or process3.is_alive()):
            logging.loginfo('waiting for nodes to finish')
            rospy.sleep(0.2)

    request.addfinalizer(fin)

    splitter = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/',
                                            control_msgs.msg.FollowJointTrajectoryAction)
    others = [
        actionlib.SimpleActionClient('/successful_action_server1/',
                                     control_msgs.msg.FollowJointTrajectoryAction),
        actionlib.SimpleActionClient('/successful_action_server2/',
                                     control_msgs.msg.FollowJointTrajectoryAction)
    ]
    r = Clients(splitter, others, ['/successful_action_server1/status',
                                   '/successful_action_server2/status'])
    rospy.sleep(2)
    return r


@pytest.fixture(scope=u'function')
def launch_invalid_joints_test_nodes(request, init_ros, ros_launch, launch_state_publisher1, launch_state_publisher2):
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    node = roslaunch.core.Node('giskardpy', 'successful_action_server.py', name='successful_action_server1',
                               output='screen')
    process1 = launch.launch(node)

    node = roslaunch.core.Node('giskardpy', 'invalid_joints_action_server.py', name='invalid_joints_action_server1',
                               output='screen')
    process2 = launch.launch(node)

    rospy.set_param('/joint_trajectory_splitter/state_topics', ['/state_publisher1', '/state_publisher2'])
    rospy.set_param('/joint_trajectory_splitter/client_topics',
                    ['/successful_action_server1', '/invalid_joints_action_server1'])
    node = roslaunch.core.Node('giskardpy', 'joint_trajectory_splitter.py', name='joint_trajectory_splitter',
                               output='screen')
    process3 = launch.launch(node)

    def fin():
        process1.stop()
        process2.stop()
        process3.stop()
        rospy.delete_param('/joint_trajectory_splitter/state_topics')
        rospy.delete_param('/joint_trajectory_splitter/client_topics')
        while (process1.is_alive() or process2.is_alive() or process3.is_alive()):
            logging.loginfo('waiting for nodes to finish')
            rospy.sleep(0.2)

    request.addfinalizer(fin)

    splitter = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/',
                                            control_msgs.msg.FollowJointTrajectoryAction)
    others = [
        actionlib.SimpleActionClient('/successful_action_server1/',
                                     control_msgs.msg.FollowJointTrajectoryAction),
        actionlib.SimpleActionClient('/invalid_joints_action_server1/',
                                     control_msgs.msg.FollowJointTrajectoryAction)
    ]
    r = Clients(splitter, others, ['/successful_action_server1/status',
                                   '/invalid_joints_action_server1/status'])
    rospy.sleep(2)
    return r


@pytest.fixture(scope=u'function')
def launch_failing_goal_test_nodes(request, init_ros, ros_launch, launch_state_publisher1, launch_state_publisher2):
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    node = roslaunch.core.Node('giskardpy', 'successful_action_server.py', name='successful_action_server1')
    process1 = launch.launch(node)

    node = roslaunch.core.Node('giskardpy', 'failing_action_server.py', name='failing_action_server1')
    process2 = launch.launch(node)

    rospy.set_param('/joint_trajectory_splitter/state_topics', ['/state_publisher1', '/state_publisher1'])
    rospy.set_param('/joint_trajectory_splitter/client_topics',
                    ['/successful_action_server1', '/failing_action_server1'])
    node = roslaunch.core.Node('giskardpy', 'joint_trajectory_splitter.py', name='joint_trajectory_splitter')
    process3 = launch.launch(node)

    def fin():
        process1.stop()
        process2.stop()
        process3.stop()
        rospy.delete_param('/joint_trajectory_splitter/state_topics')
        rospy.delete_param('/joint_trajectory_splitter/client_topics')
        while (process1.is_alive() or process2.is_alive() or process3.is_alive()):
            logging.loginfo('waiting for nodes to finish')
            rospy.sleep(0.2)

    request.addfinalizer(fin)

    splitter = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/',
                                            control_msgs.msg.FollowJointTrajectoryAction)
    others = [
        actionlib.SimpleActionClient('/successful_action_server1/',
                                     control_msgs.msg.FollowJointTrajectoryAction),
        actionlib.SimpleActionClient('/failing_action_server1/',
                                     control_msgs.msg.FollowJointTrajectoryAction)
    ]
    r = Clients(splitter, others, ['/successful_action_server1/status',
                                   '/failing_action_server1/status'])
    rospy.sleep(2)
    return r


def test_timeout(launch_timeout_test_nodes):
    splitter = launch_timeout_test_nodes.splitter
    simple_trajectory_goal = get_simple_trajectory_goal()
    splitter.wait_for_server()
    splitter.send_goal(simple_trajectory_goal)
    splitter.wait_for_result()
    status = splitter.get_state()
    assert status == actionlib.GoalStatus.ABORTED
    assert launch_timeout_test_nodes.get_other_state(0) == actionlib.GoalStatus.SUCCEEDED
    assert launch_timeout_test_nodes.get_other_state(1) == actionlib.GoalStatus.PREEMPTED


def test_invalid_joints_error(launch_invalid_joints_test_nodes):
    action_client = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/',
                                                 control_msgs.msg.FollowJointTrajectoryAction)
    simple_trajectory_goal = get_simple_trajectory_goal()
    action_client.wait_for_server()
    action_client.send_goal(simple_trajectory_goal)
    action_client.wait_for_result()
    result = action_client.get_result()
    status = action_client.get_state()
    assert status == actionlib.GoalStatus.ABORTED
    assert result.error_code == control_msgs.msg.FollowJointTrajectoryResult.INVALID_JOINTS


def test_missing_joint_in_goal(launch_successful_test_nodes):
    action_client = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/',
                                                 control_msgs.msg.FollowJointTrajectoryAction)
    missing_joint_goal = get_missing_joint_goal()
    action_client.wait_for_server()
    action_client.send_goal(missing_joint_goal)
    action_client.wait_for_result()
    result = action_client.get_result()
    status = action_client.get_state()
    assert status == actionlib.GoalStatus.ABORTED
    assert result.error_code == control_msgs.msg.FollowJointTrajectoryResult.INVALID_GOAL


def test_successful_goal(launch_successful_test_nodes):
    action_client = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/',
                                                 control_msgs.msg.FollowJointTrajectoryAction)
    simple_trajectory_goal = get_simple_trajectory_goal()
    action_client.wait_for_server()
    action_client.send_goal(simple_trajectory_goal)
    action_client.wait_for_result()
    result = action_client.get_result()
    status = action_client.get_state()
    assert status == actionlib.GoalStatus.SUCCEEDED
    assert result.error_code == control_msgs.msg.FollowJointTrajectoryResult.SUCCESSFUL


def test_cancel_goal(launch_successful_test_nodes):
    action_client = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/',
                                                 control_msgs.msg.FollowJointTrajectoryAction)
    simple_trajectory_goal = get_simple_trajectory_goal()
    action_client.wait_for_server()
    action_client.send_goal(simple_trajectory_goal)
    rospy.sleep(0.2)
    action_client.cancel_goal()
    action_client.wait_for_result()
    result = action_client.get_result()
    status = action_client.get_state()
    assert status == actionlib.GoalStatus.PREEMPTED
    assert result.error_code == control_msgs.msg.FollowJointTrajectoryResult.SUCCESSFUL
    assert launch_successful_test_nodes.get_other_state(0) == actionlib.GoalStatus.PREEMPTED
    assert launch_successful_test_nodes.get_other_state(1) == actionlib.GoalStatus.PREEMPTED


def test_failing_goal(launch_failing_goal_test_nodes):
    action_client = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/',
                                                 control_msgs.msg.FollowJointTrajectoryAction)
    long_trajectory_goal = get_long_trajectory_goal()
    action_client.wait_for_server()
    start = rospy.Time.now()
    action_client.send_goal(long_trajectory_goal)
    action_client.wait_for_result()
    end = rospy.Time.now()
    result = action_client.get_result()
    status = action_client.get_state()
    assert status == actionlib.GoalStatus.ABORTED
    assert result.error_code == control_msgs.msg.FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED
    assert end - start < rospy.Duration(20) and end - start >= rospy.Duration(10)
