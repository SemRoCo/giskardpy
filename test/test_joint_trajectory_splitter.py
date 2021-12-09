import pytest
import roslaunch
import rospy
from actionlib_msgs.msg import GoalStatusArray, GoalID
from control_msgs.msg import FollowJointTrajectoryResult

from giskard_msgs.msg import MoveResult, MoveFeedback
from giskardpy.utils import logging
from utils_for_tests import BaseBot


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
        rospy.sleep(1)
        return self.statuses[i].status_list[0].status


@pytest.fixture(scope='module')
def ros_launch(request, ros):
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    with open('urdfs/2d_base_bot.urdf', 'r') as f:
        urdf = f.read()
        rospy.set_param('robot_description', urdf)
    joint_state_publisher = roslaunch.core.Node('joint_state_publisher', 'joint_state_publisher',
                                                'joint_state_publisher')
    joint_state_publisher = launch.launch(joint_state_publisher)
    localization = roslaunch.core.Node(package='tf',
                                       node_type='static_transform_publisher',
                                       name='localization',
                                       args='0 0 0 0 0 0 map base_footprint 100')
    localization = launch.launch(localization)

    def stop_joint_state_publisher():
        joint_state_publisher.stop()
        localization.stop()
        while joint_state_publisher.is_alive() or localization.is_alive():
            logging.loginfo('waiting for nodes to finish')
            rospy.sleep(0.2)

    request.addfinalizer(stop_joint_state_publisher)
    return launch


@pytest.fixture(scope='function')
def giskard(request, ros):
    c = BaseBot()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture(scope='function')
def launch_fake_servers(request, ros_launch):
    nodes = []
    for name_space, joint_names, sleep_percent, result in request.param:
        rospy.set_param('{}/name_space'.format(name_space), name_space)
        rospy.set_param('{}/joint_names'.format(name_space), joint_names)
        rospy.set_param('{}/sleep_factor'.format(name_space), sleep_percent)
        rospy.set_param('{}/result'.format(name_space), result)
        node = roslaunch.core.Node('giskardpy', 'fake_servers.py',
                                   name=name_space,
                                   output='screen')
        nodes.append(ros_launch.launch(node))

    def fin():
        for node in nodes:
            node.stop()
            while node.is_alive():
                logging.loginfo('waiting for nodes to finish')
                rospy.sleep(0.2)

    request.addfinalizer(fin)
    rospy.sleep(1)


class Tester(object):
    # TODO test external cancel

    @pytest.mark.parametrize('launch_fake_servers', [
        [['xy', ['joint_x', 'joint_y'], 1, FollowJointTrajectoryResult.SUCCESSFUL],
         ['z', ['rot_z'], 1, FollowJointTrajectoryResult.SUCCESSFUL]],
    ],
                             indirect=True)
    def test_success(self, launch_fake_servers, giskard):
        giskard.set_joint_goal({
            'joint_x': 1,
            'joint_y': -1,
            'rot_z': 0.2,
        }, check=False)
        giskard.plan_and_execute()

    @pytest.mark.parametrize('launch_fake_servers', [[
        ['xy', ['joint_x', 'joint_y'], 1, FollowJointTrajectoryResult.SUCCESSFUL],
        ['z', ['rot_z'], 1, FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED]],
        [['xy', ['joint_x', 'joint_y'], 1, FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED],
         ['z', ['rot_z'], 1, FollowJointTrajectoryResult.SUCCESSFUL]]
    ],
                             indirect=True)
    def test_success_fail_GOAL_TOLERANCE_VIOLATED(self, launch_fake_servers, giskard):
        # TODO check for interrupted
        giskard.set_joint_goal({
            'joint_x': 0.3,
            'joint_y': -0.3,
            'rot_z': 0.1,
        }, check=False)
        giskard.plan_and_execute(expected_error_codes=[MoveResult.FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED])

    @pytest.mark.parametrize('launch_fake_servers', [
        [['xy', ['joint_x', 'joint_y'], 1, FollowJointTrajectoryResult.SUCCESSFUL],
         ['z', ['rot_z'], 10, FollowJointTrajectoryResult.SUCCESSFUL]],
        [['xy', ['joint_x', 'joint_y'], 10, FollowJointTrajectoryResult.SUCCESSFUL],
         ['z', ['rot_z'], 1, FollowJointTrajectoryResult.SUCCESSFUL]]
    ],
                             indirect=True)
    def test_timeout(self, launch_fake_servers, giskard):
        giskard.set_joint_goal({
            'joint_x': 0.3,
            'joint_y': -0.3,
            'rot_z': 0.1,
        }, check=False)
        giskard.plan_and_execute(expected_error_codes=[MoveResult.EXECUTION_TIMEOUT])

    @pytest.mark.parametrize('launch_fake_servers', [
        [['xy', ['joint_x', 'joint_y'], 0.5, FollowJointTrajectoryResult.SUCCESSFUL],
         ['z', ['rot_z'], 1, FollowJointTrajectoryResult.SUCCESSFUL]],
        [['xy', ['joint_x', 'joint_y'], 1, FollowJointTrajectoryResult.SUCCESSFUL],
         ['z', ['rot_z'], 0.5, FollowJointTrajectoryResult.SUCCESSFUL]]
    ],
                             indirect=True)
    def test_too_quick(self, launch_fake_servers, giskard):
        giskard.set_joint_goal({
            'joint_x': 0.3,
            'joint_y': -0.3,
            'rot_z': 0.1,
        }, check=False)
        giskard.plan_and_execute(expected_error_codes=[MoveResult.EXECUTION_SUCCEEDED_PREMATURELY])

    @pytest.mark.parametrize('launch_fake_servers', [
        [['xy', ['joint_x', 'joint_y'], 1, FollowJointTrajectoryResult.SUCCESSFUL],
         ['z', ['rot_z'], 1, FollowJointTrajectoryResult.SUCCESSFUL]],
    ],
                             indirect=True)
    def test_external_preempt(self, launch_fake_servers, giskard):
        pub = rospy.Publisher('/xy/cancel', GoalID, queue_size=1)
        giskard.set_joint_goal({
            'joint_x': 1,
            'joint_y': -0.3,
            'rot_z': 0.1,
        }, check=False)
        giskard.plan_and_execute(wait=False)
        while pub.get_num_connections() < 1 or \
                giskard.last_feedback is None or \
                giskard.last_feedback.state != MoveFeedback.EXECUTION:
            rospy.sleep(0.1)
        pub.publish(GoalID())
        r = giskard.get_result()
        assert r.error_codes[0] == MoveResult.EXECUTION_PREEMPTED
        pass