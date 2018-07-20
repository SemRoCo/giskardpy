#!/usr/bin/env python
from multiprocessing import Queue
from threading import Thread

import pytest
from giskard_msgs.msg import MoveActionGoal, MoveGoal, MoveActionResult, MoveResult
from hypothesis.strategies import composite

import hypothesis.strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, invariant, initialize, run_state_machine_as_test, \
    precondition
from giskardpy.python_interface import GiskardWrapper

import rospy

from hypothesis import settings

from giskardpy.test_utils import rnd_joint_state, rnd_joint_state2
from ros_trajectory_controller_main import giskard_pm


class TestPyBulletWorld(RuleBasedStateMachine):
    js = Bundle(u'js')
    joint_limits = Bundle(u'joint_limits')

    def __init__(self):
        super(TestPyBulletWorld, self).__init__()
        rospy.init_node(u'test_integration')
        self.giskard = GiskardWrapper(None)
        self.sub_result = rospy.Subscriber(u'/qp_controller/command/result', MoveActionResult, self.cb, queue_size=100)
        self.results = Queue(100)
        self.pm = None

    def cb(self, msg):
        self.results.put(msg.result)

    @initialize(target=joint_limits)
    def init(self):
        rospy.set_param(u'~interactive_marker_chains', [])
        rospy.set_param(u'~enable_gui', False)
        rospy.set_param(u'~map_frame', u'map')
        rospy.set_param(u'~joint_convergence_threshold', 0.002)
        rospy.set_param(u'~wiggle_precision_threshold', 7)
        rospy.set_param(u'~sample_period', 0.1)
        rospy.set_param(u'~default_joint_vel_limit', 0.5)
        rospy.set_param(u'~default_collision_avoidance_distance', 0.05)
        rospy.set_param(u'~fill_velocity_values', False)
        rospy.set_param(u'~nWSR', u'None')
        rospy.set_param(u'~root_link', u'base_footprint')
        rospy.set_param(u'~enable_collision_marker', False)
        rospy.set_param(u'~enable_self_collision', False)
        rospy.set_param(u'~path_to_data_folder', u'../data/pr2/')
        rospy.set_param(u'~collision_time_threshold', 15)
        rospy.set_param(u'~max_traj_length', 30)
        self.pm = giskard_pm()
        self.pm.start_plugins()
        # cart_plugin = self.pm._plugins[u'cart bullet controller'].replacement
        self.robot = self.pm._plugins[u'fk'].get_robot()
        controlled_joints = self.pm._plugins[u'controlled joints'].controlled_joints
        self.joint_limits = {joint_name: self.robot.get_joint_lower_upper_limit(joint_name) for joint_name in
                             controlled_joints if self.robot.is_joint_controllable(joint_name)}
        return self.joint_limits

    def loop_once(self):
        self.pm.update()

    # def set_goal(self):
    #     goal = MoveGoal()
    #     assert self.send_fake_goal().error_code == MoveResult.INSOLVABLE
    #     self.loop_once()

    @rule(js=rnd_joint_state2(joint_limits))
    def set_js_goal(self, js):
        self.giskard.set_joint_goal(js)
        assert self.send_fake_goal().error_code == MoveResult.SUCCESS

    def send_fake_goal(self):
        goal = MoveActionGoal()
        goal.goal = self.giskard._get_goal()

        t1 = Thread(target=self.pm._plugins[u'action server']._as.action_server.internal_goal_callback, args=(goal,))
        t1.start()
        while self.results.empty():
            self.loop_once()
        t1.join()
        result = self.results.get()
        return result

    def teardown(self):
        self.pm.stop()


@pytest.fixture(scope="session")
def shutdown_ros(request):
    def fin():
        rospy.sleep(0.5)
        rospy.signal_shutdown('holy fuck it took me way too long to figure this shit out.')

    request.addfinalizer(fin)
    return 0


TestTrees = TestPyBulletWorld.TestCase
TestTrees.settings = settings(max_examples=2, stateful_step_count=10, timeout=300)


def test_hack_to_shutdown_ros(shutdown_ros):
    pass


if __name__ == '__main__':
    pass
    state = TestPyBulletWorld()
    v1 = state.init()
    state.set_js_goal(js={'head_pan_joint': 2.0323372301955806e-08,
                          'head_tilt_joint': -0.1481484279260326,
                          'l_elbow_flex_joint': -0.25720986816287045,
                          'l_forearm_roll_joint': -2426096630891975.0,
                          'l_shoulder_lift_joint': 0.7277546370009919,
                          'l_shoulder_pan_joint': 0.009406189082520854,
                          'l_upper_arm_roll_joint': 0.0,
                          'l_wrist_flex_joint': -1.999999773072091,
                          'l_wrist_roll_joint': 9.559964998171719e-253,
                          'r_elbow_flex_joint': -1.4858974548339847,
                          'r_forearm_roll_joint': -1e-05,
                          'r_shoulder_lift_joint': -0.34563774483324,
                          'r_shoulder_pan_joint': 0.1329508508979699,
                          'r_upper_arm_roll_joint': 7.6809765110486e-07,
                          'r_wrist_flex_joint': -1.999536380792552,
                          'r_wrist_roll_joint': 1.1489250587411545e-165,
                          'torso_lift_joint': 0.2868457629922887})
    state.teardown()