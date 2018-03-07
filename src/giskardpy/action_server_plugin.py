from collections import OrderedDict

import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult
from geometry_msgs.msg import PoseArray
from giskard_msgs.msg import ControllerListAction, ControllerListGoal, Controller, ControllerListResult

from giskardpy.plugin import IOPlugin
from giskardpy.trajectory import MultiJointState, SingleJointState


class ActionServer(IOPlugin):
    def get_readings(self):
        update = {self.joint_goal_identifier: self.joint_goal}
        return update

    def update(self):
        goal = self.databus.get_data(self.solution_identifier)
        if goal is not None:
            success = False

            print('waiting for {:.3f} sec with {} points'.format(goal.trajectory.points[-1].time_from_start.to_sec(),
                                                                 len(goal.trajectory.points)))
            self._ac.send_goal(goal)
            t = rospy.get_rostime()
            while not self._ac.wait_for_result(rospy.Duration(.1)):
                if self._as.is_preempt_requested():
                    rospy.loginfo('new goal, cancel old one')
                    self._ac.cancel_all_goals()
                    break
            else:
                print('shit took {:.3f}s'.format((rospy.get_rostime() - t).to_sec()))
                r = self._ac.get_result()
                print('real result {}'.format(r))
                if r.error_code == FollowJointTrajectoryResult.SUCCESSFUL:
                    success = True

            if success:
                self._as.set_succeeded(ControllerListResult())
                print('success')
            else:
                self._as.set_aborted(ControllerListResult())

    def start(self, databus):
        return super(ActionServer, self).start(databus)

    def stop(self):
        pass

    def __init__(self):
        self.solution_identifier = 'solution'
        self.joint_goal_identifier = 'joint_goal'
        self.joint_goal = None
        # action server
        self._action_name = 'qp_controller/command'
        self._ac = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory',
                                                FollowJointTrajectoryAction)
        # self._ac = actionlib.SimpleActionClient('/follow_joint_trajectory', FollowJointTrajectoryAction)
        # self.state_sub = rospy.Subscriber('/whole_body_controller/state', JointTrajectoryControllerState,
        #                                   self.state_cb)
        # self.state_sub = rospy.Subscriber('/fake_state', JointTrajectoryControllerState, self.state_cb)
        self._as = actionlib.SimpleActionServer(self._action_name, ControllerListAction,
                                                execute_cb=self.action_server_cb, auto_start=False)
        self.pose_array_pub = rospy.Publisher('/goals', PoseArray, queue_size=10)
        self._as.start()
        self.frequency = 100
        self.rate = rospy.Rate(self.frequency)

        print('running')
        super(ActionServer, self).__init__()

    def action_server_cb(self, goal):
        rospy.loginfo('got request')
        if goal.type != ControllerListGoal.STANDARD_CONTROLLER:
            rospy.logerr('only standard controller supported')
        else:
            controller = goal.controllers[0]
            if controller.type == Controller.JOINT:
                rospy.loginfo('received joint goal')
                mjs = OrderedDict()
                for i, joint_name in enumerate(controller.goal_state.name):
                    sjs = SingleJointState(joint_name,
                                           controller.goal_state.position[joint_name],
                                           controller.goal_state.velocity[joint_name],
                                           controller.goal_state.effort[joint_name])
                    mjs[joint_name] = sjs
                self.joint_goal = mjs
            elif controller.type == Controller.TRANSLATION_3D:
                raise NotImplementedError('cartesian goals not supported')
