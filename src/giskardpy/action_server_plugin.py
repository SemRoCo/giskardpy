from Queue import Empty
from collections import OrderedDict

import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult
from geometry_msgs.msg import PoseArray
from giskard_msgs.msg import ControllerListAction, ControllerListGoal, Controller, ControllerListResult
from multiprocessing import Queue

from giskardpy.plugin import IOPlugin
from giskardpy.trajectory import SingleJointState, Transform, Point, Quaternion


class ActionServer(IOPlugin):
    def __init__(self):
        self.joint_solution_identifier = 'joint_solution'
        self.cartesian_solution_identifier = 'cartesian_solution'
        self.joint_goal_identifier = 'joint_goal'
        self.cartesian_goal_identifier = 'cartesian_goal'
        self.joint_goal = None
        self.goal_solution = None
        self.get_readings_lock = Queue(1)
        self.update_lock = Queue(1)

        super(ActionServer, self).__init__()

    def get_readings(self):
        try:
            goal = self.get_readings_lock.get_nowait()
            if isinstance(goal, Transform):
                cartesian_goal = goal
                joint_goal = None
            else:
                cartesian_goal = None
                joint_goal = goal
        except Empty:
            cartesian_goal = None
            joint_goal = None
        update = {self.joint_goal_identifier: joint_goal,
                  self.cartesian_goal_identifier: cartesian_goal,
                  self.joint_solution_identifier: None,
                  self.cartesian_solution_identifier: None}
        return update

    def update(self):
        goal_solution = self.databus.get_data(self.joint_solution_identifier)
        if goal_solution is not None:
            self.update_lock.put(goal_solution)
            self.update_lock.get()

        goal_solution = self.databus.get_data(self.cartesian_solution_identifier)
        if goal_solution is not None:
            self.update_lock.put(goal_solution)
            self.update_lock.get()

    def start(self, databus):
        super(ActionServer, self).start(databus)
        # action server
        self._action_name = 'qp_controller/command'
        # TODO remove whole body controller and use remapping
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

        print('running')

    def stop(self):
        # TODO figure out how to stop this shit
        pass

    def action_server_cb(self, goal):
        self.cb_get_readings_part(goal)
        self.cb_update_part()

    def cb_get_readings_part(self, goal):
        rospy.loginfo('received goal')
        if goal.type != ControllerListGoal.STANDARD_CONTROLLER:
            rospy.logerr('only standard controller supported')
        else:
            # TODO add goal for each controller
            controller = goal.controllers[0]
            if controller.type == Controller.JOINT:
                rospy.loginfo('its a joint goal')
                mjs = OrderedDict()
                for i, joint_name in enumerate(controller.goal_state.name):
                    sjs = SingleJointState(joint_name,
                                           controller.goal_state.position[i],
                                           0,
                                           0)
                    mjs[joint_name] = sjs
                self.get_readings_lock.put(mjs)
            elif controller.type == Controller.TRANSLATION_3D:
                rospy.loginfo('its a cart goal')
                trans_goal = Point(controller.goal_pose.pose.position.x,
                                   controller.goal_pose.pose.position.y,
                                   controller.goal_pose.pose.position.z)
                rot_goal = Quaternion(controller.goal_pose.pose.orientation.x,
                                      controller.goal_pose.pose.orientation.y,
                                      controller.goal_pose.pose.orientation.z,
                                      controller.goal_pose.pose.orientation.w)
                goal = Transform(trans_goal, rot_goal)
                self.get_readings_lock.put(goal)
        rospy.loginfo('received goal done')

    def cb_update_part(self):
        solution = self.update_lock.get()
        rospy.loginfo('solution ready')
        success = False

        print('waiting for {:.3f} sec with {} points'.format(
            solution.trajectory.points[-1].time_from_start.to_sec(),
            len(solution.trajectory.points)))
        self._ac.send_goal(solution)
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
        rospy.loginfo('done solution ready')
        self.update_lock.put(None)
