import rospy
import control_msgs.msg
import trajectory_msgs.msg
import threading
#from trajectory_msgs.msg import JointTrajectory
import actionlib
import controller_manager_msgs.srv

class JointGoalSplitter:
    def __init__(self):
        rospy.init_node('JointGoalSplitter', anonymous=True)
        self.base_client = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory/base', control_msgs.msg.FollowJointTrajectoryAction)#/hsrb/omni_base_controller/follow_joint_trajectory
        self.arm_client = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory',
                                                        control_msgs.msg.FollowJointTrajectoryAction)

        self.base_joints = rospy.wait_for_message('/whole_body_controller/base/state', control_msgs.msg.JointTrajectoryControllerState).joint_names
        self.arm_joints = rospy.wait_for_message('/whole_body_controller/state', control_msgs.msg.JointTrajectoryControllerState).joint_names

        self._as = actionlib.SimpleActionServer('/joint_goal_splitter/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction,
                                                execute_cb=self.callback, auto_start=False)
        self._as.register_preempt_callback(self.preempt_cb())
        self._as.start()

        self.pub = rospy.Publisher('/joint_goal_splitter/state', control_msgs.msg.JointTrajectoryControllerState, queue_size=10)
        self.t = threading.Thread(target=self.state_publisher_thread)
        self.t.daemon = True
        self.t.start()

        rospy.spin()

    def __del__(self):
        self.t.join()


    def callback(self, goal):
        self.success = True
        base_ids = []
        arm_ids = []
        for joint_name in self.base_joints:
            base_ids.append(goal.trajectory.joint_names.index(joint_name))
        for joint_name in self.arm_joints:
            arm_ids.append(goal.trajectory.joint_names.index(joint_name))
        base_goal = control_msgs.msg.FollowJointTrajectoryGoal()
        arm_goal = control_msgs.msg.FollowJointTrajectoryGoal()
        base_traj = trajectory_msgs.msg.JointTrajectory()
        arm_traj = trajectory_msgs.msg.JointTrajectory()
        base_traj.joint_names = self.base_joints
        arm_traj.joint_names = self.arm_joints

        base_traj_points = []
        arm_traj_points = []
        for p in goal.trajectory.points:
            if len(p.positions) > max(base_ids): #einzeln checken
                base_traj_point = trajectory_msgs.msg.JointTrajectoryPoint()
                joint_pos = [p.positions[i] for i in base_ids]
                base_traj_point.positions = tuple(joint_pos)
                base_traj_point.time_from_start.nsecs = p.time_from_start.nsecs
                base_traj_point.time_from_start.secs = p.time_from_start.secs
                base_traj_points.append(base_traj_point)

            if len(p.positions) > max(arm_ids):
                arm_traj_point = trajectory_msgs.msg.JointTrajectoryPoint()
                joint_pos_arm = [p.positions[i] for i in arm_ids]
                arm_traj_point.positions = tuple(joint_pos_arm)
                arm_traj_point.time_from_start.nsecs = p.time_from_start.nsecs
                arm_traj_point.time_from_start.secs = p.time_from_start.secs
                arm_traj_points.append(arm_traj_point)

        #base_traj.points.append(base_traj_points[len(base_traj_points) -1])
        base_traj.points = tuple(base_traj_points)
        arm_traj.points = tuple(arm_traj_points)

        base_goal.trajectory = base_traj
        arm_goal.trajectory = arm_traj

        self.base_client.send_goal(base_goal, feedback_cb=self.feedback_cb_base)
        self.arm_client.send_goal(arm_goal, feedback_cb=self.feedback_cb_arm)

        self.base_client.wait_for_result()
        self.arm_client.wait_for_result()

        self.base_result = self.base_client.get_result()
        self.arm_result = self.arm_client.get_result()



        if self.success:
            result = self.base_result
            if not self.arm_result.error_code == control_msgs.msg.FollowJointTrajectoryResult.SUCCESSFUL:
                result = self.arm_result

            self._as.set_succeeded(result)

    def feedback_cb_base(self, feedback):
        self._as.publish_feedback(feedback)


    def feedback_cb_arm(self, feedback):
        self._as.publish_feedback(feedback)


    def preempt_cb(self):
        self.base_client.cancel_goal()
        self.arm_client.cancel_goal()
        self._as.set_preempted()
        self.success = False

    def state_publisher_thread(self):
        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            state = control_msgs.msg.JointTrajectoryControllerState()
            state.joint_names = self.arm_joints + self.base_joints
            self.pub.publish(state)
            rate.sleep()


if __name__ == '__main__':
    j = JointGoalSplitter()
