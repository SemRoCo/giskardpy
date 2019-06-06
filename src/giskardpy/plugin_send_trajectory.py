import py_trees
import rospy
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState, \
    FollowJointTrajectoryResult
from py_trees_ros.actions import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import giskardpy.identifier as identifier
from giskardpy.logging import loginfo
from giskardpy.plugin import GiskardBehavior


class SendTrajectory(ActionClient, GiskardBehavior):
    error_code_to_str = {value: name for name, value in vars(FollowJointTrajectoryResult).items() if
                         isinstance(value, int)}

    def __init__(self, name, action_namespace=u'/whole_body_controller/follow_joint_trajectory'):
        GiskardBehavior.__init__(self, name)
        loginfo(u'waiting for action server \'{}\' to appear'.format(action_namespace))
        ActionClient.__init__(self, name, FollowJointTrajectoryAction, None, action_namespace)
        loginfo(u'successfully conected to action server')
        self.fill_velocity_values = self.get_god_map().safe_get_data(identifier.fill_velocity_values)

    def setup(self, timeout):
        # TODO get this from god map
        self.controller_joints = rospy.wait_for_message(u'/whole_body_controller/state',
                                                        JointTrajectoryControllerState).joint_names
        return super(SendTrajectory, self).setup(timeout)

    def initialise(self):
        super(SendTrajectory, self).initialise()
        trajectory = self.get_god_map().safe_get_data(identifier.trajectory_identifier)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = self.traj_to_msg(trajectory)
        self.action_goal = goal

    def traj_to_msg(self, trajectory):
        """
        :type traj: giskardpy.data_types.Trajectory
        :return: JointTrajectory
        """
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.stamp = rospy.get_rostime() + rospy.Duration(0.5)
        trajectory_msg.joint_names = self.controller_joints
        for time, traj_point in trajectory.items():
            p = JointTrajectoryPoint()
            p.time_from_start = rospy.Duration(time)
            for joint_name in self.controller_joints:
                if joint_name in traj_point:
                    p.positions.append(traj_point[joint_name].position)
                    if self.fill_velocity_values:
                        p.velocities.append(traj_point[joint_name].velocity)
                else:
                    raise NotImplementedError(u'generated traj does not contain all joints')
            trajectory_msg.points.append(p)
        return trajectory_msg

    def update(self):
        """
        Check only to see whether the underlying action server has
        succeeded, is running, or has cancelled/aborted for some reason and
        map these to the usual behaviour return states.

        overriding this shit because of the fucking prints
        """
        self.logger.debug("{0}.update()".format(self.__class__.__name__))
        if not self.action_client:
            self.feedback_message = "no action client, did you call setup() on your tree?"
            return py_trees.Status.INVALID
        # pity there is no 'is_connected' api like there is for c++
        if not self.sent_goal:
            self.action_client.send_goal(self.action_goal)
            self.sent_goal = True
            self.feedback_message = "sent goal to the action server"
            return py_trees.Status.RUNNING
        if self.action_client.get_state() == GoalStatus.ABORTED:
            result = self.action_client.get_result()
            self.feedback_message = self.error_code_to_str[result.error_code]
            return py_trees.Status.FAILURE
        result = self.action_client.get_result()
        if result:
            self.feedback_message = "goal reached"
            return py_trees.Status.SUCCESS
        else:
            self.feedback_message = "moving"
            return py_trees.Status.RUNNING
