#!/usr/bin/env python

import rospy
import control_msgs.msg
import pr2_controllers_msgs.msg
import trajectory_msgs.msg
import actionlib
from giskardpy import logging
import copy
import rostopic

class JointGoalSplitter:
    def __init__(self):
        rospy.init_node('JointGoalSplitter')
        self.action_clients = []
        self.joint_names = []
        self.state_topics = rospy.get_param('~state_topics', [])
        self.client_topics = rospy.get_param('~client_topics', [])
        self.number_of_clients = len(self.state_topics)
        if self.number_of_clients != len(self.client_topics):
            logging.logerr('Number of state and action topics do not match')
            exit()


        if self.number_of_clients == 0:
            logging.logwarn('No state/action topic found')
            exit()

        self.client_type = []
        for i in range(self.number_of_clients):
            type = rostopic.get_info_text(self.client_topics[i] + '/goal').split('\n')[0][6:]
            self.client_type.append(type)

        self.state_type = []
        for i in range(self.number_of_clients):
            type = rostopic.get_info_text(self.state_topics[i]).split('\n')[0][6:]
            self.state_type.append(type)

        for i in range(self.number_of_clients):
            if self.client_type[i] == 'control_msgs/FollowJointTrajectoryActionGoal':
                self.action_clients.append(actionlib.SimpleActionClient(self.client_topics[i], control_msgs.msg.FollowJointTrajectoryAction))
            elif self.client_type[i] == 'pr2_controllers_msgs/JointTrajectoryActionGoal':
                self.action_clients.append(actionlib.SimpleActionClient(self.client_topics[i], pr2_controllers_msgs.msg.JointTrajectoryAction))
            else:
                logging.logerr('wrong client topic type:' + self.client_type[i])
                exit()
            self.joint_names.append(rospy.wait_for_message(self.state_topics[i], control_msgs.msg.JointTrajectoryControllerState).joint_names)

        self.current_controller_state = control_msgs.msg.JointTrajectoryControllerState()
        total_number_joints = 0
        for joint_name_list in self.joint_names:
            total_number_joints += len(joint_name_list)
            self.current_controller_state.joint_names.extend(joint_name_list)


        self.current_controller_state.desired.positions = [0 for i in range(total_number_joints)]
        self.current_controller_state.desired.accelerations = [0 for i in range(total_number_joints)]
        self.current_controller_state.desired.effort = [0 for i in range(total_number_joints)]
        self.current_controller_state.desired.velocities = [0 for i in range(total_number_joints)]

        self.current_controller_state.actual = copy.deepcopy(self.current_controller_state.desired)
        self.current_controller_state.error = copy.deepcopy(self.current_controller_state.desired)

        self.state_pub = rospy.Publisher('/whole_body_controller/state', control_msgs.msg.JointTrajectoryControllerState, queue_size=10)

        for i in range(self.number_of_clients):
            if self.state_type[i] == 'control_msgs/JointTrajectoryControllerState':
                rospy.Subscriber(self.state_topics[i], control_msgs.msg.JointTrajectoryControllerState, self.state_cb_update)
            elif self.state_type[i] == 'pr2_controllers_msgs/JointTrajectoryControllerState':
                rospy.Subscriber(self.state_topics[i], pr2_controllers_msgs.msg.JointTrajectoryControllerState, self.state_cb_update)
            else:
                logging.logerr('wrong state topic type')
                exit()

        rospy.Subscriber(self.state_topics[0], control_msgs.msg.JointTrajectoryControllerState, self.state_cb_publish)

        self._as = actionlib.SimpleActionServer('/whole_body_controller/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction,
                                                execute_cb=self.callback, auto_start=False)
        self._as.register_preempt_callback(self.preempt_cb)
        self._as.start()

        rospy.spin()


    def callback(self, goal):
        logging.loginfo('received goal')
        self.success = True

        idx = []

        for joint_name_list in self.joint_names:
            index_list = []
            for joint_name in joint_name_list:
                index_list.append(goal.trajectory.joint_names.index(joint_name))

            idx.append(index_list)

        #action_goals = [control_msgs.msg.FollowJointTrajectoryGoal() for i in range(self.number_of_clients)]

        action_goals = []
        for i in range(self.number_of_clients):
            if self.client_type[i] == 'control_msgs/FollowJointTrajectoryActionGoal':
                action_goals.append(control_msgs.msg.FollowJointTrajectoryGoal())
            elif self.client_type[i] == 'pr2_controllers_msgs/JointTrajectoryActionGoal':
                action_goals.append(pr2_controllers_msgs.msg.JointTrajectoryGoal())
            else:
                logging.logerr('Wrong Action type:' + self.client_type[i])
                exit()


        goal_trajectories_points = [[] for i in range(self.number_of_clients)]

        for p in goal.trajectory.points:
            traj_points = []
            for i, index_list in enumerate(idx):
                traj_point = trajectory_msgs.msg.JointTrajectoryPoint()
                joint_pos = [p.positions[j] for j in index_list]
                traj_point.positions = tuple(joint_pos)
                traj_point.time_from_start.nsecs = p.time_from_start.nsecs
                traj_point.time_from_start.secs = p.time_from_start.secs
                goal_trajectories_points[i].append(traj_point)

        for i, a_goal in enumerate(action_goals):
            a_goal.trajectory.joint_names = self.joint_names[i]
            a_goal.trajectory.points = tuple(goal_trajectories_points[i])


        logging.loginfo('send splitted goals')
        for i in range(self.number_of_clients):
            self.action_clients[i].send_goal(action_goals[i]), self.feedback_cb

        trajectory_duration = goal.trajectory.points[-1].time_from_start + rospy.Duration(secs=2)
        for i in range(self.number_of_clients):
            self.action_clients[i].wait_for_result(timeout=trajectory_duration)


        if self.success:
            result = control_msgs.msg.FollowJointTrajectoryResult.SUCCESSFUL
            for action_client in self.action_clients:
                temp_result = action_client.get_result()
                if temp_result._type == 'pr2_controllers_msgs/JointTrajectoryResult':
                    continue
                result = temp_result
                if result.error_code != control_msgs.msg.FollowJointTrajectoryResult.SUCCESSFUL:
                    break

            self._as.set_succeeded(result)


    def feedback_cb(self, feedback):
        self._as.publish_feedback(feedback)


    def preempt_cb(self):
        for action_client in self.action_clients:
            action_client.cancel_goal()

        self._as.set_preempted()
        self.success = False

    def state_cb_update(self, state):
        self.current_controller_state.header = state.header
        for i in range(len(state.joint_names)):
            index = self.current_controller_state.joint_names.index(state.joint_names[i])
            if len(state.actual.positions) > i:
                self.current_controller_state.actual.positions[index] = state.actual.positions[i]
            if len(state.desired.positions) > i:
                self.current_controller_state.desired.positions[index] = state.desired.positions[i]
            if len(state.error.positions) > i:
                self.current_controller_state.error.positions[index] = state.error.positions[i]

            if len(state.actual.velocities) > i:
                self.current_controller_state.actual.velocities[index] = state.actual.velocities[i]
            if len(state.error.velocities) > i:
                self.current_controller_state.error.velocities[index] = state.error.velocities[i]
            if len(state.desired.velocities) > i:
                self.current_controller_state.desired.velocities[index] = state.desired.velocities[i]

            if len(state.actual.effort) > i:
                self.current_controller_state.actual.effort[index] = state.actual.effort[i]
            if len(state.error.effort) > i:
                self.current_controller_state.error.effort[index] = state.error.effort[i]
            if len(state.desired.effort) > i:
                self.current_controller_state.desired.effort[index] = state.desired.effort[i]

            if len(state.desired.accelerations) > i:
                self.current_controller_state.desired.accelerations[index] = state.desired.accelerations[i]
            if len(state.actual.accelerations) > i:
                self.current_controller_state.actual.accelerations[index] = state.actual.accelerations[i]
            if len(state.error.accelerations) > i:
                self.current_controller_state.error.accelerations[index] = state.error.accelerations[i]

    def state_cb_publish(self, state):
        self.state_pub.publish(self.current_controller_state)



if __name__ == '__main__':
    j = JointGoalSplitter()
