#!/usr/bin/env python

import rospy
import control_msgs.msg
import trajectory_msgs.msg
try:
    import pr2_controllers_msgs.msg
except ImportError:
    pass
import actionlib
from rospy import AnyMsg

from giskardpy import logging
import copy
import rostopic


class done_cb(object):
    def __init__(self, name, action_clients, _as):
        self.name = name
        self.action_clients = action_clients
        self._as = _as

    def __call__(self, state, result):
        if result._type == 'pr2_controllers_msgs/JointTrajectoryResult':
            return
        if result.error_code != control_msgs.msg.FollowJointTrajectoryResult.SUCCESSFUL:
            for client in self.action_clients:
                client.cancel_goal()
            self._as.set_aborted(result)
            logging.logwarn(u'Joint Trajector Splitter: client \'{}\' failed to execute action goal \n {}'.format(self.name, result))

class JointTrajectorySplitter:
    def __init__(self):
        self.action_clients = []
        self.joint_names = []
        while not rospy.has_param('~state_topics'):
            logging.loginfo('waiting for param ' + '~state_topics')
            rospy.sleep(1)
        while not rospy.has_param('~client_topics'):
            logging.loginfo('waiting for param ' + '~client_topics')
            rospy.sleep(1)
        self.state_topics = rospy.get_param('~state_topics', [])
        self.client_topics = rospy.get_param('~client_topics', [])
        self.number_of_clients = len(self.state_topics)
        if self.number_of_clients != len(self.client_topics):
            logging.logerr('number of state and action topics do not match')
            exit()


        if self.number_of_clients == 0:
            logging.logerr('the state_topic or client_topic parameter is empty')
            exit()

        self.client_type = []
        for i in range(self.number_of_clients):
            waiting_for_topic = True
            while waiting_for_topic:
                try:
                    rospy.wait_for_message(self.state_topics[i], AnyMsg, timeout=10)
                    waiting_for_topic = False
                    type = rostopic.get_info_text(self.client_topics[i] + '/goal').split('\n')[0][6:]
                    self.client_type.append(type)
                except rostopic.ROSTopicException as e:
                    logging.logerr('Exception: {}'.format(e))
                    logging.logerr('unknown topic \'{}/goal\' \nmissing / in front of topic name?'.format(self.client_topics[i]))
                    exit()
                except rospy.ROSException as e:
                    if e.message == 'rospy shutdown':
                        exit()
                    logging.loginfo('waiting for state topic {}'.format(self.state_topics[i]))

        self.state_type = []
        for i in range(self.number_of_clients):
            try:
                type = rostopic.get_info_text(self.state_topics[i]).split('\n')[0][6:]
                self.state_type.append(type)
            except rostopic.ROSTopicException:
                logging.logerr(
                    'unknown topic \'{}/goal\' \nmissing / in front of topic name?'.format(
                        self.state_topics[i]))
                exit()

        for i in range(self.number_of_clients):
            if self.client_type[i] == 'control_msgs/FollowJointTrajectoryActionGoal':
                self.action_clients.append(actionlib.SimpleActionClient(self.client_topics[i], control_msgs.msg.FollowJointTrajectoryAction))
            elif self.client_type[i] == 'pr2_controllers_msgs/JointTrajectoryActionGoal':
                self.action_clients.append(actionlib.SimpleActionClient(self.client_topics[i], pr2_controllers_msgs.msg.JointTrajectoryAction))
            else:
                logging.logerr('wrong client topic type:' + self.client_type[i] + '\nmust be either control_msgs/FollowJointTrajectoryActionGoal or pr2_controllers_msgs/JointTrajectoryActionGoal')
                exit()
            self.joint_names.append(rospy.wait_for_message(self.state_topics[i], control_msgs.msg.JointTrajectoryControllerState).joint_names)
            logging.loginfo('connected to {}'.format(self.client_topics[i]))

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
                logging.logerr('wrong state topic type ' + self.state_type[i] + '\nmust be either control_msgs/JointTrajectoryControllerState or pr2_controllers_msgs/JointTrajectoryControllerState')
                exit()

        rospy.Subscriber(self.state_topics[0], control_msgs.msg.JointTrajectoryControllerState, self.state_cb_publish)

        self._as = actionlib.SimpleActionServer('/whole_body_controller/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction,
                                                execute_cb=self.callback, auto_start=False)
        self._as.register_preempt_callback(self.preempt_cb)
        self._as.start()
        logging.loginfo(u'Joint Trajector Splitter: running')
        rospy.spin()


    def callback(self, goal):
        logging.loginfo('received goal')
        self.success = True

        idx = []

        for joint_name_list in self.joint_names:
            index_list = []
            for joint_name in joint_name_list:
                try:
                    index_list.append(goal.trajectory.joint_names.index(joint_name))
                except ValueError:
                    logging.logerr('the goal does not contain the joint ' + joint_name + ' but it is published by one of the state topics')
                    result = control_msgs.msg.FollowJointTrajectoryResult()
                    result.error_code = control_msgs.msg.FollowJointTrajectoryResult.INVALID_GOAL
                    self._as.set_aborted(result)
                    return

            idx.append(index_list)

        action_goals = []
        for i in range(self.number_of_clients):
            if self.client_type[i] == 'control_msgs/FollowJointTrajectoryActionGoal':
                action_goals.append(control_msgs.msg.FollowJointTrajectoryGoal())
            else:
                action_goals.append(pr2_controllers_msgs.msg.JointTrajectoryGoal())

        goal_trajectories_points = [[] for i in range(self.number_of_clients)]

        for p in goal.trajectory.points:
            for i, index_list in enumerate(idx):
                traj_point = trajectory_msgs.msg.JointTrajectoryPoint()
                joint_pos = [p.positions[j] for j in index_list]
                traj_point.positions = tuple(joint_pos)
                if p.velocities:
                    joint_vel = [p.velocities[j] for j in index_list]
                    traj_point.velocities = tuple(joint_vel)
                if p.accelerations:
                    joint_acc = [p.accelerations[j] for j in index_list]
                    traj_point.accelerations = tuple(joint_acc)
                if p.effort:
                    joint_effort = [p.effort[j] for j in index_list]
                    traj_point.effort = tuple(joint_effort)
                traj_point.time_from_start.nsecs = p.time_from_start.nsecs
                traj_point.time_from_start.secs = p.time_from_start.secs
                goal_trajectories_points[i].append(traj_point)

        for i, a_goal in enumerate(action_goals):
            a_goal.trajectory.header = goal.trajectory.header
            a_goal.trajectory.joint_names = self.joint_names[i]
            a_goal.trajectory.points = tuple(goal_trajectories_points[i])


        logging.loginfo('send splitted goals')
        for i in range(self.number_of_clients):
            self.action_clients[i].send_goal(action_goals[i],
                                             feedback_cb=self.feedback_cb,
                                             done_cb=done_cb(self.client_topics[i], self.action_clients, self._as))

        timeout = goal.trajectory.points[-1].time_from_start + rospy.Duration(secs=2)
        for i in range(self.number_of_clients):
            start = rospy.Time.now()
            finished_before_timeout = self.action_clients[i].wait_for_result(timeout=timeout)
            now = rospy.Time.now()
            timeout = timeout - (now - start)
            if not finished_before_timeout:
                logging.logwarn("Client took to long to finish action; stopping {}".format(self.client_topics[i]))
                self.success = False
                break
            else:
                if self._as.is_active():
                    logging.loginfo('Client {} succeeded'.format(self.client_topics[i]))

        if self._as.is_active():
            if self.success:
                self._as.set_succeeded()
            else:
                self.cancel_all_goals()
                self._as.set_aborted()

    def cancel_all_goals(self):
        logging.logwarn('Canceling all goals of connected controllers')
        for client in self.action_clients:
            client.cancel_all_goals()

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
    rospy.init_node('joint_trajectory_splitter')
    j = JointTrajectorySplitter()
