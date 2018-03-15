import numpy as np
from Queue import Empty
from collections import OrderedDict
import pylab as plt
import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult, FollowJointTrajectoryGoal, \
    JointTrajectoryControllerState
from giskard_msgs.msg import ControllerListAction, ControllerListGoal, Controller, ControllerListResult
from multiprocessing import Queue

from trajectory_msgs.msg import JointTrajectoryPoint

from giskardpy.plugin import Plugin
from giskardpy.trajectory import SingleJointState, Transform, Point, Quaternion, Trajectory


class ActionServer(Plugin):
    # TODO find a better name for this
    def __init__(self, js_identifier='js', trajectory_identifier='traj', cartesian_goal_identifier='cartesian_goal',
                 joint_goal_identifier='joint_goal', time_identifier='time', collision_identifier='collision'):
        self.joint_goal_identifier = joint_goal_identifier
        self.cartesian_goal_identifier = cartesian_goal_identifier
        self.trajectory_identifier = trajectory_identifier
        self.js_identifier = js_identifier
        self.time_identifier = time_identifier
        self.collision_identifier = collision_identifier

        self.joint_goal = None
        self.goal_solution = None
        self.get_readings_lock = Queue(1)
        self.update_lock = Queue(1)

        super(ActionServer, self).__init__()

    def create_parallel_universe(self):
        muh = self.new_universe
        self.new_universe = False
        return muh

    def end_parallel_universe(self):
        return super(ActionServer, self).end_parallel_universe()

    def post_mortem_analysis(self, god_map):
        collisions = god_map.get_data(self.collision_identifier)
        if len(collisions) == 0 or collisions is None:
            trajectory = god_map.get_data(self.trajectory_identifier)
            js = god_map.get_data(self.js_identifier)
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = self.controller_joints
            for time, traj_point in trajectory.items():
                p = JointTrajectoryPoint()
                p.time_from_start = rospy.Duration(time)
                for joint_name in self.controller_joints:
                    if joint_name in traj_point:
                        p.positions.append(traj_point[joint_name].position)
                        p.velocities.append(traj_point[joint_name].velocity)
                    else:
                        p.positions.append(js[joint_name].position)
                        p.velocities.append(js[joint_name].velocity)
                goal.trajectory.points.append(p)
            self.update_lock.put(goal)
        else:
            print('collision during rollout!!!!!!!!!!!!!!!!!!!')
            self.update_lock.put(None)
        self.update_lock.get()

    def get_readings(self):
        try:
            goal = self.get_readings_lock.get_nowait()
            if isinstance(goal, Transform):
                self.new_universe = True
                cartesian_goal = goal
                joint_goal = None
            else:
                self.new_universe = True
                cartesian_goal = None
                joint_goal = goal
        except Empty:
            cartesian_goal = None
            joint_goal = None
        update = {self.joint_goal_identifier: joint_goal,
                  self.cartesian_goal_identifier: cartesian_goal}
        return update

    def update(self):
        pass

    def start(self, god_map):
        super(ActionServer, self).start(god_map)
        self.new_universe = False
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
        self.controller_joints = rospy.wait_for_message('/whole_body_controller/state',
                                                        JointTrajectoryControllerState).joint_names
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
                # TODO don't ignore root and tip
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

    def cb_update_part(self):
        solution = self.update_lock.get()
        rospy.loginfo('solution ready')
        success = False
        if solution is not None:
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
        else:
            print('no solution found')

        if success:
            self._as.set_succeeded(ControllerListResult())
            print('success')
        else:
            self._as.set_aborted(ControllerListResult())
        rospy.loginfo('finished movement')
        self.update_lock.put(None)

    def get_replacement_parallel_universe(self):
        return LogTrajectory(trajectory_identifier=self.trajectory_identifier,
                             joint_state_identifier=self.js_identifier,
                             time_identifier=self.time_identifier)


class LogTrajectory(Plugin):
    def __init__(self, trajectory_identifier='traj', joint_state_identifier='js', time_identifier='time'):
        self.trajectory_identifier = trajectory_identifier
        self.joint_state_identifier = joint_state_identifier
        self.time_identifier = time_identifier
        self.precision = 0.0025
        super(LogTrajectory, self).__init__()

    def get_readings(self):
        return {self.trajectory_identifier: self.trajectory}

    def update(self):
        self.trajectory = self.god_map.get_data(self.trajectory_identifier)
        time = self.god_map.get_data(self.time_identifier)
        current_js = self.god_map.get_data(self.joint_state_identifier)
        if self.trajectory is None:
            self.trajectory = Trajectory()
        self.trajectory.set(time, current_js)
        if (time >= 1 and np.abs([v.velocity for v in current_js.values()]).max() < self.precision) or time >= 10:
            print('done')
            # self.plot_trajectory(self.trajectory)
            self.stop_universe = True

    def start(self, god_map):
        self.stop_universe = False
        super(LogTrajectory, self).start(god_map)

    def stop(self):
        pass

    def end_parallel_universe(self):
        return self.stop_universe

    def plot_trajectory(self, tj):
        positions = []
        velocities = []
        for time, point in tj.items():
            positions.append([v.position for v in point.values()])
            velocities.append([v.velocity for v in point.values()])
        positions = np.array(positions)
        velocities = np.array(velocities)
        plt.plot(positions - positions.mean(axis=0))
        plt.show()
        plt.plot(velocities)
        plt.show()
        pass
