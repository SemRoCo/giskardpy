import numpy as np
from Queue import Empty, Queue
from collections import OrderedDict, defaultdict
import pylab as plt
from itertools import combinations, product

import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult, FollowJointTrajectoryGoal, \
    JointTrajectoryControllerState
from giskard_msgs.msg import Controller, CollisionEntry
from giskard_msgs.msg import MoveCmd
from giskard_msgs.msg._MoveAction import MoveAction
from giskard_msgs.msg._MoveFeedback import MoveFeedback
from giskard_msgs.msg._MoveGoal import MoveGoal
from giskard_msgs.msg._MoveResult import MoveResult

from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from visualization_msgs.msg import MarkerArray

from giskardpy.exceptions import MAX_NWSR_REACHEDException, QPSolverException, SolverTimeoutError, \
    IntersectingCollisionException, InsolvableException
from giskardpy.plugin import Plugin
from giskardpy.tfwrapper import transform_pose
from giskardpy.trajectory import ClosestPointInfo
from giskardpy.trajectory import SingleJointState, Transform, Point, Quaternion, Trajectory


class ActionServerPlugin(Plugin):
    # TODO find a better name for this
    def __init__(self, cartesian_goal_identifier, js_identifier, trajectory_identifier, time_identifier,
                 closest_point_identifier, controlled_joints_identifier, collision_goal_identifier,
                 plot_trajectory=False):
        self.plot_trajectory = plot_trajectory
        self.goal_identifier = cartesian_goal_identifier
        self.controlled_joints_identifier = controlled_joints_identifier
        self.trajectory_identifier = trajectory_identifier
        self.js_identifier = js_identifier
        self.time_identifier = time_identifier
        self.closest_point_identifier = closest_point_identifier
        self.collision_goal_identifier = collision_goal_identifier

        self.joint_goal = None
        self.start_js = None
        self.goal_solution = None
        self.get_readings_lock = Queue(1)
        self.update_lock = Queue(1)

        super(ActionServerPlugin, self).__init__()

    def create_parallel_universe(self):
        muh = self.new_universe
        self.new_universe = False
        return muh

    def end_parallel_universe(self):
        return super(ActionServerPlugin, self).end_parallel_universe()

    def update(self):
        self.controlled_joints = self.god_map.get_data([self.controlled_joints_identifier])
        self.current_js = self.god_map.get_data([self.js_identifier])
        goals = None
        cmd = None
        try:
            cmd = self.get_readings_lock.get_nowait()  # type: MoveCmd
            rospy.loginfo('got goal')
            goals = {}
            goals[str(Controller.JOINT)] = {}
            goals[str(Controller.TRANSLATION_3D)] = {}
            goals[str(Controller.ROTATION_3D)] = {}
            # goals['max_trajectory_length'] = cmd.max_trajectory_length
            # TODO support multiple move cmds
            for controller in cmd.controllers:
                # TODO support collisions
                self.new_universe = True
                goal_key = str(controller.type)
                if controller.type == Controller.JOINT:
                    # TODO check for unknown joint names
                    rospy.loginfo('got joint goal')
                    for i, joint_name in enumerate(controller.goal_state.name):
                        goals[goal_key][joint_name] = {'weight': controller.weight,
                                                       'p_gain' : controller.p_gain,
                                                       'max_speed': controller.max_speed,
                                                       'position': controller.goal_state.position[i]}
                elif controller.type in [Controller.TRANSLATION_3D, Controller.ROTATION_3D]:
                    root = controller.root_link
                    tip = controller.tip_link
                    controller.goal_pose = transform_pose(root, controller.goal_pose)
                    goals[goal_key][root, tip] = controller
            feedback = MoveFeedback()
            feedback.phase = MoveFeedback.PLANNING
            self._as.publish_feedback(feedback)
        except Empty:
            pass
        self.god_map.set_data([self.goal_identifier], goals)
        self.god_map.set_data([self.js_identifier], self.current_js if self.start_js is None else self.start_js)
        self.god_map.set_data([self.collision_goal_identifier], cmd.collisions if cmd is not None else None)

    def post_mortem_analysis(self, god_map, exception):
        result = MoveResult()
        result.error_code = MoveResult.INSOLVABLE
        if isinstance(exception, MAX_NWSR_REACHEDException):
            result.error_code = MoveResult.MAX_NWSR_REACHED
        elif isinstance(exception, QPSolverException):
            result.error_code = MoveResult.QP_SOLVER_ERROR
        elif isinstance(exception, KeyError):
            result.error_code = MoveResult.UNKNOWN_OBJECT
        elif isinstance(exception, SolverTimeoutError):
            result.error_code = MoveResult.SOLVER_TIMEOUT
        elif isinstance(exception, InsolvableException):
            result.error_code = MoveResult.INSOLVABLE
        if exception is None:
            if not self.closest_point_constraint_violated(god_map):
                result.error_code = MoveResult.SUCCESS
                trajectory = god_map.get_data([self.trajectory_identifier])
                self.start_js = god_map.get_data([self.js_identifier])
                result.trajectory.joint_names = self.controller_joints
                for time, traj_point in trajectory.items():
                    p = JointTrajectoryPoint()
                    p.time_from_start = rospy.Duration(time)
                    for joint_name in self.controller_joints:
                        if joint_name in traj_point:
                            p.positions.append(traj_point[joint_name].position)
                            p.velocities.append(traj_point[joint_name].velocity)
                        else:
                            p.positions.append(self.start_js[joint_name].position)
                            p.velocities.append(self.start_js[joint_name].velocity)
                    result.trajectory.points.append(p)
            else:
                result.error_code = MoveResult.END_STATE_COLLISION
        self.update_lock.put(result)
        self.update_lock.join()

    def closest_point_constraint_violated(self, god_map):
        cp = god_map.get_data([self.closest_point_identifier])
        for link_name, cpi_info in cp.items():  # type: (str, ClosestPointInfo)
            if cpi_info.contact_distance < cpi_info.min_dist * 0.9:
                print(cpi_info.link_a, cpi_info.link_b, cpi_info.contact_distance)
                return True
        return False

    def action_server_cb(self, goal):
        """
        :param goal:
        :type goal: MoveGoal
        """
        rospy.loginfo('received goal')
        self.execute = goal.type == MoveGoal.PLAN_AND_EXECUTE
        # TODO do we really want to check for start state collision?
        if True or not self.closest_point_constraint_violated(self.god_map):
            result = None
            for i, move_cmd in enumerate(goal.cmd_seq):
                # TODO handle empty controller case
                self.get_readings_lock.put(move_cmd)
                intermediate_result = self.update_lock.get()  # type: MoveResult
                if intermediate_result.error_code != MoveResult.SUCCESS:
                    result = intermediate_result
                    break
                if result is None:
                    result = intermediate_result
                else:
                    step_size = result.trajectory.points[1].time_from_start - \
                                result.trajectory.points[0].time_from_start
                    end_of_last_point = result.trajectory.points[-1].time_from_start + step_size
                    for point in intermediate_result.trajectory.points:  # type: JointTrajectoryPoint
                        point.time_from_start += end_of_last_point
                        result.trajectory.points.append(point)
                if i < len(goal.cmd_seq) - 1:
                    self.update_lock.task_done()
            else:  # if not break
                rospy.loginfo('solution ready')
                feedback = MoveFeedback()
                feedback.phase = MoveFeedback.EXECUTION
                if result.error_code == MoveResult.SUCCESS and self.execute:
                    goal = FollowJointTrajectoryGoal()
                    goal.trajectory = result.trajectory
                    if self._as.is_preempt_requested():
                        rospy.loginfo('new goal, cancel old one')
                        self._ac.cancel_all_goals()
                        result.error_code = MoveResult.INTERRUPTED
                    else:
                        self._ac.send_goal(goal)
                        t = rospy.get_rostime()
                        expected_duration = goal.trajectory.points[-1].time_from_start.to_sec()
                        rospy.loginfo('waiting for {:.3f} sec with {} points'.format(expected_duration,
                                                                                     len(goal.trajectory.points)))

                        while not self._ac.wait_for_result(rospy.Duration(.1)):
                            time_passed = (rospy.get_rostime() - t).to_sec()
                            feedback.progress = min(time_passed / expected_duration, 1)
                            self._as.publish_feedback(feedback)
                            if self._as.is_preempt_requested():
                                rospy.loginfo('new goal, cancel old one')
                                self._ac.cancel_all_goals()
                                result.error_code = MoveResult.INTERRUPTED
                                break
                            if time_passed > expected_duration + 0.1:
                                rospy.loginfo('controller took too long to execute trajectory')
                                self._ac.cancel_all_goals()
                                result.error_code = MoveResult.INTERRUPTED
                                break
                        else:  # if not break
                            print('shit took {:.3f}s'.format((rospy.get_rostime() - t).to_sec()))
                            r = self._ac.get_result()
                            if r.error_code == FollowJointTrajectoryResult.SUCCESSFUL:
                                result.error_code = MoveResult.SUCCESS
        else:
            result = MoveResult()
            result.error_code = MoveResult.START_STATE_COLLISION
        self.start_js = None
        if result.error_code != MoveResult.SUCCESS:
            self._as.set_aborted(result)
        else:
            self._as.set_succeeded(result)
        rospy.loginfo('finished movement {}'.format(result.error_code))
        try:
            self.update_lock.task_done()
        except ValueError:
            pass

    def get_default_joint_goal(self):
        joint_goal = OrderedDict()
        for joint_name in sorted(self.controller_joints):
            joint_goal[joint_name] = {'weight': 1,
                                      'position': self.current_js[joint_name].position}
        return joint_goal

    def start_once(self):
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
        self._as = actionlib.SimpleActionServer(self._action_name, MoveAction,
                                                execute_cb=self.action_server_cb, auto_start=False)
        self.controller_joints = rospy.wait_for_message('/whole_body_controller/state',
                                                        JointTrajectoryControllerState).joint_names
        self._as.start()

        print('running')

    def stop(self):
        # TODO figure out how to stop this shit
        pass

    def copy(self):
        self.child = LogTrajectoryPlugin(trajectory_identifier=self.trajectory_identifier,
                                         joint_state_identifier=self.js_identifier,
                                         time_identifier=self.time_identifier,
                                         plot_trajectory=self.plot_trajectory,
                                         goal_identifier=self.goal_identifier,
                                         controlled_joints_identifier=self.controlled_joints_identifier,
                                         is_preempted=lambda: self._as.is_preempt_requested())
        return self.child

    def __del__(self):
        # TODO find a way to cancel all goals when giskard is killed
        self._ac.cancel_all_goals()


class LogTrajectoryPlugin(Plugin):
    def __init__(self, trajectory_identifier, joint_state_identifier, time_identifier, goal_identifier,
                 controlled_joints_identifier, plot_trajectory=False, is_preempted=lambda: False):
        self.plot = plot_trajectory
        self.controlled_joints_identifier = controlled_joints_identifier
        self.goal_identifier = goal_identifier
        self.trajectory_identifier = trajectory_identifier
        self.joint_state_identifier = joint_state_identifier
        self.time_identifier = time_identifier
        self.is_preempted = is_preempted
        self.precision = 0.005
        self.wiggle_precision = 5
        super(LogTrajectoryPlugin, self).__init__()

    def simplify_js(self, js):
        return tuple(round(x.position, self.wiggle_precision) for x in js.values())

    def update(self):
        current_js = self.god_map.get_data([self.joint_state_identifier])
        time = self.god_map.get_data([self.time_identifier])
        trajectory = self.god_map.get_data([self.trajectory_identifier])
        # traj_length = self.god_map.get_data([self.goal_identifier, 'max_trajectory_length'])
        rounded_js = self.simplify_js(current_js)
        if trajectory is None:
            trajectory = Trajectory()
        trajectory.set(time, current_js)
        self.god_map.set_data([self.trajectory_identifier], trajectory)

        if self.is_preempted():
            print('goal preempted')
            self.stop_universe = True
            return
        if time >= 1:
            if np.abs([v.velocity for v in current_js.values()]).max() < self.precision:
                print('done')
                if self.plot:
                    plot_trajectory(trajectory, set(self.god_map.get_data([self.controlled_joints_identifier])))
                self.stop_universe = True
                return
            if rounded_js in self.past_joint_states:
                self.stop_universe = True
                raise InsolvableException('endless wiggling detected')
        self.past_joint_states.add(rounded_js)

    def start_always(self):
        self.stop_universe = False
        self.past_joint_states = set()

    def stop(self):
        pass

    def end_parallel_universe(self):
        return self.stop_universe

def plot_trajectory(tj, controlled_joints):
    """
    :param tj:
    :type tj: Trajectory
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    line_styles = ['', '--', '-.']
    fmts = [''.join(x) for x in product(line_styles, colors)]
    positions = []
    velocities = []
    names = [x for x in tj._points[0.0].keys() if x in controlled_joints]
    for time, point in tj.items():
        positions.append([v.position for j, v in point.items() if j in controlled_joints])
        velocities.append([v.velocity for j, v in point.items() if j in controlled_joints])
    positions = np.array(positions)
    velocities = np.array(velocities)

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_title('position')
    ax2.set_title('velocity')
    positions -= positions.mean(axis=0)
    for i, position in enumerate(positions.T):
        ax1.plot(position, fmts[i], label=names[i])
        ax2.plot(velocities[:,i], fmts[i])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc='center', bbox_to_anchor=(1.45, 0))

    plt.show()

def plot_trajectory2(tj):
    """
    :param tj:
    :type tj: Trajectory
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    line_styles = ['', '--', '-.']
    fmts = [''.join(x) for x in product(line_styles, colors)]
    positions = []
    velocities = []
    time = []
    names = tj.joint_names
    for point in tj.points:
        positions.append(point.positions)
        velocities.append(point.velocities)
        time.append(point.time_from_start)
    positions = np.array(positions)
    velocities = np.array(velocities).T
    time = np.array([x.to_sec() for x in time])

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_title('position')
    ax2.set_title('velocity')
    positions -= positions.mean(axis=0)
    for i, position in enumerate(positions.T):
        ax1.plot(time, position, fmts[i], label=names[i])
        ax2.plot(time, velocities[i], fmts[i])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc='center', bbox_to_anchor=(1.45, 0))

    plt.show()
