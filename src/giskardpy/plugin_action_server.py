import numpy as np
from Queue import Empty, Queue
from collections import OrderedDict
import pylab as plt
from itertools import product

import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult, FollowJointTrajectoryGoal, \
    JointTrajectoryControllerState
from giskard_msgs.msg import Controller
from giskard_msgs.msg import MoveCmd
from giskard_msgs.msg._MoveAction import MoveAction
from giskard_msgs.msg._MoveFeedback import MoveFeedback
from giskard_msgs.msg._MoveGoal import MoveGoal
from giskard_msgs.msg._MoveResult import MoveResult

from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

from giskardpy.exceptions import MAX_NWSR_REACHEDException, QPSolverException, SolverTimeoutError, InsolvableException, \
    SymengineException, PathCollisionException, UnknownBodyException
from giskardpy.plugin import Plugin
from giskardpy.plugin_log_trajectory import LogTrajectoryPlugin
from giskardpy.tfwrapper import transform_pose
from giskardpy.data_types import ClosestPointInfo
from giskardpy.data_types import SingleJointState, Transform, Point, Quaternion, Trajectory
from giskardpy.utils import closest_point_constraint_violated

ERROR_CODE_TO_NAME = {getattr(MoveResult, x): x for x in dir(MoveResult) if x.isupper()}

class ActionServerPlugin(Plugin):
    # TODO find a better name than ActionServerPlugin
    def __init__(self, cartesian_goal_identifier, js_identifier, trajectory_identifier, time_identifier,
                 closest_point_identifier, controlled_joints_identifier, collision_goal_identifier,
                 pyfunction_identifier, joint_convergence_threshold, wiggle_precision_threshold, fill_velocity_values,
                 collision_time_threshold, max_traj_length,
                 plot_trajectory=False):
        self.fill_velocity_values = fill_velocity_values
        self.plot_trajectory = plot_trajectory
        self.goal_identifier = cartesian_goal_identifier
        self.controlled_joints_identifier = controlled_joints_identifier
        self.trajectory_identifier = trajectory_identifier
        self.js_identifier = js_identifier
        self.time_identifier = time_identifier
        self.closest_point_identifier = closest_point_identifier
        self.collision_goal_identifier = collision_goal_identifier
        self.joint_convergence_threshold = joint_convergence_threshold
        self.wiggle_precision_threshold = wiggle_precision_threshold
        self.pyfunction_identifier = pyfunction_identifier
        self.collision_time_threshold = collision_time_threshold
        self.max_traj_length = max_traj_length

        self.joint_goal = None
        self.start_js = None
        self.goal_solution = None
        self.move_cmd_queue = Queue(1)
        self.results_queue = Queue(1)

        super(ActionServerPlugin, self).__init__()

    def start_once(self):
        self.new_universe = False
        self._action_name = u'qp_controller/command'
        # TODO remove whole body controller and use remapping
        self._ac = actionlib.SimpleActionClient(u'/whole_body_controller/follow_joint_trajectory',
                                                FollowJointTrajectoryAction)
        self._as = actionlib.SimpleActionServer(self._action_name, MoveAction,
                                                execute_cb=self.action_server_cb, auto_start=False)
        self.controller_joints = rospy.wait_for_message(u'/whole_body_controller/state',
                                                        JointTrajectoryControllerState).joint_names
        self._as.start()

    def stop(self):
        self._as = None
        self._ac = None

    def copy(self):
        self.child = LogTrajectoryPlugin(trajectory_identifier=self.trajectory_identifier,
                                         joint_state_identifier=self.js_identifier,
                                         time_identifier=self.time_identifier,
                                         plot_trajectory=self.plot_trajectory,
                                         goal_identifier=self.goal_identifier,
                                         closest_point_identifier=self.closest_point_identifier,
                                         controlled_joints_identifier=self.controlled_joints_identifier,
                                         joint_convergence_threshold=self.joint_convergence_threshold,
                                         wiggle_precision_threshold=self.wiggle_precision_threshold,
                                         is_preempted=lambda: self._as.is_preempt_requested(),
                                         collision_time_threshold=self.collision_time_threshold,
                                         max_traj_length=self.max_traj_length)
        return self.child

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
        cmd = self.get_move_cmd_from_action_server()
        if cmd is not None:
            goals = self.cmd_to_goals(cmd)
            self.new_universe = True
            self.publish_feedback(MoveFeedback.PLANNING, 0)
        self.god_map.set_data([self.goal_identifier], goals)
        # TODO create a more obvious way to modify a god map for a parallel universe
        self.god_map.set_data([self.js_identifier], self.current_js if self.start_js is None else self.start_js)
        self.god_map.set_data([self.collision_goal_identifier], cmd.collisions if cmd is not None else None)

    def cmd_to_goals(self, cmd):
        """
        :type cmd: MoveCmd
        :rtype: dict
        """
        goals = {}
        goals[str(Controller.JOINT)] = {}
        goals[str(Controller.TRANSLATION_3D)] = {}
        goals[str(Controller.ROTATION_3D)] = {}
        for controller in cmd.controllers:
            t = str(controller.type)
            if controller.type == Controller.JOINT:
                goals[t].update(self.joint_controller_to_goal(controller))
            elif controller.type == Controller.TRANSLATION_3D:
                goals[t].update(self.cart_controller_to_goal(controller))
            elif controller.type == Controller.ROTATION_3D:
                goals[t].update(self.cart_controller_to_goal(controller))
        return goals

    def joint_controller_to_goal(self, controller):
        # TODO check for unknown joint names?
        goals = {}
        rospy.loginfo(u'got joint goal')
        for i, joint_name in enumerate(controller.goal_state.name):
            goals[joint_name] = {u'weight': controller.weight,
                                 u'p_gain': controller.p_gain,
                                 u'max_speed': controller.max_speed,
                                 u'position': controller.goal_state.position[i]}
        return goals

    def cart_controller_to_goal(self, controller):
        goals = {}
        root = controller.root_link
        tip = controller.tip_link
        controller.goal_pose = transform_pose(root, controller.goal_pose)
        goals[root, tip] = controller
        return goals

    def post_mortem_analysis(self, god_map, exception):
        self.publish_feedback(MoveFeedback.PLANNING, 1)
        result = MoveResult()
        result.error_code = self.exception_to_error_code(exception)
        if result.error_code == MoveResult.SUCCESS:
            last_cp = god_map.get_data([self.closest_point_identifier])
            if not closest_point_constraint_violated(last_cp):
                result.trajectory = self.get_traj_msg(god_map)
            else:
                result.error_code = MoveResult.END_STATE_COLLISION
        # keep pyfunctions created in parallel universe
        # TODO this sucks find a better way
        self.god_map.set_data([self.pyfunction_identifier], god_map.get_data([self.pyfunction_identifier]))
        self.send_to_action_server_and_wait(result)

    def send_to_action_server_and_wait(self, result):
        self.results_queue.put(result)
        self.results_queue.join()

    def get_move_cmd_from_action_server(self):
        """
        :rtype: MoveCmd
        """
        try:
            return self.move_cmd_queue.get_nowait()
        except Empty:
            return None

    def let_process_manager_continue(self):
        self.results_queue.task_done()

    def send_to_process_manager_and_wait(self, move_cmd):
        """
        :type move_cmd: MoveCmd
        :rtype: MoveResult
        """
        self.move_cmd_queue.put(move_cmd)
        return self.results_queue.get()

    def get_traj_msg(self, god_map):
        """
        :type god_map: giskardpy.god_map.GodMap
        :rtype: JointTrajectory
        """
        trajectory_msg = JointTrajectory()
        trajectory = god_map.get_data([self.trajectory_identifier])
        self.start_js = god_map.get_data([self.js_identifier])
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
                    p.positions.append(self.start_js[joint_name].position)
                    if self.fill_velocity_values:
                        p.velocities.append(self.start_js[joint_name].velocity)
            trajectory_msg.points.append(p)
        return trajectory_msg

    def exception_to_error_code(self, exception):
        """
        :type exception: Exception
        :rtype: int
        """
        error_code = MoveResult.SUCCESS
        if self._as.is_preempt_requested():
            # TODO throw exception on preempted in order to get rid of if?
            error_code = MoveResult.INTERRUPTED
        elif isinstance(exception, MAX_NWSR_REACHEDException):
            error_code = MoveResult.MAX_NWSR_REACHED
        elif isinstance(exception, QPSolverException):
            error_code = MoveResult.QP_SOLVER_ERROR
        elif isinstance(exception, UnknownBodyException):
            error_code = MoveResult.UNKNOWN_OBJECT
        elif isinstance(exception, SolverTimeoutError):
            error_code = MoveResult.SOLVER_TIMEOUT
        elif isinstance(exception, InsolvableException):
            error_code = MoveResult.INSOLVABLE
        elif isinstance(exception, SymengineException):
            error_code = MoveResult.SYMENGINE_ERROR
        elif isinstance(exception, PathCollisionException):
            error_code = MoveResult.PATH_COLLISION
        return error_code

    def action_server_cb(self, goal):
        """
        :param goal:
        :type goal: MoveGoal
        """
        rospy.loginfo(u'goal received')
        self.execute = goal.type == MoveGoal.PLAN_AND_EXECUTE
        if goal.type == MoveGoal.UNDEFINED:
            result = MoveResult()
            # TODO new error code
            result.error_code = MoveResult.INSOLVABLE
            self._as.set_aborted(result)
        else:
            result = MoveResult()
            for i, move_cmd in enumerate(goal.cmd_seq):  # type: (int, MoveCmd)
                # TODO handle empty controller case
                intermediate_result = self.send_to_process_manager_and_wait(move_cmd)
                result.error_code = intermediate_result.error_code
                if result.error_code != MoveResult.SUCCESS:
                    # clear traj from prev cmds
                    result.trajectory = JointTrajectory()
                    break
                result.trajectory = self.append_trajectory(result.trajectory, intermediate_result.trajectory)
                if i < len(goal.cmd_seq) - 1:
                    self.let_process_manager_continue()
            else:  # if not break
                rospy.loginfo(u'found solution')
                if result.error_code == MoveResult.SUCCESS and self.execute:
                    result.error_code = self.send_to_robot(result)

            self.start_js = None
            if result.error_code != MoveResult.SUCCESS:
                self._as.set_aborted(result)
            else:
                self._as.set_succeeded(result)
            self.let_process_manager_continue()
        rospy.loginfo(u'goal result: {}'.format(ERROR_CODE_TO_NAME[result.error_code]))

    def send_to_robot(self, result):
        """
        :type result: MoveResult
        :return: error code from MoveResult
        :rtype: int
        """
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = result.trajectory
        if self._as.is_preempt_requested():
            rospy.loginfo(u'new goal, cancel old one')
            self._ac.cancel_all_goals()
            error_code = MoveResult.INTERRUPTED
        else:
            self._ac.send_goal(goal)
            expected_duration = goal.trajectory.points[-1].time_from_start.to_sec()
            rospy.loginfo(u'waiting for {:.3f} sec with {} points'.format(expected_duration,
                                                                          len(goal.trajectory.points)))
            error_code = self.wait_for_result_and_feed_back_feedback(expected_duration)
        return error_code

    def append_trajectory(self, traj1, traj2):
        """
        :type traj1: JointTrajectory
        :type traj2: JointTrajectory
        :rtype: JointTrajectory
        """
        # FIXME probably overwrite traj1
        if len(traj1.points) == 0:
            return traj2
        # FIXME this step size assume a fixed distance between traj points
        step_size = traj1.points[1].time_from_start - \
                    traj1.points[0].time_from_start
        end_of_last_point = traj1.points[-1].time_from_start + step_size
        for point in traj2.points:  # type: JointTrajectoryPoint
            point.time_from_start += end_of_last_point
            traj1.points.append(point)
        return traj1

    def publish_feedback(self, phase, progress):
        feedback = MoveFeedback()
        feedback.phase = phase
        feedback.progress = progress
        self._as.publish_feedback(feedback)

    def wait_for_result_and_feed_back_feedback(self, expected_duration):
        """
        :type expected_duration: float
        :return: error code from MoveResult
        :rtype: int
        """
        t = rospy.get_rostime()
        phase = MoveFeedback.EXECUTION
        error_code = MoveResult.SUCCESS
        while not self._ac.wait_for_result(rospy.Duration(.1)):
            time_passed = (rospy.get_rostime() - t).to_sec()
            self.publish_feedback(phase, min(time_passed / expected_duration, 1))

            if self._as.is_preempt_requested():
                rospy.loginfo(u'new goal, cancel old one')
                self._ac.cancel_all_goals()
                error_code = MoveResult.INTERRUPTED
                break
            if time_passed > expected_duration + 0.1:  # TODO new error code
                rospy.loginfo(u'controller took too long to execute trajectory')
                self._ac.cancel_all_goals()
                error_code = MoveResult.INTERRUPTED
                break
        else:  # if not break
            print('shit took {:.3f}s'.format((rospy.get_rostime() - t).to_sec()))
            r = self._ac.get_result()
            if r.error_code == FollowJointTrajectoryResult.SUCCESSFUL:
                error_code = MoveResult.SUCCESS
        return error_code

    def __del__(self):
        # TODO find a way to cancel all goals when giskard is killed
        self._ac.cancel_all_goals()
