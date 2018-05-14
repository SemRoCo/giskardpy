import numpy as np
from Queue import Empty, Queue
from collections import OrderedDict, defaultdict
import pylab as plt
import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult, FollowJointTrajectoryGoal, \
    JointTrajectoryControllerState
from giskard_msgs.msg import Controller
from giskard_msgs.msg._MoveAction import MoveAction
from giskard_msgs.msg._MoveGoal import MoveGoal
from giskard_msgs.msg._MoveResult import MoveResult

from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from visualization_msgs.msg import MarkerArray

from giskardpy.exceptions import MAX_NWSR_REACHEDException, QPSolverException
from giskardpy.plugin import Plugin
from giskardpy.tfwrapper import TfWrapper
from giskardpy.trajectory import SingleJointState, Transform, Point, Quaternion, Trajectory


class ActionServerPlugin(Plugin):
    # TODO find a better name for this
    def __init__(self, cartesian_goal_identifier, js_identifier, trajectory_identifier, time_identifier,
                 collision_identifier, controlled_joints_identifier, plot_trajectory=False):
        self.plot_trajectory = plot_trajectory
        self.cartesian_goal_identifier = cartesian_goal_identifier
        self.controlled_joints_identifier = controlled_joints_identifier
        self.trajectory_identifier = trajectory_identifier
        self.js_identifier = js_identifier
        self.time_identifier = time_identifier
        self.collision_identifier = collision_identifier

        self.tf = TfWrapper()
        self.joint_goal = None
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

    def post_mortem_analysis(self, god_map, exception):
        collisions = god_map.get_data(self.collision_identifier)
        in_collision = False
        result = MoveResult()
        result.error_code = MoveResult.INSOLVABLE
        if isinstance(exception, MAX_NWSR_REACHEDException):
            result.error_code = MoveResult.MAX_NWSR_REACHED
        elif isinstance(exception, QPSolverException):
            result.error_code = MoveResult.QP_SOLVER_ERROR
        if exception is None:
            if collisions is not None:
                for (l1, l2), collision_info in collisions.items():
                    in_collision = in_collision or collision_info.contact_distance < 0.0
            if not in_collision:
                result.error_code = MoveResult.SUCCESS
                trajectory = god_map.get_data(self.trajectory_identifier)
                js = god_map.get_data(self.js_identifier)
                result.trajectory.joint_names = self.controller_joints
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
                    result.trajectory.points.append(p)
            else:
                result.error_code = MoveResult.END_STATE_COLLISION
        self.update_lock.put(result)
        self.update_lock.join()

    def get_readings(self):
        goals = None
        try:
            goal = self.get_readings_lock.get_nowait() # type: MoveGoal
            rospy.loginfo('got goal')
            self.execute = goal.type == MoveGoal.PLAN_AND_EXECUTE
            if len(goal.cmd_seq) >= 1:
                goals = defaultdict(dict)
            # TODO support multiple move cmds
            for controller in goal.cmd_seq[0].controllers:
                # TODO support collisions
                self.new_universe = True
                goal_key = str(controller.type)
                if controller.type == Controller.JOINT:
                    rospy.loginfo('got joint goal')
                    for i, joint_name in enumerate(controller.goal_state.name):
                        goals[goal_key][joint_name] = {'weight': controller.weight,
                                                       'position': controller.goal_state.position[i]}
                elif controller.type in [Controller.TRANSLATION_3D, Controller.ROTATION_3D]:
                    root = controller.root_link
                    tip = controller.tip_link
                    controller.goal_pose = self.tf.transform_pose(root, controller.goal_pose)
                    goals[goal_key][root, tip] = controller
        except Empty:
            pass
        update = {self.cartesian_goal_identifier: goals}
        return update

    def get_default_joint_goal(self):
        joint_goal = OrderedDict()
        for joint_name in sorted(self.controller_joints):
            joint_goal[joint_name] = {'weight': 1,
                                      'position': self.current_js[joint_name].position}
        return joint_goal

    def update(self):
        self.controlled_joints = self.god_map.get_data(self.controlled_joints_identifier)
        self.current_js = self.god_map.get_data(self.js_identifier)

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

    def action_server_cb(self, goal):
        self.cb_get_readings_part(goal)
        self.cb_update_part()

    def cb_get_readings_part(self, goal):
        rospy.loginfo('received goal')
        self.get_readings_lock.put(goal)

    def cb_update_part(self):
        result = self.update_lock.get() # type: MoveResult
        rospy.loginfo('solution ready')
        if result.error_code == MoveResult.SUCCESS and self.execute:
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = result.trajectory
            print('waiting for {:.3f} sec with {} points'.format(
                goal.trajectory.points[-1].time_from_start.to_sec(),
                len(goal.trajectory.points)))
            self._ac.send_goal(goal)
            t = rospy.get_rostime()
            while not self._ac.wait_for_result(rospy.Duration(.1)):
                if self._as.is_preempt_requested():
                    rospy.loginfo('new goal, cancel old one')
                    self._ac.cancel_all_goals()
                    result.error_code = MoveResult.INTERRUPTED
                    break
            else: # if not break
                print('shit took {:.3f}s'.format((rospy.get_rostime() - t).to_sec()))
                r = self._ac.get_result()
                print('real result {}'.format(r))
                if r.error_code == FollowJointTrajectoryResult.SUCCESSFUL:
                    result.error_code = MoveResult.SUCCESS

        if result.error_code != MoveResult.SUCCESS:
            self._as.set_aborted(result)
        else:
            self._as.set_succeeded(result)
        rospy.loginfo('finished movement {}'.format(result.error_code))
        self.update_lock.task_done()

    def copy(self):
        return LogTrajectoryPlugin(trajectory_identifier=self.trajectory_identifier,
                                   joint_state_identifier=self.js_identifier,
                                   time_identifier=self.time_identifier,
                                   plot_trajectory=self.plot_trajectory)

    def __del__(self):
        # TODO find a way to cancel all goals when giskard is killed
        self._ac.cancel_all_goals()


class LogTrajectoryPlugin(Plugin):
    def __init__(self, trajectory_identifier, joint_state_identifier, time_identifier, plot_trajectory=False):
        self.plot = plot_trajectory
        self.trajectory_identifier = trajectory_identifier
        self.joint_state_identifier = joint_state_identifier
        self.time_identifier = time_identifier
        self.precision = 0.0025
        super(LogTrajectoryPlugin, self).__init__()

    def get_readings(self):
        return {self.trajectory_identifier: self.trajectory}

    def update(self):
        self.trajectory = self.god_map.get_data(self.trajectory_identifier)
        time = self.god_map.get_data(self.time_identifier)
        current_js = self.god_map.get_data(self.joint_state_identifier)
        if self.trajectory is None:
            self.trajectory = Trajectory()
        self.trajectory.set(time, current_js)
        if (time >= 1 and np.abs([v.velocity for v in current_js.values()]).max() < self.precision) or time >= 20:
            print('done')
            if self.plot:
                self.plot_trajectory(self.trajectory)
            self.stop_universe = True

    def start_always(self):
        self.stop_universe = False

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
        plt.title('position')
        plt.plot(positions - positions.mean(axis=0))
        plt.show()
        plt.title('velocity')
        plt.plot(velocities)
        plt.show()
        pass
