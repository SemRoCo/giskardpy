import numpy as np
import pylab as plt
from itertools import product

from giskardpy.exceptions import SolverTimeoutError, InsolvableException, \
    SymengineException, PathCollisionException
from giskardpy.plugin import PluginBase, NewPluginBase
from giskardpy.data_types import SingleJointState, Transform, Point, Quaternion, Trajectory
from giskardpy.utils import closest_point_constraint_violated


class LogTrajectoryPlugin(PluginBase):
    """
    Keeps track of the joint trajectory and stores it in the god map.
    """

    # TODO another plugin should trigger the end of the universe
    # TODO expects time to start at zero, get rid of this assumption
    # TODO user should be able to specify joint convergence threshold per joint
    # TODO add parameter for location where debug plots should be saved
    def __init__(self, trajectory_identifier, joint_state_identifier, time_identifier, goal_identifier,
                 closest_point_identifier,
                 controlled_joints_identifier, joint_convergence_threshold, wiggle_precision_threshold,
                 collision_time_threshold, max_traj_length,
                 plot_trajectory=False, is_preempted=lambda: False):
        """
        :type trajectory_identifier: str
        :type joint_state_identifier: str
        :type time_identifier: str
        :type goal_identifier: str
        :type closest_point_identifier: str
        :type controlled_joints_identifier: str
        :param joint_convergence_threshold: if the maximum joint velocity falls below this value, the current universe is killed
        :type joint_convergence_threshold: float
        :param wiggle_precision_threshold: rounds joint states to this many decimal places and stops the universe if a joint state is seen twice
        :type wiggle_precision_threshold: float
        :param collision_time_threshold: if the robot is in collision after this many s, it is assumed, that it can't get out and the univserse is killed
        :type collision_time_threshold: float
        :param max_traj_length: if no traj can be found that takes less than this many s to execute, the planning is stopped.
        :type max_traj_length: float
        :param plot_trajectory: saves a plot of the joint traj for debugging.
        :type plot_trajectory: bool
        :param is_preempted: if this function evaluates to True, the planning is stopped
        """
        self.plot = plot_trajectory
        self.closest_point_identifier = closest_point_identifier
        self.controlled_joints_identifier = controlled_joints_identifier
        self.goal_identifier = goal_identifier
        self.trajectory_identifier = trajectory_identifier
        self.joint_state_identifier = joint_state_identifier
        self.time_identifier = time_identifier
        self.is_preempted = is_preempted
        self.precision = joint_convergence_threshold
        self.max_traj_length = max_traj_length
        self.collision_time_threshold = collision_time_threshold
        self.wiggle_precision = wiggle_precision_threshold
        super(LogTrajectoryPlugin, self).__init__()

    def round_js(self, js):
        """
        :param js: joint_name -> SingleJointState
        :type js: dict
        :return: a sequence of all the rounded joint positions
        :rtype: tuple
        """
        return tuple(round(x.position, self.wiggle_precision) for x in js.values())

    def update(self):
        current_js = self.god_map.get_data([self.joint_state_identifier])
        time = self.god_map.get_data([self.time_identifier])
        trajectory = self.god_map.get_data([self.trajectory_identifier])
        # traj_length = self.god_map.get_data([self.goal_identifier, 'max_trajectory_length'])
        rounded_js = self.round_js(current_js)
        if trajectory is None:
            trajectory = Trajectory()
        trajectory.set(time, current_js)
        self.god_map.set_data([self.trajectory_identifier], trajectory)

        if self.is_preempted():
            print(u'goal preempted')
            self.stop_universe = True
            return
        if time >= 1:
            if time > self.max_traj_length:
                self.stop_universe = True
                raise SolverTimeoutError(u'didn\'t a solution after {} s'.format(self.max_traj_length))
            if np.abs([v.velocity for v in current_js.values()]).max() < self.precision or \
                    (self.plot and time > self.max_traj_length):
                print(u'done')
                if self.plot:
                    plot_trajectory(trajectory, set(self.god_map.get_data([self.controlled_joints_identifier])))
                self.stop_universe = True
                return
            if not self.plot and (rounded_js in self.past_joint_states):
                self.stop_universe = True
                raise InsolvableException(u'endless wiggling detected')
            if time >= self.collision_time_threshold:
                cp = self.god_map.get_data([self.closest_point_identifier])
                if closest_point_constraint_violated(cp, tolerance=1):
                    self.stop_universe = True
                    raise PathCollisionException(
                        u'robot is in collision after {} seconds'.format(self.collision_time_threshold))
        self.past_joint_states.add(rounded_js)

    def initialize(self):
        self.stop_universe = False
        self.past_joint_states = set()

    def stop(self):
        pass

    def end_parallel_universe(self):
        return self.stop_universe

class NewLogTrajPlugin(NewPluginBase):
    # TODO split log and interrupt conditions
    def __init__(self, trajectory_identifier, joint_state_identifier, time_identifier):
        """
        :type trajectory_identifier: str
        :type joint_state_identifier: str
        :type time_identifier: str
        """
        self.trajectory_identifier = trajectory_identifier
        self.joint_state_identifier = joint_state_identifier
        self.time_identifier = time_identifier
        super(NewLogTrajPlugin, self).__init__()

    def setup(self):
        super(NewLogTrajPlugin, self).setup()

    def initialize(self):
        self.stop_universe = False
        self.past_joint_states = set()
        super(NewLogTrajPlugin, self).initialize()

    def update(self):
        current_js = self.god_map.get_data([self.joint_state_identifier])
        time = self.god_map.get_data([self.time_identifier])
        trajectory = self.god_map.get_data([self.trajectory_identifier])
        if trajectory is None:
            trajectory = Trajectory()
        trajectory.set(time, current_js)
        self.god_map.set_data([self.trajectory_identifier], trajectory)

        return super(NewLogTrajPlugin, self).update()


def plot_trajectory(tj, controlled_joints):
    """
    :type tj: Trajectory
    :param controlled_joints: only joints in this list will be added to the plot
    :type controlled_joints: list
    """
    colors = [u'b', u'g', u'r', u'c', u'm', u'y', u'k']
    line_styles = [u'', u'--', u'-.']
    fmts = [u''.join(x) for x in product(line_styles, colors)]
    positions = []
    velocities = []
    times = []
    names = [x for x in tj._points[0.0].keys() if x in controlled_joints]
    for time, point in tj.items():
        positions.append([v.position for j, v in point.items() if j in controlled_joints])
        velocities.append([v.velocity for j, v in point.items() if j in controlled_joints])
        times.append(time)
    positions = np.array(positions)
    velocities = np.array(velocities).T
    times = np.array(times)

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_title(u'position')
    ax2.set_title(u'velocity')
    # positions -= positions.mean(axis=0)
    for i, position in enumerate(positions.T):
        ax1.plot(times, position, fmts[i], label=names[i])
        ax2.plot(times, velocities[i], fmts[i])
    box = ax1.get_position()
    ax1.set_ylim(-3, 1)
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc=u'center', bbox_to_anchor=(1.45, 0))

    plt.savefig(u'trajectory.pdf')
