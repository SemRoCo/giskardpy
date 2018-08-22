import numpy as np
import pylab as plt
from itertools import product


from giskardpy.exceptions import SolverTimeoutError, InsolvableException, \
    SymengineException, PathCollisionException
from giskardpy.plugin import Plugin
from giskardpy.data_types import SingleJointState, Transform, Point, Quaternion, Trajectory
from giskardpy.utils import closest_point_constraint_violated


class LogTrajectoryPlugin(Plugin):
    def __init__(self, trajectory_identifier, joint_state_identifier, time_identifier, goal_identifier,
                 closest_point_identifier,
                 controlled_joints_identifier, joint_convergence_threshold, wiggle_precision_threshold,
                 collision_time_threshold, max_traj_length,
                 plot_trajectory=False, is_preempted=lambda: False):
        # TODO use a separete plugin to trigger preempted
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
                if closest_point_constraint_violated(cp, multiplier=1):
                    self.stop_universe = True
                    raise PathCollisionException(
                        u'robot is in collision after {} seconds'.format(self.collision_time_threshold))
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
    ax1.set_title('position')
    ax2.set_title('velocity')
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
    ax1.legend(loc='center', bbox_to_anchor=(1.45, 0))

    plt.savefig('trajectory.pdf')

