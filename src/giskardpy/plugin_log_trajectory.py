import numpy as np
import pylab as plt
from itertools import product
from giskardpy.plugin import NewPluginBase
from giskardpy.data_types import Trajectory


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
        self.trajectory = Trajectory()
        self.god_map.safe_set_data([self.trajectory_identifier], self.trajectory)
        super(NewLogTrajPlugin, self).initialize()

    def update(self):
        current_js = self.god_map.safe_get_data([self.joint_state_identifier])
        time = self.god_map.safe_get_data([self.time_identifier])
        trajectory = self.god_map.safe_get_data([self.trajectory_identifier])
        trajectory.set(time, current_js)
        self.god_map.safe_set_data([self.trajectory_identifier], trajectory)

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
