from py_trees import Status

from giskardpy import identifier
from giskardpy.data_types import Trajectory
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import trajectory_to_np
import numpy as np
import matplotlib.pyplot as plt

class PlotTrajectoryFFT(GiskardBehavior):
    def __init__(self, name, joint_name):
        self.joint_name = joint_name
        super(PlotTrajectoryFFT, self).__init__(name)

    def initialise(self):
        self.path_to_data_folder = self.get_god_map().get_data(identifier.data_folder)

    def update(self):
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        if trajectory:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            controlled_joints = self.get_robot().controlled_joints
            plot_fft(trajectory, controlled_joints, self.path_to_data_folder, sample_period, self.joint_name)
        return Status.SUCCESS

def plot_fft(tj, controlled_joints, path_to_data_folder, sample_period, joint_name):
    """
    :type tj: Trajectory
    :param controlled_joints: only joints in this list will be added to the plot
    :type controlled_joints: list
    """
    plt.clf()
    names, position, velocity, times = trajectory_to_np(tj, controlled_joints)
    joint_index = names.index(joint_name)
    position = position[:,joint_index]
    velocity = velocity[:,joint_index]
    velocity -= np.average(velocity)

    freq = np.fft.rfftfreq(len(velocity), d=sample_period)

    fft = np.fft.rfft(velocity)
    plt.plot(freq, np.abs(fft.real), label=u'real')
    plt.plot(freq, np.abs(fft.imag), label=u'img')

    plt.grid()
    plt.xlabel(u'hz')
    plt.ylabel(u'amp')
    plt.legend()
    plt.savefig(path_to_data_folder + u'{}_fft.pdf'.format(joint_name))
