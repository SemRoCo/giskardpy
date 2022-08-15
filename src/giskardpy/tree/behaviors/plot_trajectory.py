from threading import Thread

from py_trees import Status

from giskardpy import identifier
from giskardpy.utils.logging import logwarn
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.utils import plot_trajectory


class PlotTrajectory(GiskardBehavior):
    plot_thread: Thread

    @profile
    def __init__(self, name, enabled, wait=False, joint_filter=None, **kwargs):
        super().__init__(name)
        self.wait = wait
        self.kwargs = kwargs
        self.joint_filter = joint_filter
        self.path_to_data_folder = self.get_god_map().get_data(identifier.data_folder)

    @profile
    def initialise(self):
        self.plot_thread = Thread(target=self.plot)
        self.plot_thread.start()

    def plot(self):
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        if trajectory:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            if self.joint_filter is not None:
                controlled_joints = self.joint_filter
            else:
                controlled_joints = list(trajectory.get_exact(0).keys())
            try:
                plot_trajectory(tj=trajectory,
                                controlled_joints=controlled_joints,
                                path_to_data_folder=self.path_to_data_folder,
                                sample_period=sample_period,
                                diff_after=2,
                                **self.kwargs)
            except Exception as e:
                logwarn(e)
                logwarn('failed to save trajectory.pdf')

    @profile
    def update(self):
        if self.wait and self.plot_thread.is_alive():
            return Status.RUNNING
        return Status.SUCCESS
