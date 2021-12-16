from threading import Thread

from py_trees import Status

from giskardpy import identifier
from giskardpy.utils.logging import logwarn
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.utils import plot_trajectory


class PlotTrajectory(GiskardBehavior):
    plot_thread: Thread

    def __init__(self, name, enabled, history, velocity_threshold, scaling, normalize_position, tick_stride, wait=False,
                 order=4):
        super(PlotTrajectory, self).__init__(name)
        self.order = order
        self.history = history
        self.velocity_threshold = velocity_threshold
        self.scaling = scaling
        self.normalize_position = normalize_position
        self.tick_stride = tick_stride
        self.wait = wait
        self.path_to_data_folder = self.get_god_map().get_data(identifier.data_folder)

    @profile
    def initialise(self):
        self.plot_thread = Thread(target=self.plot)
        self.plot_thread.start()

    def plot(self):
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        if trajectory:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            controlled_joints = list(trajectory.get_exact(0).keys())
            try:
                plot_trajectory(tj=trajectory,
                                controlled_joints=controlled_joints,
                                path_to_data_folder=self.path_to_data_folder,
                                sample_period=sample_period,
                                order=self.order,
                                velocity_threshold=self.velocity_threshold,
                                scaling=self.scaling,
                                normalize_position=self.normalize_position,
                                tick_stride=self.tick_stride,
                                history=self.history)
            except Exception as e:
                logwarn('failed to save trajectory.pdf')

    @profile
    def update(self):
        if self.wait and self.plot_thread.is_alive():
                return Status.RUNNING
        return Status.SUCCESS
