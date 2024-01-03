from threading import Thread

from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy.utils.logging import logwarn


class PlotTrajectory(GiskardBehavior):
    plot_thread: Thread

    @profile
    def __init__(self, name, wait=False, joint_filter=None, normalize_position: bool = False, **kwargs):
        super().__init__(name)
        self.wait = wait
        self.normalize_position = normalize_position
        self.kwargs = kwargs
        self.path_to_data_folder = self.god_map.get_data(identifier.tmp_folder)

    @profile
    def initialise(self):
        self.plot_thread = Thread(target=self.plot)
        self.plot_thread.start()

    def plot(self):
        trajectory = self.god_map.get_data(identifier.trajectory)
        if trajectory:
            sample_period = self.god_map.get_data(identifier.sample_period)
            try:
                trajectory.plot_trajectory(path_to_data_folder=self.path_to_data_folder,
                                           sample_period=sample_period,
                                           normalize_position=self.normalize_position,
                                           **self.kwargs)
            except Exception as e:
                logwarn(e)
                logwarn('failed to save trajectory.pdf')

    @record_time
    @profile
    def update(self):
        if self.wait and self.plot_thread.is_alive():
            return Status.RUNNING
        return Status.SUCCESS
