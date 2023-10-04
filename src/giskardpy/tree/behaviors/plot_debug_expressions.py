import traceback
from threading import Lock
import numpy as np

from giskardpy.data_types import JointStates
from giskardpy.god_map import god_map
from giskardpy.model.trajectory import Trajectory
from giskardpy.tree.behaviors.plot_trajectory import PlotTrajectory
from giskardpy.utils.logging import logwarn

plot_lock = Lock()


class PlotDebugExpressions(PlotTrajectory):
    @profile
    def __init__(self, name, wait=True, normalize_position: bool = False, **kwargs):
        super().__init__(name=name,
                         normalize_position=normalize_position,
                         wait=wait,
                         **kwargs)
        # self.path_to_data_folder += 'debug_expressions/'
        # create_path(self.path_to_data_folder)

    def split_traj(self, traj) -> Trajectory:
        new_traj = Trajectory()
        for time, js in traj.items():
            new_js = JointStates()
            for name, js_ in js.items():
                if isinstance(js_.position, np.ndarray):
                    if len(js_.position.shape) == 1:
                        for x in range(js_.position.shape[0]):
                            tmp_name = f'{name}|{x}'
                            new_js[tmp_name].position = js_.position[x]
                            new_js[tmp_name].velocity = js_.velocity[x]
                    else:
                        for x in range(js_.position.shape[0]):
                            for y in range(js_.position.shape[1]):
                                tmp_name = f'{name}|{x}_{y}'
                                new_js[tmp_name].position = js_.position[x, y]
                                new_js[tmp_name].velocity = js_.velocity[x, y]
                else:
                    new_js[name] = js_
                new_traj.set(time, new_js)

        return new_traj

    def plot(self):
        trajectory = god_map.debug_expression_manager.debug_trajectory
        if trajectory and len(trajectory.items()) > 0:
            sample_period = god_map.qp_controller_config.sample_period
            traj = self.split_traj(trajectory)
            try:
                traj.plot_trajectory(path_to_data_folder=self.path_to_data_folder,
                                     sample_period=sample_period,
                                     file_name=f'debug.pdf',
                                     filter_0_vel=False,
                                     **self.kwargs)
            except Exception:
                traceback.print_exc()
                logwarn('failed to save debug.pdf')
