from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from giskardpy import identifier
from giskardpy.god_map import GodMap
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.tree.behaviors.tf_publisher import TfPublishingModes
from giskardpy.tree.garden import OpenLoop, ClosedLoop, StandAlone, ControlModes


class BehaviorTreeConfig(GodMapWorshipper, ABC):
    def __init__(self, mode: ControlModes):
        self._control_mode = mode

    @abstractmethod
    def setup(self):
        ...

    def _create_behavior_tree(self):
        if self._control_mode == ControlModes.open_loop:
            behavior_tree = OpenLoop()
        elif self._control_mode == ControlModes.close_loop:
            behavior_tree = ClosedLoop()
        elif self._control_mode == ControlModes.standalone:
            behavior_tree = StandAlone()
        else:
            raise KeyError(f'Robot interface mode \'{self._control_mode}\' is not supported.')
        self.god_map.set_data(identifier.tree_manager, behavior_tree)

    def set_defaults(self):
        pass

    def set_tree_tick_rate(self, rate: float = 0.05):
        self.tree_tick_rate = rate

    def add_visualization_marker_publisher(self,
                                           add_to_sync: Optional[bool] = None,
                                           add_to_planning: Optional[bool] = None,
                                           add_to_control_loop: Optional[bool] = None,
                                           use_decomposed_meshes: bool = True):
        self.tree_manager.configure_visualization_marker(add_to_sync=add_to_sync, add_to_planning=add_to_planning,
                                                          add_to_control_loop=add_to_control_loop,
                                                          use_decomposed_meshes=use_decomposed_meshes)

    def add_qp_data_publisher(self, publish_lb: bool = False, publish_ub: bool = False,
                              publish_lbA: bool = False, publish_ubA: bool = False,
                              publish_bE: bool = False, publish_Ax: bool = False,
                              publish_Ex: bool = False, publish_xdot: bool = False,
                              publish_weights: bool = False, publish_g: bool = False,
                              publish_debug: bool = False, add_to_base: bool = False):
        self.tree_manager.add_qp_data_publisher(publish_lb=publish_lb,
                                                 publish_ub=publish_ub,
                                                 publish_lbA=publish_lbA,
                                                 publish_ubA=publish_ubA,
                                                 publish_bE=publish_bE,
                                                 publish_Ax=publish_Ax,
                                                 publish_Ex=publish_Ex,
                                                 publish_xdot=publish_xdot,
                                                 publish_weights=publish_weights,
                                                 publish_g=publish_g,
                                                 publish_debug=publish_debug,
                                                 add_to_base=add_to_base)

    def add_trajectory_plotter(self, normalize_position: bool = False, wait: bool = False):
        self.tree_manager.add_plot_trajectory(normalize_position, wait)

    def add_debug_trajectory_plotter(self, normalize_position: bool = False, wait: bool = False):
        self.tree_manager.add_plot_debug_trajectory(normalize_position=normalize_position, wait=wait)

    def add_debug_marker_publisher(self):
        self.tree_manager.add_debug_marker_publisher()

    def add_tf_publisher(self, include_prefix: bool = True, tf_topic: str = 'tf',
                         mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects):
        self.tree_manager.add_tf_publisher(include_prefix=include_prefix, tf_topic=tf_topic, mode=mode)


class StandAloneConfig(BehaviorTreeConfig):
    def __init__(self):
        super().__init__(ControlModes.standalone)

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=False, add_to_control_loop=True)
        self.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)


class OpenLoopConfig(BehaviorTreeConfig):
    def __init__(self):
        super().__init__(ControlModes.open_loop)

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=True, add_to_control_loop=False)


class ClosedLoopConfig(BehaviorTreeConfig):
    def __init__(self):
        super().__init__(ControlModes.close_loop)

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=False, add_to_control_loop=False)
