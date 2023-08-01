from abc import ABC
from enum import Enum
from typing import Optional

from giskardpy import identifier
from giskardpy.god_map import GodMap
from giskardpy.tree.garden import OpenLoop, ClosedLoop, StandAlone


class ControlModes(Enum):
    open_loop = 1
    close_loop = 2
    standalone = 3


class TfPublishingModes(Enum):
    nothing = 0
    all = 1
    attached_objects = 2

    world_objects = 4
    attached_and_world_objects = 6


class BehaviorTreeConfig(ABC):
    tree_tick_rate: float = 0.05
    god_map = GodMap()

    def __init__(self, mode: ControlModes):
        self.control_mode = mode

    def _create_behavior_tree(self):
        if self.control_mode == ControlModes.open_loop:
            self.behavior_tree = OpenLoop()
        elif self.control_mode == ControlModes.close_loop:
            self.behavior_tree = ClosedLoop()
        elif self.control_mode == ControlModes.standalone:
            self.behavior_tree = StandAlone()
        else:
            raise KeyError(f'Robot interface mode \'{self.control_mode}\' is not supported.')
        self.god_map.set_data(identifier.tree_manager, self.behavior_tree)

    def set_defaults(self):
        pass

    def set_tree_tick_rate(self, rate: float = 0.05):
        self.tree_tick_rate = rate

    def add_visualization_marker_publisher(self,
                                           add_to_sync: Optional[bool] = None,
                                           add_to_planning: Optional[bool] = None,
                                           add_to_control_loop: Optional[bool] = None,
                                           use_decomposed_meshes: bool = True):
        self.behavior_tree.configure_visualization_marker(add_to_sync=add_to_sync, add_to_planning=add_to_planning,
                                                          add_to_control_loop=add_to_control_loop,
                                                          use_decomposed_meshes=use_decomposed_meshes)

    def add_qp_data_publisher(self, publish_lb: bool = False, publish_ub: bool = False,
                              publish_lbA: bool = False, publish_ubA: bool = False,
                              publish_bE: bool = False, publish_Ax: bool = False,
                              publish_Ex: bool = False, publish_xdot: bool = False,
                              publish_weights: bool = False, publish_g: bool = False,
                              publish_debug: bool = False, add_to_base: bool = False):
        self.behavior_tree.add_qp_data_publisher(publish_lb=publish_lb,
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
        self.behavior_tree.add_plot_trajectory(normalize_position, wait)

    def add_debug_trajectory_plotter(self, normalize_position: bool = False, wait: bool = False):
        self.behavior_tree.add_plot_debug_trajectory(normalize_position=normalize_position, wait=wait)

    def add_debug_marker_publisher(self):
        self.behavior_tree.add_debug_marker_publisher()

    def add_tf_publisher(self, include_prefix: bool = True, tf_topic: str = 'tf',
                         mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects):
        self.behavior_tree.add_tf_publisher(include_prefix=include_prefix, tf_topic=tf_topic, mode=mode)


class StandAloneConfig(BehaviorTreeConfig):
    def __init__(self):
        super().__init__(ControlModes.standalone)

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=False, add_to_control_loop=True)
        self.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)
