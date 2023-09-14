from abc import ABC, abstractmethod
from typing import Optional

from giskardpy import identifier
from giskardpy.god_map import GodMap
from giskardpy.tree.behaviors.compile_debug_expressions import CompileDebugExpressions
from giskardpy.tree.behaviors.tf_publisher import TfPublishingModes
from giskardpy.tree.garden import OpenLoop, ClosedLoop, StandAlone, TreeManager
from giskardpy.tree.control_modes import ControlModes


class BehaviorTreeConfig(ABC):
    god_map = GodMap()

    def __init__(self, mode: ControlModes):
        self._control_mode = mode

    @abstractmethod
    def setup(self):
        """
        Implement this method to configure the behavior tree using it's self. methods.
        """

    @property
    def tree_manager(self) -> TreeManager:
        return self.god_map.get_data(identifier.tree_manager)

    def _create_behavior_tree(self):
        TreeManager(self._control_mode)

    def set_defaults(self):
        pass

    def set_tree_tick_rate(self, rate: float = 0.05):
        """
        How often the tree ticks per second.
        :param rate: in /s
        """
        self.tree_tick_rate = rate

    def add_sleeper(self, time: float):
        self.tree_manager.add_sleeper(time)

    def add_visualization_marker_publisher(self,
                                           add_to_sync: Optional[bool] = None,
                                           add_to_planning: Optional[bool] = None,
                                           add_to_control_loop: Optional[bool] = None,
                                           use_decomposed_meshes: bool = True):
        """

        :param add_to_sync: Markers are published while waiting for a goal.
        :param add_to_planning: Markers are published during planning, only relevant in open loop mode.
        :param add_to_control_loop: Markers are published during the closed loop control sequence, this is slow.
        :param use_decomposed_meshes: True: publish decomposed meshes used for collision avoidance, these likely only
                                            available on the machine where Giskard is running.
                                      False: use meshes defined in urdf.
        """
        if add_to_sync:
            self.tree_manager.tree.wait_for_goal.publish_state.add_visualization_marker_behavior(use_decomposed_meshes)
        if add_to_control_loop:
            self.tree_manager.tree.process_goal.control_loop_branch.publish_state.add_visualization_marker_behavior(use_decomposed_meshes)
        # FIXME add to planning

    def add_qp_data_publisher(self, publish_lb: bool = False, publish_ub: bool = False,
                              publish_lbA: bool = False, publish_ubA: bool = False,
                              publish_bE: bool = False, publish_Ax: bool = False,
                              publish_Ex: bool = False, publish_xdot: bool = False,
                              publish_weights: bool = False, publish_g: bool = False,
                              publish_debug: bool = False, add_to_base: bool = False):
        """
        QP data is streamed and can be visualized in e.g. plotjuggler. Useful for debugging.
        """
        self.add_evaluate_debug_expressions()
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
        """
        Plots the generated trajectories.
        :param normalize_position: Positions are centered around zero.
        :param wait: True: Behavior tree waits for this plotter to finish.
                     False: Plot is generated in a separate thread to not slow down Giskard.
        """
        self.tree_manager.tree.post_processing.add_plot_trajectory(normalize_position, wait)

    def add_debug_trajectory_plotter(self, normalize_position: bool = False, wait: bool = False):
        """
        Plots debug expressions defined in goals.
        """
        self.add_evaluate_debug_expressions()
        self.tree_manager.tree.post_processing.add_plot_debug_trajectory(normalize_position=normalize_position, wait=wait)

    def add_debug_marker_publisher(self):
        """
        Publishes debug expressions defined in goals.
        """
        self.add_evaluate_debug_expressions()
        self.tree_manager.add_debug_marker_publisher()

    def add_tf_publisher(self, include_prefix: bool = True, tf_topic: str = 'tf',
                         mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects):
        """
        Publishes tf for Giskard's internal state.
        """
        self.tree_manager.tree.wait_for_goal.publish_state.add_tf_publisher(include_prefix=include_prefix,
                                                                            tf_topic=tf_topic,
                                                                            mode=mode)

    def add_evaluate_debug_expressions(self):
        self.tree_manager.tree.prepare_control_loop.add_child(CompileDebugExpressions())
        self.tree_manager.tree.process_goal.control_loop_branch.add_evaluate_debug_expressions()


class StandAloneBTConfig(BehaviorTreeConfig):
    def __init__(self, planning_sleep: Optional[float] = None):
        super().__init__(ControlModes.standalone)
        self.planning_sleep = planning_sleep

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=False, add_to_control_loop=True)
        self.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)
        self.add_trajectory_plotter()
        self.add_debug_trajectory_plotter()
        # self.add_debug_marker_publisher()
        if self.planning_sleep is not None:
            self.add_sleeper(self.planning_sleep)


class OpenLoopBTConfig(BehaviorTreeConfig):
    def __init__(self, planning_sleep: Optional[float] = None):
        super().__init__(ControlModes.open_loop)
        self.planning_sleep = planning_sleep

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=True, add_to_control_loop=False)
        if self.planning_sleep is not None:
            self.add_sleeper(self.planning_sleep)


class ClosedLoopBTConfig(BehaviorTreeConfig):
    def __init__(self):
        super().__init__(ControlModes.close_loop)

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=False, add_to_control_loop=False)
        #self.add_qp_data_publisher(publish_xdot=True, publish_lb=True, publish_ub=True)
