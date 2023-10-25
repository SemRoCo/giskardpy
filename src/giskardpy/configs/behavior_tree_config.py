from abc import ABC, abstractmethod
from typing import Optional

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.tf_publisher import TfPublishingModes
from giskardpy.tree.branches.giskard_bt import GiskardBT
from giskardpy.tree.control_modes import ControlModes


class BehaviorTreeConfig(ABC):

    def __init__(self, mode: ControlModes):
        self._control_mode = mode

    @abstractmethod
    def setup(self):
        """
        Implement this method to configure the behavior tree using it's self. methods.
        """

    @property
    def tree(self) -> GiskardBT:
        return god_map.tree

    def _create_behavior_tree(self):
        god_map.tree = GiskardBT(control_mode=self._control_mode)

    def set_defaults(self):
        pass

    def set_tree_tick_rate(self, rate: float = 0.05):
        """
        How often the tree ticks per second.
        :param rate: in /s
        """
        self.tree_tick_rate = rate

    def add_sleeper(self, time: float):
        self.add_sleeper(time)

    def add_visualization_marker_publisher(self,
                                           add_to_sync: Optional[bool] = None,
                                           add_to_control_loop: Optional[bool] = None,
                                           use_decomposed_meshes: bool = True):
        """

        :param add_to_sync: Markers are published while waiting for a goal.
        :param add_to_control_loop: Markers are published during the closed loop control sequence, this is slow.
        :param use_decomposed_meshes: True: publish decomposed meshes used for collision avoidance, these likely only
                                            available on the machine where Giskard is running.
                                      False: use meshes defined in urdf.
        """
        if add_to_sync:
            self.tree.wait_for_goal.publish_state.add_visualization_marker_behavior(use_decomposed_meshes)
        if add_to_control_loop:
            self.tree.control_loop_branch.publish_state.add_visualization_marker_behavior(
                use_decomposed_meshes)

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
        if god_map.is_planning():
            self.tree.execute_traj.base_closed_loop.publish_state.add_qp_data_publisher(
                publish_lb=publish_lb,
                publish_ub=publish_ub,
                publish_lbA=publish_lbA,
                publish_ubA=publish_ubA,
                publish_bE=publish_bE,
                publish_Ax=publish_Ax,
                publish_Ex=publish_Ex,
                publish_xdot=publish_xdot,
                publish_weights=publish_weights,
                publish_g=publish_g,
                publish_debug=publish_debug)
        else:
            self.tree.control_loop_branch.publish_state.add_qp_data_publisher(
                publish_lb=publish_lb,
                publish_ub=publish_ub,
                publish_lbA=publish_lbA,
                publish_ubA=publish_ubA,
                publish_bE=publish_bE,
                publish_Ax=publish_Ax,
                publish_Ex=publish_Ex,
                publish_xdot=publish_xdot,
                publish_weights=publish_weights,
                publish_g=publish_g,
                publish_debug=publish_debug)

    def add_trajectory_plotter(self, normalize_position: bool = False, wait: bool = False):
        """
        Plots the generated trajectories.
        :param normalize_position: Positions are centered around zero.
        :param wait: True: Behavior tree waits for this plotter to finish.
                     False: Plot is generated in a separate thread to not slow down Giskard.
        """
        self.tree.cleanup_control_loop.add_plot_trajectory(normalize_position, wait)

    def add_debug_trajectory_plotter(self, normalize_position: bool = False, wait: bool = False):
        """
        Plots debug expressions defined in goals.
        """
        self.add_evaluate_debug_expressions()
        self.tree.cleanup_control_loop.add_plot_debug_trajectory(normalize_position=normalize_position,
                                                                              wait=wait)

    def add_gantt_chart_plotter(self):
        self.add_evaluate_debug_expressions()
        self.tree.cleanup_control_loop.add_plot_gantt_chart()

    def add_goal_graph_plotter(self):
        self.add_evaluate_debug_expressions()
        self.tree.prepare_control_loop.add_plot_goal_graph()

    def add_debug_marker_publisher(self):
        """
        Publishes debug expressions defined in goals.
        """
        self.add_evaluate_debug_expressions()
        self.tree.control_loop_branch.publish_state.add_debug_marker_publisher()

    def add_tf_publisher(self, include_prefix: bool = True, tf_topic: str = 'tf',
                         mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects):
        """
        Publishes tf for Giskard's internal state.
        """
        self.tree.wait_for_goal.publish_state.add_tf_publisher(include_prefix=include_prefix,
                                                                            tf_topic=tf_topic,
                                                                            mode=mode)

    def add_evaluate_debug_expressions(self):
        self.tree.prepare_control_loop.add_compile_debug_expressions()
        if god_map.is_closed_loop():
            self.tree.control_loop_branch.add_evaluate_debug_expressions(log_traj=False)
        else:
            self.tree.control_loop_branch.add_evaluate_debug_expressions(log_traj=True)
        if god_map.is_planning():
            god_map.tree.execute_traj.prepare_base_control.add_compile_debug_expressions()
            god_map.tree.execute_traj.base_closed_loop.add_evaluate_debug_expressions(log_traj=False)


class StandAloneBTConfig(BehaviorTreeConfig):
    def __init__(self, planning_sleep: Optional[float] = None, debug_mode: bool = False):
        super().__init__(ControlModes.standalone)
        self.planning_sleep = planning_sleep
        if god_map.is_in_github_workflow():
            debug_mode = False
        self.debug_mode = debug_mode

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_control_loop=True)
        self.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)
        if self.debug_mode:
            self.add_trajectory_plotter(wait=True)
            self.add_debug_trajectory_plotter(wait=True)
            self.add_gantt_chart_plotter()
            self.add_goal_graph_plotter()
            self.add_debug_marker_publisher()
        # self.add_debug_marker_publisher()
        if self.planning_sleep is not None:
            self.add_sleeper(self.planning_sleep)


class OpenLoopBTConfig(BehaviorTreeConfig):
    def __init__(self, planning_sleep: Optional[float] = None, debug_mode: bool = False):
        super().__init__(ControlModes.open_loop)
        self.planning_sleep = planning_sleep
        if god_map.is_in_github_workflow():
            debug_mode = False
        self.debug_mode = debug_mode

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_control_loop=True)
        if self.debug_mode:
            self.add_trajectory_plotter(wait=True)
            self.add_debug_trajectory_plotter(wait=True)
            # self.add_gantt_chart_plotter()
            # self.add_goal_graph_plotter()
            self.add_debug_marker_publisher()
            self.add_qp_data_publisher(
                publish_debug=True,
                publish_xdot=True,
                # publish_lbA=True,
                # publish_ubA=True
            )
        if self.planning_sleep is not None:
            self.add_sleeper(self.planning_sleep)


class ClosedLoopBTConfig(BehaviorTreeConfig):
    def __init__(self, debug_mode: bool = False):
        super().__init__(ControlModes.close_loop)
        if god_map.is_in_github_workflow():
            debug_mode = False
        self.debug_mode = debug_mode

    def setup(self):
        self.add_visualization_marker_publisher(add_to_sync=True, add_to_control_loop=False)
        # self.add_qp_data_publisher(publish_xdot=True, publish_lb=True, publish_ub=True)
        if self.debug_mode:
            # self.add_trajectory_plotter(wait=True)
            # self.add_debug_trajectory_plotter(wait=True)
            # self.add_gantt_chart_plotter()
            # self.add_goal_graph_plotter()
            self.add_debug_marker_publisher()
            self.add_qp_data_publisher(
                publish_debug=True,
                publish_xdot=True,
                # publish_lbA=True,
                # publish_ubA=True
            )
