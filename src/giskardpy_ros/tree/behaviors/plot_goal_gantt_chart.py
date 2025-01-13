import traceback
from copy import copy
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from line_profiler import profile
from py_trees import Status
from giskardpy.goals.collision_avoidance import CollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.middleware import get_middleware
from giskardpy.motion_graph.graph_node import MotionStatechartNode
from giskardpy.motion_graph.monitors.monitors import Monitor, EndMotion, CancelMotion
from giskardpy.god_map import god_map
from giskardpy.motion_graph.tasks.task import LifeCycleState, Task
from giskardpy_ros.tree.behaviors import plot_motion_graph
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import create_path, cm_to_inch


class PlotGanttChart(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot gantt chart'):
        super().__init__(name)

    def plot_gantt_chart(self, tasks: List[MotionStatechartNode], monitors: List[MotionStatechartNode],
                         goals: List[MotionStatechartNode], file_name: str):
        monitor_plot_filter = np.array([monitor.plot for monitor in god_map.motion_graph_manager.monitor_state.nodes])
        goal_plot_filter = np.array([goal.plot for goal in god_map.motion_graph_manager.goal_state.nodes])
        task_plot_filter = np.array([not isinstance(g, CollisionAvoidance) for g in tasks])

        monitor_history, task_history, goal_history = self.get_new_history()
        num_monitors = monitor_plot_filter.tolist().count(True)
        num_tasks = task_plot_filter.tolist().count(True)
        num_goals = goal_plot_filter.tolist().count(True)
        num_bars = num_monitors + num_tasks + num_goals

        # Create mappings from task/monitor names to numerical y positions
        task_plot_indices = [i for i, flag in enumerate(task_plot_filter) if flag]
        task_names = [tasks[i].name[:50] for i in task_plot_indices]
        task_positions = {name: index for index, name in enumerate(task_names)}

        monitor_plot_indices = [i for i, flag in enumerate(monitor_plot_filter) if flag]
        monitor_names = [monitors[i].name[:50] for i in monitor_plot_indices]
        monitor_positions = {name: num_tasks + index for index, name in enumerate(monitor_names)}

        goal_plot_indices = [i for i, flag in enumerate(goal_plot_filter) if flag]
        goal_names = [goals[i].name[:50] for i in goal_plot_indices]
        goal_positions = {name: num_tasks + num_monitors + index for index, name in enumerate(goal_names)}

        # Combine positions and names
        thing_positions = {**task_positions, **monitor_positions, **goal_positions}
        thing_names = task_names + monitor_names + goal_names

        cm_per_second = cm_to_inch(2.5)
        figure_height = 0.7 + num_bars * 0.19
        plt.figure(figsize=(god_map.time * cm_per_second + 2.5, figure_height))
        plt.grid(True, axis='x', zorder=-1)

        self.plot_history(task_history, tasks, task_plot_filter, task_positions)
        plt.axhline(y=num_tasks - 0.5, color='black', linestyle='--')
        self.plot_history(monitor_history, monitors, monitor_plot_filter, monitor_positions)
        plt.axhline(y=num_tasks+num_monitors - 0.5, color='black', linestyle='--')
        self.plot_history(goal_history, goals, goal_plot_filter, goal_positions)

        plt.xlabel('Time [s]')
        plt.xlim(0, monitor_history[-1][0])
        plt.xticks(np.arange(0, god_map.time, 1))

        plt.ylabel('Tasks | Monitors')
        plt.ylim(-0.8, num_bars - 1 + 0.8)
        plt.yticks([thing_positions[name] for name in thing_names], thing_names)
        plt.gca().yaxis.tick_right()

        plt.subplots_adjust(left=0.05, right=0.75)
        plt.tight_layout()

        create_path(file_name)
        plt.savefig(file_name)
        get_middleware().loginfo(f'Saved gantt chart to {file_name}.')

    def plot_history(self,
                     history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]],
                     things,
                     filter: np.ndarray,
                     thing_positions: Dict[str, float],
                     bar_height: float = 0.8):
        state = {t.name: (0, 0, LifeCycleState.not_started) for t in things}
        for end_time, (bool_states, history_states) in history:
            for thing_id, status in enumerate(history_states):
                bool_status = bool_states[thing_id]
                if not filter[thing_id]:
                    continue
                thing = things[thing_id]
                start_time, last_bool_status, last_status = state[thing.name]
                outer_color = plot_motion_graph.LiftCycleStateToColor[last_status]
                inner_color = plot_motion_graph.ObservationStateToColor[last_bool_status]
                if status != last_status or bool_status != last_bool_status:
                    y_pos = thing_positions[thing.name[:50]]
                    # Top half bar (old fat bar)
                    plt.barh(y_pos + bar_height / 4, end_time - start_time, height=bar_height / 2, left=start_time,
                             color=outer_color, zorder=2)
                    # Bottom half bar (old thin bar)
                    plt.barh(y_pos - bar_height / 4, end_time - start_time, height=bar_height / 2, left=start_time,
                             color=inner_color, zorder=2)
                    state[thing.name] = (end_time, bool_status, status)

    def get_new_history(self) \
            -> Tuple[List[Tuple[float, Tuple[np.ndarray, np.ndarray]]],
                     List[Tuple[float, Tuple[np.ndarray, np.ndarray]]],
                     List[Tuple[float, Tuple[np.ndarray, np.ndarray]]]]:
        # because the monitor state doesn't get updated after the final end motion becomes true
        try:
            god_map.motion_graph_manager.evaluate_node_states()
        except Exception as e:
            # if the motion was cancelled, this call will cause an exception
            pass

        # add Nones to make sure all bars gets "ended"
        new_end_time = god_map.time + god_map.qp_controller.mpc_dt

        task_history = copy(god_map.motion_graph_manager.task_state_history)
        task_history.append((new_end_time, ([None] * len(task_history[0][1][0]),
                                            [None] * len(task_history[0][1][0]))))
        monitor_history = copy(god_map.motion_graph_manager.monitor_state_history)
        monitor_history.append((new_end_time, ([None] * len(monitor_history[0][1][0]),
                                               [None] * len(monitor_history[0][1][0]))))
        goal_history = copy(god_map.motion_graph_manager.goal_state_history)
        goal_history.append((new_end_time, ([None] * len(goal_history[0][1][0]),
                                            [None] * len(goal_history[0][1][0]))))

        return monitor_history, task_history, goal_history

    @record_time
    @profile
    def update(self):
        if not god_map.motion_graph_manager.monitor_state_history:
            return Status.SUCCESS
        try:
            tasks = god_map.motion_graph_manager.task_state.nodes
            monitors = god_map.motion_graph_manager.monitor_state.nodes
            goals = god_map.motion_graph_manager.goal_state.nodes
            file_name = god_map.tmp_folder + f'gantt_charts/goal_{GiskardBlackboard().move_action_server.goal_id}.pdf'
            self.plot_gantt_chart(tasks, monitors, goals, file_name)
        except Exception as e:
            get_middleware().logwarn(f'Failed to create goal gantt chart: {e}.')
            traceback.print_exc()

        return Status.SUCCESS
