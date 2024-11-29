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
from giskardpy.motion_graph.monitors.monitors import Monitor, EndMotion, CancelMotion
from giskardpy.god_map import god_map
from giskardpy.motion_graph.tasks.task import TaskState
from giskardpy_ros.tree.behaviors import plot_motion_graph
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import create_path


class PlotGanttChart(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot gantt chart'):
        super().__init__(name)

    def plot_gantt_chart(self, goals: List[Goal], monitors: List[Monitor], file_name: str):
        monitor_plot_filter = np.array([monitor.plot for monitor in god_map.monitor_manager.monitors.values()])
        tasks = [task for g in goals for task in g.tasks]
        task_plot_filter = np.array([not isinstance(g, CollisionAvoidance) for g in goals for _ in g.tasks])

        monitor_history, task_history = self.get_new_history()
        num_monitors = monitor_plot_filter.tolist().count(True)
        num_tasks = task_plot_filter.tolist().count(True)
        num_bars = num_monitors + num_tasks

        figure_height = 0.7 + num_bars * 0.19
        plt.figure(figsize=(god_map.time * 0.25 + 5, figure_height))
        plt.grid(True, axis='x', zorder=-1)

        self.plot_task_history(task_history, tasks, task_plot_filter)
        plt.axhline(y=num_tasks - 0.5, color='black', linestyle='--')
        self.plot_monitor_history(monitor_history, monitors, monitor_plot_filter)

        plt.xlabel('Time [s]')
        plt.xlim(0, monitor_history[-1][0])
        plt.xticks(np.arange(0, god_map.time, 1))

        plt.ylabel('Tasks | Monitors')
        plt.ylim(-1, num_bars)
        plt.gca().yaxis.tick_right()

        plt.subplots_adjust(left=0.05, right=0.75)
        plt.tight_layout()

        create_path(file_name)
        plt.savefig(file_name)
        get_middleware().loginfo(f'Saved gantt chart to {file_name}.')

    def plot_task_history(self,
                          history: List[Tuple[float, List[Optional[TaskState]]]],
                          things, filter: np.ndarray,
                          bar_height: float = 0.8):
        color_map = plot_motion_graph.task_state_to_color
        state: Dict[str, Tuple[float, TaskState]] = {t.name: (0, TaskState.not_started) for t in things}
        for end_time, history_state in history:
            for thing_id, status in enumerate(history_state):
                if not filter[thing_id]:
                    continue
                thing = things[thing_id]
                start_time, last_status = state[thing.name]
                if status != last_status:
                    plt.barh(thing.name[:50], end_time - start_time, height=bar_height, left=start_time,
                             color=color_map[last_status], zorder=2)
                    state[thing.name] = (end_time, status)

    def plot_monitor_history(self,
                             history: List[Tuple[float, Tuple[np.ndarray, np.ndarray]]],
                             things,
                             filter: np.ndarray,
                             bar_height: float = 0.8):
        color_map = plot_motion_graph.monitor_state_to_color
        state = {t.name: (0, 0, TaskState.not_started) for t in things}
        for end_time, (bool_states, history_states) in history:
            for thing_id, status in enumerate(history_states):
                bool_status = bool_states[thing_id]
                if not filter[thing_id]:
                    continue
                thing = things[thing_id]
                start_time, last_bool_status, last_status = state[thing.name]
                outer_color, inner_color = color_map[last_status, last_bool_status]
                if status != last_status or bool_status != last_bool_status:
                    plt.barh(thing.name[:50], end_time - start_time, height=bar_height, left=start_time,
                             color=outer_color, zorder=2)
                    plt.barh(thing.name[:50], end_time - start_time, height=bar_height / 2, left=start_time,
                             color=inner_color, zorder=2)
                    state[thing.name] = (end_time, bool_status, status)

    def get_new_history(self) \
            -> Tuple[List[Tuple[float, Tuple[np.ndarray, np.ndarray]]],
            List[Tuple[float, List[Optional[TaskState]]]]]:
        # because the monitor state doesn't get updated after the final end motion becomes true
        god_map.monitor_manager.evaluate_monitors()

        # add Nones to make sure all bars gets "ended"
        new_end_time = god_map.time + god_map.qp_controller.mpc_dt

        monitor_history = copy(god_map.monitor_manager.state_history)
        monitor_history.append((new_end_time, ([None] * len(monitor_history[0][1][0]), [None] * len(monitor_history[0][1][0]))))

        task_history = copy(god_map.motion_goal_manager.state_history)
        task_history.append((new_end_time, [None] * len(task_history[0][1])))

        return monitor_history, task_history

    @record_time
    @profile
    def update(self):
        if not god_map.monitor_manager.state_history:
            return Status.SUCCESS
        try:
            goals = list(god_map.motion_goal_manager.motion_goals.values())
            monitors = list(god_map.monitor_manager.monitors.values())
            file_name = god_map.tmp_folder + f'gantt_charts/goal_{GiskardBlackboard().move_action_server.goal_id}.pdf'
            self.plot_gantt_chart(goals, monitors, file_name)
        except Exception as e:
            get_middleware().logwarn(f'Failed to create goal gantt chart: {e}.')
            traceback.print_exc()

        return Status.SUCCESS
