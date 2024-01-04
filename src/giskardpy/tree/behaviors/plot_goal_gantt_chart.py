import traceback
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from py_trees import Status
from giskardpy.goals.collision_avoidance import CollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.monitors.monitors import Monitor
from giskardpy.monitors.payload_monitors import PayloadMonitor, EndMotion, CancelMotion
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import create_path


class PlotGanttChart(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot gantt chart'):
        super().__init__(name)

    def plot_gantt_chart(self, goals: List[Goal], monitors: List[Monitor], file_name: str):
        monitor_plot_filter = np.array([monitor.plot for monitor in god_map.monitor_manager.monitors])
        tasks = [task for g in goals for task in g.tasks]
        task_plot_filter = np.array([not isinstance(g, CollisionAvoidance) for g in goals for _ in g.tasks])

        monitor_history, task_history = self.get_new_history()
        num_monitors = monitor_plot_filter.tolist().count(True)
        num_tasks = task_plot_filter.tolist().count(True)
        num_bars = num_monitors + num_tasks

        figure_height = num_bars * 0.3
        plt.figure(figsize=(god_map.time * 0.25 + 5, figure_height))

        self.plot_history(task_history, tasks, task_plot_filter)
        plt.axhline(y=num_tasks-0.5, color='black', linestyle='--')
        self.plot_history(monitor_history, monitors, monitor_plot_filter)

        plt.xlabel('Time [s]')
        plt.xlim(0, monitor_history[-1][0])
        plt.xticks(np.arange(0, god_map.time, 1))
        # Create an instance of MaxNLocator with the desired maximum number of ticks
        # locator = MaxNLocator(nbins=20)  # Adjust 'nbins' as needed for more ticks

        # Set the locator for the x-axis
        # plt.gca().xaxis.set_major_locator(locator)

        plt.ylabel('Tasks | Monitors')
        plt.ylim(-1, num_bars)
        plt.gca().yaxis.tick_right()
        plt.grid()

        # legend_colors = ['green', 'gray', 'white', 'blue']
        # legend_labels = ['Monitor==True\nTask started but not ended',
        #                  'Monitor==False but active\nTask on hold',
        #                  'Monitor==False and inactive\nTask not started or ended',
        #                  'Monitor that ended/cancelled the motion']
        # legend_handles = [mpatches.Patch(color=color, label=label, edgecolor='black', linewidth=1) for color, label in
        #                   zip(legend_colors, legend_labels)]

        # def abs_to_relative(value):
        #     return value/figure_height

        # Add custom legend to the plot
        # plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0, abs_to_relative(-0.9)))

        # plt.subplots_adjust(left=0.05, right=0.75, bottom=0.4)
        plt.subplots_adjust(left=0.05, right=0.75)
        plt.tight_layout()

        create_path(file_name)
        plt.savefig(file_name)
        logging.loginfo(f'Saved gantt chart to {file_name}.')

    def plot_history(self, history: List[Tuple[float, List[Optional[Status]]]], things, filter: np.ndarray,
                     bar_height: float = 0.8):
        color_map = {Status.FAILURE: 'white', Status.RUNNING: 'gray', Status.SUCCESS: 'green'}
        state: Dict[str, Tuple[float, Status]] = {t.name: (0, Status.FAILURE) for t in things}
        for end_time, history_state in history:
            for thing_id, status in enumerate(history_state):
                if not filter[thing_id]:
                    continue
                thing = things[thing_id]
                start_time, last_status = state[thing.name]
                if status != last_status:
                    plt.barh(thing.name[:50], end_time - start_time, height=bar_height, left=start_time,
                             color=color_map[last_status])
                    if isinstance(thing, (EndMotion, CancelMotion)) and status == Status.SUCCESS:
                        plt.barh(thing.name[:50], end_time, height=bar_height/2, color='blue')
                    state[thing.name] = (end_time, status)

    def get_new_history(self) \
            -> Tuple[List[Tuple[float, List[Optional[Status]]]], List[Tuple[float, List[Optional[Status]]]]]:
        # because the monitor state doesn't get updated after the final end motion becomes true
        god_map.monitor_manager.evaluate_monitors()
        monitor_history: List[Tuple[float, List[Optional[Status]]]] = []
        task_history: List[Tuple[float, List[Optional[Status]]]] = []
        for time_id, (time, state) in enumerate(god_map.monitor_manager.state_history):
            next_monitor_state = [Status.SUCCESS if x else Status.FAILURE for x in state]
            next_task_state = []
            if time_id >= 1:
                for monitor_id, monitor in enumerate(god_map.monitor_manager.monitors):
                    monitor_state = state[monitor_id]
                    if not monitor.start_monitors:
                        if isinstance(monitor, PayloadMonitor) and not monitor_state:
                            next_monitor_state[monitor_id] = Status.RUNNING
                        continue
                    prev_state = god_map.monitor_manager.state_history[time_id-1][1]
                    active = np.all(prev_state[monitor.state_filter])
                    if not monitor_state and active:
                        next_monitor_state[monitor_id] = Status.RUNNING

            for goal_id, goal in enumerate(god_map.motion_goal_manager.motion_goals.values()):
                for task_id, task in enumerate(goal.tasks):
                    if task.start_monitors:
                        started = np.all(state[god_map.monitor_manager.to_state_filter(task.start_monitors)])
                    else:
                        started = True
                    if task.hold_monitors:
                        held = np.all(state[god_map.monitor_manager.to_state_filter(task.hold_monitors)])
                    else:
                        held = False
                    if task.end_monitors:
                        ended = np.all(state[god_map.monitor_manager.to_state_filter(task.end_monitors)])
                    else:
                        ended = False
                    task_state = Status.FAILURE
                    if not ended:
                        if started:
                            if held:
                                task_state = Status.RUNNING
                            else:
                                task_state = Status.SUCCESS
                    next_task_state.append(task_state)

            monitor_history.append((time, next_monitor_state))
            task_history.append((time, next_task_state))
        # add Nones to make sure all bars gets "ended"
        new_end_time = god_map.time + god_map.qp_controller_config.sample_period
        monitor_history.append((new_end_time, [None] * len(monitor_history[0][1])))
        task_history.append((new_end_time, [None] * len(task_history[0][1])))

        return monitor_history, task_history

    @record_time
    @profile
    def update(self):
        if not god_map.monitor_manager.state_history:
            return Status.SUCCESS
        try:
            goals = list(god_map.motion_goal_manager.motion_goals.values())
            monitors = god_map.monitor_manager.monitors
            file_name = god_map.giskard.tmp_folder + f'gantt_charts/goal_{god_map.goal_id}.pdf'
            self.plot_gantt_chart(goals, monitors, file_name)
        except Exception as e:
            logging.logwarn(f'Failed to create goal gantt chart: {e}.')
            traceback.print_exc()

        return Status.SUCCESS
