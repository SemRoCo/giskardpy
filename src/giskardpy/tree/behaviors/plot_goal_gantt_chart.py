from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from py_trees import Status
from sortedcontainers import SortedDict

from giskardpy.goals.collision_avoidance import CollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.monitors.payload_monitors import PayloadMonitor
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard
from giskardpy.utils.utils import create_path, string_shortener


class PlotGanttChart(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot gantt chart'):
        super().__init__(name)

    def plot_gantt_chart(self, goals: Dict[str, Goal], monitors: List[Monitor], file_name: str):
        light_green = (133 / 255, 232 / 255, 133 / 255)
        gray = (128 / 255, 128 / 255, 128 / 255)
        tasks = []
        start_dates = []
        end_dates = []
        bar_height = 0.8
        monitors = [monitor for monitor in monitors if monitor.plot]

        for goal_name, goal in goals.items():
            for i, task in enumerate(goal.tasks):
                if isinstance(goal, CollisionAvoidance):
                    continue
                    # if i > 0:
                    #     break
                    # tasks.append(string_shortener(f'{goal_name}', max_lines=5, max_line_length=50))
                else:
                    tasks.append(task.name)
                if not task.start_monitors:
                    start_dates.append([0])
                else:
                    start_dates.append(
                        [x.state_flip_times[0] if x.state_flip_times else None for x in task.start_monitors])
                if not task.end_monitors:
                    end_dates.append([god_map.time])
                else:
                    end_dates.append(
                        [x.state_flip_times[-1] if x.state_flip_times else None for x in task.end_monitors])

        plt.figure(figsize=(god_map.time * 0.25 + 5, int(len(monitors) + len(tasks)) * 0.3))

        for i, (task, start_date, end_date) in enumerate(reversed(list(zip(tasks, start_dates, end_dates)))):
            if None in start_date:
                start_date = god_map.time
            else:
                start_date = max(start_date)
            if None in end_date:
                end_date = god_map.time
            else:
                end_date = max(end_date)
            plt.barh(task[:50], end_date - start_date, height=bar_height, left=start_date,
                     color=light_green)

        monitors = self.add_running_to_monitors(monitors)
        for monitor in reversed(monitors):
            if (isinstance(monitor, PayloadMonitor) and monitor.run_call_in_thread
                    or not isinstance(monitor, PayloadMonitor) and monitor.start_monitors):
                colors = ['white', gray, 'green']
            else:
                colors = ['white', 'green']
            start_date = 0
            state = False
            for i, end_date in enumerate(monitor.state_flip_times):
                plt.barh(monitor.formatted_name()[:50], end_date - start_date, height=bar_height, left=start_date,
                         color=colors[i % len(colors)])
                start_date = end_date

        plt.gca().yaxis.tick_right()
        plt.subplots_adjust(left=0.01, right=0.75)
        plt.xlabel('Time [s]')
        plt.ylabel('Tasks')
        plt.xlim(0, god_map.time)
        plt.ylim(-1, len(monitors) + len(tasks))
        plt.tight_layout()
        plt.grid()
        create_path(file_name)
        plt.savefig(file_name)
        logging.loginfo(f'Saved gantt chart to {file_name}.')

    def add_running_to_monitors(self, monitors: List[Monitor]) -> List[Monitor]:
        monitor_time_line: Dict[float, List[Tuple[Monitor, Status]]] = SortedDict()
        for monitor in monitors:
            monitor.state_flip_times.append(god_map.time)
            state_counter = 0
            if isinstance(monitor, PayloadMonitor) and monitor.run_call_in_thread:
                states = [Status.FAILURE, Status.RUNNING, Status.SUCCESS]
            else:
                states = [Status.FAILURE, Status.SUCCESS]
            for time in monitor.state_flip_times:
                if time not in monitor_time_line:
                    monitor_time_line[time] = []
                state_counter += 1
                state_counter = state_counter % len(states)
                state = states[state_counter]
                monitor_time_line[time].append((monitor, state))
        total_state: Dict[str, Status] = {monitor.name: Status.FAILURE for monitor in monitors}
        for time in monitor_time_line:
            for (monitor, state) in monitor_time_line[time]:
                total_state[monitor.name] = state
            for monitor in monitors:
                start_monitor_names = [m.name for m in monitor.start_monitors]
                active = np.all(
                    [status == Status.SUCCESS for m, status in total_state.items() if m in start_monitor_names])
                if (not isinstance(monitor, PayloadMonitor)
                        and monitor.start_monitors
                        and total_state[monitor.name] == Status.FAILURE
                        and active):
                    monitor.state_flip_times.append(time)
                    monitor.state_flip_times = list(sorted(monitor.state_flip_times))
                    total_state[monitor.name] = Status.RUNNING

        return monitors

    @record_time
    @profile
    def update(self):
        try:
            goals = god_map.motion_goal_manager.motion_goals
            file_name = god_map.giskard.tmp_folder + f'gantt_charts/goal_{god_map.goal_id}.pdf'
            self.plot_gantt_chart(goals, god_map.monitor_manager.monitors, file_name)
        except Exception as e:
            logging.logwarn(f'Failed to create goal gantt chart: {e}.')
        return Status.SUCCESS
