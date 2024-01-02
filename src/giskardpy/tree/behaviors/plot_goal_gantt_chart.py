from typing import List, Dict

import matplotlib.pyplot as plt
from py_trees import Status

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

        plt.figure(figsize=(god_map.time + 10, int(len(monitors) + len(tasks)) * 0.5))

        for i, (task, start_date, end_date) in enumerate(reversed(list(zip(tasks, start_dates, end_dates)))):
            if None in start_date:
                start_date = god_map.time
            else:
                start_date = max(start_date)
            if None in end_date:
                end_date = god_map.time
            else:
                end_date = max(end_date)
            plt.barh(task[:50], end_date - start_date, height=0.8, left=start_date,
                     color=light_green)

        # monitor_state: Dict[str, bool] = {monitor.name: False for monitor in monitors}
        # todo indicate when all start monitors are active
        for monitor in reversed(monitors):
            if isinstance(monitor, PayloadMonitor) and monitor.run_call_in_thread:
                colors = ['white', light_green, 'green']
            else:
                colors = ['white', 'green']
            monitor.state_flip_times.append(god_map.time)
            start_date = 0
            state = False
            for i, end_date in enumerate(monitor.state_flip_times):

                if state:
                    plt.barh(monitor.formatted_name()[:50], end_date - start_date, height=0.8, left=start_date,
                             color=colors[i % len(colors)])
                else:
                    plt.barh(monitor.formatted_name()[:50], end_date - start_date, height=0.8, left=start_date,
                             color=colors[i % len(colors)])
                start_date = end_date
                state = not state

        plt.gca().yaxis.tick_right()
        plt.subplots_adjust(left=0.01, right=0.75)
        plt.xlabel('Time [s]')
        plt.ylabel('Tasks')
        plt.tight_layout()
        plt.grid()
        create_path(file_name)
        plt.savefig(file_name)
        logging.loginfo(f'Saved gantt chart to {file_name}.')

    @record_time
    @profile
    def update(self):
        goals = god_map.motion_goal_manager.motion_goals
        file_name = god_map.giskard.tmp_folder + f'gantt_charts/goal_{god_map.goal_id}.pdf'
        self.plot_gantt_chart(goals, god_map.monitor_manager.monitors, file_name)
        return Status.SUCCESS
