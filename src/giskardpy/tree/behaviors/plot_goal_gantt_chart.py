from typing import List, Dict

import matplotlib.pyplot as plt
from py_trees import Status

from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard
from giskardpy.utils.utils import create_path, string_shortener


class PlotGanttChart(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot gantt chart'):
        super().__init__(name)

    def plot_gantt_chart(self, goals: Dict[str, Goal], file_name: str):
        tasks = []
        start_dates = []
        end_dates = []
        for goal_name, goal in goals.items():
            for task in goal.tasks:
                tasks.append(string_shortener(f'{goal_name} - {task.name}',
                                              max_lines=5, max_line_length=50))
                if not task.to_start:
                    start_dates.append([0])
                else:
                    start_dates.append([x.state_flip_times[0] for x in task.to_start])
                if not task.to_end:
                    end_dates.append([god_map.time])
                else:
                    end_dates.append([x.state_flip_times[-1] if x.state_flip_times else None for x in task.to_end])

        plt.figure(figsize=(10, 5))

        for i, (task, start_date, end_date) in enumerate(zip(tasks, start_dates, end_dates)):
            if None in end_date:
                end_date = god_map.time
            else:
                end_date = max(end_date)
            plt.barh(task, end_date - max(start_date), height=0.8, left=start_date,
                     color=(133 / 255, 232 / 255, 133 / 255))

        for monitor in god_map.monitor_manager.monitors:
            state = False
            for flip_event in monitor.state_flip_times:
                monitor_name = string_shortener(monitor.name,
                                                max_lines=2, max_line_length=50)
                text = f'{monitor_name}\n{state} -> {not state}'
                state = not state
                plt.axvline(x=flip_event, color='k', linestyle='--')
                plt.text(flip_event, (len(tasks) - 1) / 2, text, color='k', rotation='vertical', va='center')

        plt.xlabel('Time [s]')
        plt.ylabel('Tasks')
        plt.tight_layout()
        create_path(file_name)
        plt.savefig(file_name)
        logging.loginfo(f'Saved gantt chart to {file_name}.')

    @record_time
    @profile
    def update(self):
        goals = god_map.motion_goal_manager.motion_goals
        file_name = god_map.giskard.tmp_folder + f'gantt_charts/goal_{god_map.goal_id}.png'
        self.plot_gantt_chart(goals, file_name)
        return Status.SUCCESS
