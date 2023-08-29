from threading import Thread
from typing import List, Dict

import matplotlib.pyplot as plt
from py_trees import Status

from giskardpy import identifier
from giskardpy.goals.goal import Goal
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy.utils.logging import logwarn


class PlotGanttChart(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot gantt chart'):
        super().__init__(name)

    def plot_gantt_chart(self, goals: Dict[str, Goal]):
        tasks = []
        start_dates = []
        end_dates = []
        for goal_name, goal in goals.items():
            for task in goal.tasks:
                tasks.append(f'{goal_name} - {task.name}')
                if task.to_start is None:
                    start_dates.append(0)
                else:
                    start_dates.append(task.to_start.state_flip_times[0])
                if task.to_end is None:
                    end_dates.append(self.trajectory_time_in_seconds)
                else:
                    end_dates.append(task.to_end.state_flip_times[0])

        plt.figure(figsize=(10, 5))

        for i, (task, start_date, end_date) in enumerate(zip(tasks, start_dates, end_dates)):
            plt.barh(task, end_date - start_date, height=0.8, left=start_date, color=(133/255, 232/255, 133/255))

        for monitor in self.monitors:
            state = False
            for flip_event in monitor.state_flip_times:
                text = f'{monitor.name} {state} -> {not state}'
                state = not state
                plt.axvline(x=flip_event, color='k', linestyle='--')
                plt.text(flip_event, (len(tasks)-1)/2, text, color='k', rotation='vertical', va='center')

        plt.xlabel('Time')
        plt.ylabel('Tasks')
        plt.tight_layout()
        plt.savefig('text.png')

    @record_time
    @profile
    def update(self):
        goals = self.god_map.get_data(identifier.goals)
        self.plot_gantt_chart(goals)
        return Status.SUCCESS
