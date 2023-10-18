from collections import defaultdict
from typing import List

import pydot
from py_trees import Status

from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.tasks.task import Task
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard
from giskardpy.utils.utils import create_path, string_shortener


def generate_graph(tasks: List[Task], monitors: List[Monitor], output_file: str = 'graph.png'):
    def shorten(string: str) -> str:
        return string_shortener(original_str=string,
                                max_lines=4,
                                max_line_length=25)

    # Initialize graph
    graph = pydot.Dot(graph_type='digraph')

    # Flags to track special cases
    add_t0 = False
    to_end_only_monitors = set([shorten(monitor.name) for monitor in monitors])

    # Add monitor nodes with black border and text (default)
    for monitor in monitors:
        graph.add_node(pydot.Node(shorten(monitor.name), shape='ellipse', color='black', fontcolor='black'))

    # Add task nodes with black border and text (default)
    for task in tasks:
        graph.add_node(pydot.Node(shorten(task.name), shape='box', color='black', fontcolor='black'))

        for monitor in task.to_start:
            to_start = shorten(monitor.name)
            if to_start:
                graph.add_edge(pydot.Edge(to_start, shorten(task.name), color='green'))
                to_end_only_monitors.discard(to_start)
            else:
                add_t0 = True
                graph.add_edge(pydot.Edge('t_0', shorten(task.name), color='green'))
        for monitor in task.to_hold:
            to_hold = shorten(monitor.name)
            graph.add_edge(pydot.Edge(to_hold, shorten(task.name), color='orange'))
            to_end_only_monitors.discard(to_hold)
        for monitor in task.to_end:
            to_end = shorten(monitor.name)
            graph.add_edge(pydot.Edge(shorten(task.name), to_end, color='red'))


    # Add "t_0" node with red border and black text if needed
    if add_t0:
        t0_node = pydot.Node('t_0', shape='ellipse', color='red', fontcolor='black')
        graph.add_node(t0_node)

    # Update border color for to_end_only_monitors to red
    for monitor_id in to_end_only_monitors:
        monitor_node = pydot.Node(monitor_id, shape='ellipse', color='red', fontcolor='black')
        graph.add_node(monitor_node)

    # Save or show the graph
    create_path(output_file)
    graph.write_png(output_file)
    logging.loginfo(f'Saved task graph at {output_file}.')


class PlotGoalGraph(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot goal graph'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        tasks = []
        for goal in god_map.motion_goal_manager.motion_goals.values():
            tasks.extend(goal.tasks)
        file_name = god_map.giskard.tmp_folder + f'task_graphs/goal_{god_map.goal_id}.png'
        generate_graph(tasks, god_map.monitor_manager.monitors, file_name)
        return Status.SUCCESS
