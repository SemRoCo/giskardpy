from collections import defaultdict
from typing import List

import pydot
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import JointStates
from giskardpy.exceptions import ExecutionException
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.tasks.task import Task
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard
from giskardpy.utils.utils import create_path


def generate_graph(tasks: List[Task], monitors: List[Monitor], output_file: str = 'graph.png'):
    # Initialize graph
    graph = pydot.Dot(graph_type='digraph')

    # Flags to track special cases
    add_t0 = False
    to_end_only_monitors = set([monitor.name for monitor in monitors])

    # Add monitor nodes with black border and text (default)
    for monitor in monitors:
        graph.add_node(pydot.Node(monitor.name, shape='ellipse', color='black', fontcolor='black'))

    # Add task nodes with black border and text (default)
    for task in tasks:
        graph.add_node(pydot.Node(task.name, shape='box', color='black', fontcolor='black'))

        if task.to_start is not None:
            to_start = task.to_start.name
        else:
            to_start = None
        if task.to_end is not None:
            to_end = task.to_end.name
        else:
            to_end = None
        if task.to_hold is not None:
            to_hold = task.to_hold.name
        else:
            to_hold = None

        if to_start:
            graph.add_edge(pydot.Edge(to_start, task.name, color='green'))
            to_end_only_monitors.discard(to_start)
        else:
            add_t0 = True
            graph.add_edge(pydot.Edge('t_0', task.name, color='green'))

        if to_end:
            graph.add_edge(pydot.Edge(task.name, to_end, color='red'))

        if to_hold:
            graph.add_edge(pydot.Edge(to_hold, task.name, color='orange'))
            to_end_only_monitors.discard(to_hold)

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
    logging.loginfo(f'saved task graph at {output_file}')


class PlotGoalGraph(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot goal graph'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        tasks = []
        for goal in self.god_map.get_data(identifier.goals).values():
            tasks.extend(goal.tasks)
        file_name = self.god_map.get_data(identifier.tmp_folder) + f'/task_graphs/goal_{self.goal_id}.png'
        generate_graph(tasks, self.monitors, file_name)
        return Status.SUCCESS
