from collections import defaultdict
from typing import List, Dict

import pydot
from py_trees import Status

from giskardpy.goals.collision_avoidance import CollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.tasks.task import Task
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard
from giskardpy.utils.utils import create_path, string_shortener


def generate_graph(goals: Dict[str, Goal], monitors: List[Monitor], output_file: str = 'graph.png'):
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
    monitor_nodes = {}
    for monitor in monitors:
        monitor_name = shorten(monitor.name)
        monitor_nodes[monitor_name] = pydot.Node(monitor_name, shape='ellipse', color='black', fontcolor='black')

    # Add task nodes with black border and text (default)
    for goal_name, goal in goals.items():
        for i, task in enumerate(goal.tasks):
            if isinstance(goal, CollisionAvoidance):
                task_name = string_shortener(f'{goal_name}', max_lines=5, max_line_length=50)
                if i > 0:
                    break
            else:
                task_name = shorten(f'{goal_name} - {task.name}')
            graph.add_node(pydot.Node(task_name, shape='box', color='black', fontcolor='black'))

            for monitor in task.to_start:
                to_start = shorten(monitor.name)
                if to_start:
                    graph.add_edge(pydot.Edge(to_start, task_name, color='green'))
                    to_end_only_monitors.discard(to_start)
                    if to_start in monitor_nodes:
                        graph.add_node(monitor_nodes[to_start])
                        del monitor_nodes[to_start]
                else:
                    add_t0 = True
                    graph.add_edge(pydot.Edge('t_0', task_name, color='green'))
            for monitor in task.to_hold:
                to_hold = shorten(monitor.name)
                graph.add_edge(pydot.Edge(to_hold, task_name, color='orange'))
                to_end_only_monitors.discard(to_hold)
                if to_hold in monitor_nodes:
                    graph.add_node(monitor_nodes[to_hold])
                    del monitor_nodes[to_hold]
            for monitor in task.to_end:
                to_end = shorten(monitor.name)
                graph.add_edge(pydot.Edge(task_name, to_end, color='red'))
                if to_end in monitor_nodes:
                    graph.add_node(monitor_nodes[to_end])
                    del monitor_nodes[to_end]


    # Add "t_0" node with red border and black text if needed
    if add_t0:
        t0_node = pydot.Node('t_0', shape='ellipse', color='red', fontcolor='black')
        graph.add_node(t0_node)

    # Update border color for to_end_only_monitors to red
    for monitor_id in to_end_only_monitors:
        if monitor_id not in monitor_nodes: # this means they've been added
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
        file_name = god_map.giskard.tmp_folder + f'task_graphs/goal_{god_map.goal_id}.png'
        generate_graph(god_map.motion_goal_manager.motion_goals, god_map.monitor_manager.monitors, file_name)
        return Status.SUCCESS
