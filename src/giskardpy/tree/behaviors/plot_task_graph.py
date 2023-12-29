from collections import defaultdict
from typing import List, Dict

import pydot
from py_trees import Status

from giskardpy.goals.collision_avoidance import CollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.goals.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.goals.monitors.payload_monitors import EndMotion
from giskardpy.goals.tasks.task import Task
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard
from giskardpy.utils.utils import create_path, string_shortener


def create_graph(goals: List[Goal], all_monitors: List[Monitor], output_file: str = 'graph.png'):
    graph = pydot.Dot(graph_type='digraph')

    def add_or_get_node(thing):
        node_id = thing.formatted_name(quoted=True)
        nodes = graph.get_node(node_id)
        if not nodes:
            if isinstance(thing, Monitor):
                if isinstance(thing, EndMotion):
                    shape = 'box'
                    color = 'red'
                elif isinstance(thing, ExpressionMonitor):
                    shape = 'box'
                    color = 'black'
                else:  # isinstance(thing, PayloadMonitor)
                    shape = 'octagon'
                    color = 'black'
            else:  # isinstance(thing, Task)
                shape = 'ellipse'
                color = 'black'
            node = pydot.Node(node_id, shape=shape, color=color)
            graph.add_node(node)
        else:
            node = nodes[0]  # Get the first node from the list
        return node

    # Process monitors and their start_monitors
    for monitor in all_monitors:
        monitor_node = add_or_get_node(monitor)
        for sub_monitor in monitor.start_monitors:
            sub_monitor_node = add_or_get_node(sub_monitor)
            graph.add_edge(pydot.Edge(sub_monitor_node, monitor_node, color='green'))

    # Process goals and their connections
    for goal in goals:
        for i, task in enumerate(goal.tasks):
            if isinstance(goal, CollisionAvoidance):
                task = goal
                if i > 0:
                    break
            # else:
            #     task_name = f'{goal.formatted_name}\n---------\n{task.formatted_name}'
            goal_node = add_or_get_node(task)

            for monitor in task.start_monitors:
                monitor_node = add_or_get_node(monitor)
                graph.add_edge(pydot.Edge(monitor_node, goal_node, color='green'))

            for monitor in task.hold_monitors:
                monitor_node = add_or_get_node(monitor)
                graph.add_edge(pydot.Edge(monitor_node, goal_node, color='yellow'))

            for monitor in task.end_monitors:
                monitor_node = add_or_get_node(monitor)
                graph.add_edge(pydot.Edge(goal_node, monitor_node, color='red'))

    graph.write_png(output_file)
    logging.loginfo(f'Saved task graph at {output_file}.')


class PlotTaskMonitorGraph(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot task graph'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        file_name = god_map.giskard.tmp_folder + f'task_graphs/goal_{god_map.goal_id}.png'
        create_graph(god_map.motion_goal_manager.motion_goals.values(), god_map.monitor_manager.monitors, file_name)
        return Status.SUCCESS
