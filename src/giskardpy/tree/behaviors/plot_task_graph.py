from typing import List, Union

import pydot
from py_trees import Status

import giskardpy.casadi_wrapper as cas
from giskardpy.goals.collision_avoidance import CollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.monitors.payload_monitors import EndMotion, CancelMotion
from giskardpy.tasks.task import Task
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


def create_graph(goals: List[Goal], all_monitors: List[Monitor], output_file: str = 'graph.png'):
    graph = pydot.Dot(graph_type='digraph')

    def add_or_get_node(thing: Union[Monitor, Task]):
        node_id = thing.formatted_name(quoted=True)
        nodes = graph.get_node(node_id)
        if not nodes:
            if isinstance(thing, Monitor):
                if isinstance(thing, EndMotion):
                    shape = 'octagon'
                    color = 'red'
                elif isinstance(thing, CancelMotion):
                    shape = 'octagon'
                    color = 'orange'
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

    # Process monitors and their start_condition
    for monitor in all_monitors:
        if monitor.plot:
            monitor_node = add_or_get_node(monitor)
            for expr in monitor.start_condition.free_symbols():
                sub_monitor = god_map.monitor_manager.get_monitor_from_state_expr(expr)
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

            for expr in task.start_condition.free_symbols():
                monitor = god_map.monitor_manager.get_monitor_from_state_expr(expr)
                monitor_node = add_or_get_node(monitor)
                graph.add_edge(pydot.Edge(monitor_node, goal_node, color='green'))

            for expr in task.hold_condition.free_symbols():
                monitor = god_map.monitor_manager.get_monitor_from_state_expr(expr)
                monitor_node = add_or_get_node(monitor)
                graph.add_edge(pydot.Edge(monitor_node, goal_node, color='yellow'))

            for expr in task.end_condition.free_symbols():
                monitor = god_map.monitor_manager.get_monitor_from_state_expr(expr)
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
