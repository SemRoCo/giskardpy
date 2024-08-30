import re
from typing import List, Union, Dict, Tuple, Optional

import pydot
from py_trees import Status

import giskard_msgs.msg as giskard_msgs
from giskard_msgs.msg import ExecutionState
from giskardpy.data_types import TaskState
from giskardpy.god_map import god_map
from giskardpy.motion_graph.monitors.payload_monitors import CancelMotion
from giskardpy.motion_graph.monitors.monitor_manager import EndMotion
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.behaviors.publish_feedback import giskard_state_to_execution_state
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard
from giskardpy.utils.utils import create_path, json_str_to_kwargs


def extract_monitor_names_from_condition(condition: str) -> List[str]:
    return re.findall(r"'(.*?)'", condition)


def search_for_monitor(monitor_name: str, execution_state: ExecutionState) -> giskard_msgs.Monitor:
    return [m for m in execution_state.monitors if m.name == monitor_name][0]


task_state_to_color: Dict[TaskState, str] = {
    TaskState.not_started: 'gray',
    TaskState.running: 'green',
    TaskState.on_hold: 'orange',
    TaskState.succeeded: 'palegreen',
    TaskState.failed: 'red'
}

monitor_state_to_color: Dict[Tuple[TaskState, int], str] = {
    (TaskState.not_started, 1): 'darkgreen',
    (TaskState.running, 1): 'green',
    (TaskState.on_hold, 1): 'turquoise',
    (TaskState.succeeded, 1): 'palegreen',
    (TaskState.failed, 1): 'gray',

    (TaskState.not_started, 0): 'darkred',
    (TaskState.running, 0): 'red',
    (TaskState.on_hold, 0): 'orange',
    (TaskState.succeeded, 0): 'lightpink',
    (TaskState.failed, 0): 'black'
}


def format_condition(condition: str) -> str:
    condition = condition.replace(' and ', '<BR/>       and ')
    condition = condition.replace(' or ', '<BR/>       nor ')
    condition = condition.replace('1.0', 'True')
    condition = condition.replace('0.0', 'False')
    return condition


def format_monitor_msg(msg: giskard_msgs.Monitor, color: str, name_bg_color: str) -> str:
    start_condition = format_condition(msg.start_condition)
    kwargs = json_str_to_kwargs(msg.kwargs)
    hold_condition = format_condition(kwargs['hold_condition'])
    end_condition = format_condition(kwargs['end_condition'])
    if msg.monitor_class in {EndMotion.__name__, CancelMotion.__name__}:
        hold_condition = None
        end_condition = None
    return conditions_to_str(msg.name, start_condition, hold_condition, end_condition, color, name_bg_color)


def format_task_msg(msg: giskard_msgs.MotionGoal, color: str, name_bg_color: str) -> str:
    start_condition = format_condition(msg.start_condition)
    hold_condition = format_condition(msg.hold_condition)
    end_condition = format_condition(msg.end_condition)
    return conditions_to_str(msg.name, start_condition, hold_condition, end_condition, color, name_bg_color)


def conditions_to_str(name: str, start_condition: str, pause_condition: Optional[str], end_condition: Optional[str],
                      color: str, name_bg_color: str) -> str:
    label = (f'<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0">'
             f'<TR><TD WIDTH="100%" HEIGHT="{LineWidth}"></TD></TR>'
             f'<TR><TD><B> {name} </B></TD></TR>'
             f'<TR><TD WIDTH="100%" BGCOLOR="{color}" HEIGHT="{LineWidth * 1.5}"></TD></TR>'
             f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">start:{start_condition}</FONT></TD></TR>')
    if pause_condition is not None:
        label += (f'<TR><TD WIDTH="100%" BGCOLOR="{color}" HEIGHT="{LineWidth}"></TD></TR>'
                  f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">pause:{pause_condition}</FONT></TD></TR>')
    if end_condition is not None:
        label += (f'<TR><TD WIDTH="100%" BGCOLOR="{color}" HEIGHT="{LineWidth}"></TD></TR>'
                  f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">end  :{end_condition}</FONT></TD></TR>')
    label += f'</TABLE>>'
    return label


MyBLUE = '#003399'
MyGREEN = '#006600'
MyORANGE = '#996900'
MyRED = '#993000'
MyGRAY = '#E0E0E0'
FONT = 'sans-serif'
LineWidth = 6
ArrowSize = 1.5
Fontsize = 25
ConditionFont = 'monospace'


def format_msg(msg: Union[giskard_msgs.Monitor, giskard_msgs.MotionGoal], color: str, name_bg_color: str) -> str:
    if isinstance(msg, giskard_msgs.MotionGoal):
        return format_task_msg(msg, color, name_bg_color)
    return format_monitor_msg(msg, color, name_bg_color)


def add_boarder_to_node(graph: pydot.Graph, node: pydot.Node, num: int, color: str, style: str) -> None:
    child = node
    for i in range(num):
        c = pydot.Cluster(graph_name=f'{node.get_name()}{i}', penwidth=LineWidth, style=style, color=color)
        if i == 0:
            c.add_node(child)
        else:
            c.add_subgraph(child)
        child = c
    if num > 0:
        graph.add_subgraph(c)


def execution_state_to_dot_graph(execution_state: ExecutionState) -> pydot.Dot:
    graph = pydot.Dot(graph_type='digraph', ranksep=1.5)

    # graph.add_node(make_legend_node())

    def add_or_get_node(thing: Union[giskard_msgs.Monitor, giskard_msgs.MotionGoal]):
        bgcolor = 'white'
        name_bg_color = 'white'
        num_extra_boarders = 0
        node_id = str(thing.name)
        color = 'black'
        boarder_style = 'rounded'
        if isinstance(thing, giskard_msgs.Monitor):
            style = 'filled, rounded'
            # color = MyBLUE
            if thing.monitor_class == EndMotion.__name__:
                # color = MyGREEN
                num_extra_boarders = 1
                boarder_style = 'rounded'
            elif thing.monitor_class == CancelMotion.__name__:
                # color = MyRED
                num_extra_boarders = 1
                boarder_style = 'dashed, rounded'
        else:  # isinstance(thing, Task)
            style = 'filled, diagonals'
            # color = 'black'
            bgcolor = MyGRAY
        label = format_msg(thing, color, name_bg_color)
        nodes = graph.get_node(label)
        if not nodes:
            node = pydot.Node(node_id,
                              label=label,
                              shape='rectangle',
                              color=color,
                              style=style,
                              margin=0,
                              fontname=FONT,
                              fillcolor=bgcolor,
                              fontsize=Fontsize,
                              penwidth=LineWidth)
            add_boarder_to_node(graph=graph, node=node, num=num_extra_boarders, color=color, style=boarder_style)
            graph.add_node(node)
        else:
            node = nodes[0]  # Get the first node from the list
        return node

    # Process monitors and their start_condition
    for i, monitor in enumerate(execution_state.monitors):
        kwargs = json_str_to_kwargs(monitor.kwargs)
        hold_condition = format_condition(kwargs['hold_condition'])
        end_condition = format_condition(kwargs['end_condition'])
        monitor_node = add_or_get_node(monitor)
        # monitor_node.obj_dict['attributes']['color'] = monitor_state_to_color[
        #     (execution_state.monitor_life_cycle_state[i],
        #      execution_state.monitor_state[i])]
        free_symbols = extract_monitor_names_from_condition(monitor.start_condition)
        for sub_monitor_name in free_symbols:
            sub_monitor = search_for_monitor(sub_monitor_name, execution_state)
            sub_monitor_node = add_or_get_node(sub_monitor)
            graph.add_edge(
                pydot.Edge(sub_monitor_node, monitor_node, penwidth=LineWidth, color=MyGREEN, arrowsize=ArrowSize))
        free_symbols = extract_monitor_names_from_condition(hold_condition)
        for sub_monitor_name in free_symbols:
            sub_monitor = search_for_monitor(sub_monitor_name, execution_state)
            sub_monitor_node = add_or_get_node(sub_monitor)
            graph.add_edge(
                pydot.Edge(sub_monitor_node, monitor_node, penwidth=LineWidth, color=MyORANGE, arrowsize=ArrowSize))
        free_symbols = extract_monitor_names_from_condition(end_condition)
        for sub_monitor_name in free_symbols:
            sub_monitor = search_for_monitor(sub_monitor_name, execution_state)
            sub_monitor_node = add_or_get_node(sub_monitor)
            graph.add_edge(
                pydot.Edge(sub_monitor_node, monitor_node, color=MyRED, penwidth=LineWidth, arrowhead='none',
                           arrowtail='normal',
                           dir='both', arrowsize=ArrowSize))

    # Process goals and their connections
    for i, task in enumerate(execution_state.tasks):
        # TODO add one collision avoidance task?
        goal_node = add_or_get_node(task)
        # goal_node.obj_dict['attributes']['color'] = task_state_to_color[execution_state.task_state[i]]
        for monitor_name in extract_monitor_names_from_condition(task.start_condition):
            monitor = search_for_monitor(monitor_name, execution_state)
            monitor_node = add_or_get_node(monitor)
            graph.add_edge(pydot.Edge(monitor_node, goal_node, penwidth=LineWidth, color=MyGREEN, arrowsize=ArrowSize))

        for monitor_name in extract_monitor_names_from_condition(task.hold_condition):
            monitor = search_for_monitor(monitor_name, execution_state)
            monitor_node = add_or_get_node(monitor)
            graph.add_edge(pydot.Edge(monitor_node, goal_node, penwidth=LineWidth, color=MyORANGE, arrowsize=ArrowSize))

        for monitor_name in extract_monitor_names_from_condition(task.end_condition):
            monitor = search_for_monitor(monitor_name, execution_state)
            monitor_node = add_or_get_node(monitor)
            graph.add_edge(pydot.Edge(goal_node, monitor_node, color=MyRED, penwidth=LineWidth, arrowhead='none',
                                      arrowtail='normal',
                                      dir='both', arrowsize=ArrowSize))

    return graph


class PlotTaskMonitorGraph(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot task graph'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        file_name = god_map.giskard.tmp_folder + f'task_graphs/goal_{god_map.move_action_server.goal_id}.pdf'
        execution_state = giskard_state_to_execution_state()
        graph = execution_state_to_dot_graph(execution_state)
        create_path(file_name)
        graph.write_pdf(file_name)
        # graph.write_dot(file_name + '.dot')
        logging.loginfo(f'Saved task graph at {file_name}.')
        return Status.SUCCESS
