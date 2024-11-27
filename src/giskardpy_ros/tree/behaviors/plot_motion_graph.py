import re
from typing import List, Union, Dict, Tuple, Optional

import pydot
from line_profiler import profile
from py_trees import Status

import giskard_msgs.msg as giskard_msgs
from giskard_msgs.msg import ExecutionState
from giskardpy.data_types.data_types import TaskState
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.motion_graph.monitors.payload_monitors import CancelMotion
from giskardpy.motion_graph.monitors.monitor_manager import EndMotion
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import create_path
from giskardpy_ros.ros1.msg_converter import json_str_to_ros_kwargs
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.behaviors.publish_feedback import giskard_state_to_execution_state
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard

MyBLUE = '#0000DD'
MyGREEN = '#006600'
MyORANGE = '#996900'
MyRED = '#993000'
MyGRAY = '#E0E0E0'
MonitorTrueGreen = '#B6E5A0'
MonitorFalseRed = '#FF8961'
FONT = 'sans-serif'
LineWidth = 6
ArrowSize = 1.5
Fontsize = 25
ConditionFont = 'monospace'
NotStartedColor = '#9F9F9F'


def extract_monitor_names_from_condition(condition: str) -> List[str]:
    return re.findall(r"'(.*?)'", condition)


def search_for_monitor(monitor_name: str, execution_state: ExecutionState) -> giskard_msgs.Monitor:
    return [m for m in execution_state.monitors if m.name == monitor_name][0]


task_state_to_color: Dict[TaskState, Tuple[str, str]] = {
    TaskState.not_started: (NotStartedColor, MyGRAY),
    TaskState.running: (MyBLUE, MyGRAY),
    TaskState.on_hold: (MyORANGE, MyGRAY),
    TaskState.succeeded: (MyGREEN, MyGRAY),
    TaskState.failed: ('red', MyGRAY)
}

monitor_state_to_color: Dict[Tuple[TaskState, int], Tuple[str, str]] = {
    (TaskState.not_started, 1): (NotStartedColor, MonitorTrueGreen),
    (TaskState.running, 1): (MyBLUE, MonitorTrueGreen),
    (TaskState.on_hold, 1): (MyORANGE, MonitorTrueGreen),
    (TaskState.succeeded, 1): (MyGREEN, MonitorTrueGreen),
    (TaskState.failed, 1): ('red', MonitorTrueGreen),

    (TaskState.not_started, 0): (NotStartedColor, MonitorFalseRed),
    (TaskState.running, 0): (MyBLUE, MonitorFalseRed),
    (TaskState.on_hold, 0): (MyORANGE, MonitorFalseRed),
    (TaskState.succeeded, 0): (MyGREEN, MonitorFalseRed),
    (TaskState.failed, 0): ('red', MonitorFalseRed),
}


def format_condition(condition: str) -> str:
    condition = condition.replace(' and ', '<BR/>       and ')
    condition = condition.replace(' or ', '<BR/>       nor ')
    condition = condition.replace('1.0', 'True')
    condition = condition.replace('0.0', 'False')
    return condition


def format_monitor_msg(msg: giskard_msgs.Monitor, color: str) -> str:
    start_condition = format_condition(msg.start_condition)
    kwargs = json_str_to_ros_kwargs(msg.kwargs)
    hold_condition = format_condition(kwargs['hold_condition'])
    end_condition = format_condition(kwargs['end_condition'])
    if msg.monitor_class in {EndMotion.__name__, CancelMotion.__name__}:
        hold_condition = None
        end_condition = None
    return conditions_to_str(msg.name, start_condition, hold_condition, end_condition, color)


def format_task_msg(msg: giskard_msgs.MotionGoal, color: str) -> str:
    start_condition = format_condition(msg.start_condition)
    hold_condition = format_condition(msg.hold_condition)
    end_condition = format_condition(msg.end_condition)
    return conditions_to_str(msg.name, start_condition, hold_condition, end_condition, color)


def conditions_to_str(name: str, start_condition: str, pause_condition: Optional[str], end_condition: Optional[str],
                      color: str) -> str:
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


def format_msg(msg: Union[giskard_msgs.Monitor, giskard_msgs.MotionGoal], color: str) -> str:
    if isinstance(msg, giskard_msgs.MotionGoal):
        return format_task_msg(msg, color)
    return format_monitor_msg(msg, color)


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


def execution_state_to_dot_graph(execution_state: ExecutionState, use_state_color: bool = False) -> pydot.Dot:
    graph = pydot.Dot(graph_type='digraph', ranksep=1.)

    def add_node(thing: Union[giskard_msgs.Monitor, giskard_msgs.MotionGoal], color: str, bg_color: str) \
            -> pydot.Node:
        num_extra_boarders = 0
        node_id = str(thing.name)
        boarder_style = 'rounded'
        if isinstance(thing, giskard_msgs.Monitor):
            style = 'filled, rounded'
            if thing.monitor_class == EndMotion.__name__:
                num_extra_boarders = 1
                boarder_style = 'rounded'
            elif thing.monitor_class == CancelMotion.__name__:
                num_extra_boarders = 1
                boarder_style = 'dashed, rounded'
        else:  # isinstance(thing, Task)
            style = 'filled, diagonals'
        label = format_msg(thing, color)
        node = pydot.Node(node_id,
                          label=label,
                          shape='rectangle',
                          color=color,
                          style=style,
                          margin=0,
                          fontname=FONT,
                          fillcolor=bg_color,
                          fontsize=Fontsize,
                          penwidth=LineWidth)
        add_boarder_to_node(graph=graph, node=node, num=num_extra_boarders, color=color, style=boarder_style)
        graph.add_node(node)
        return node

    # Process monitors and their conditions
    for i, monitor in enumerate(execution_state.monitors):
        if use_state_color:
            color, bg_color = monitor_state_to_color[(execution_state.monitor_life_cycle_state[i],
                                                      execution_state.monitor_state[i])]
        else:
            color, bg_color = 'black', 'white'
        kwargs = json_str_to_ros_kwargs(monitor.kwargs)
        hold_condition = format_condition(kwargs['hold_condition'])
        end_condition = format_condition(kwargs['end_condition'])
        monitor_node = add_node(monitor, color, bg_color)
        free_symbols = extract_monitor_names_from_condition(monitor.start_condition)
        for sub_monitor_name in free_symbols:
            graph.add_edge(pydot.Edge(sub_monitor_name, monitor_node, penwidth=LineWidth, color=MyGREEN,
                                      arrowsize=ArrowSize))
        free_symbols = extract_monitor_names_from_condition(hold_condition)
        for sub_monitor_name in free_symbols:
            graph.add_edge(pydot.Edge(sub_monitor_name, monitor_node, penwidth=LineWidth, color=MyORANGE,
                                      arrowsize=ArrowSize))
        free_symbols = extract_monitor_names_from_condition(end_condition)
        for sub_monitor_name in free_symbols:
            graph.add_edge(pydot.Edge(monitor_node, sub_monitor_name, color=MyRED, penwidth=LineWidth, arrowhead='none',
                                      arrowtail='normal',
                                      dir='both', arrowsize=ArrowSize))

    # Process goals and their connections
    for i, task in enumerate(execution_state.tasks):
        # TODO add one collision avoidance task?
        if use_state_color:
            color, bg_color = task_state_to_color[execution_state.task_state[i]]
        else:
            color, bg_color = 'black', MyGRAY
        goal_node = add_node(task, color, bg_color)
        for monitor_name in extract_monitor_names_from_condition(task.start_condition):
            graph.add_edge(pydot.Edge(monitor_name, goal_node, penwidth=LineWidth, color=MyGREEN, arrowsize=ArrowSize))

        for monitor_name in extract_monitor_names_from_condition(task.hold_condition):
            graph.add_edge(pydot.Edge(monitor_name, goal_node, penwidth=LineWidth, color=MyORANGE, arrowsize=ArrowSize))

        for monitor_name in extract_monitor_names_from_condition(task.end_condition):
            graph.add_edge(pydot.Edge(goal_node, monitor_name, color=MyRED, penwidth=LineWidth, arrowhead='none',
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
        get_middleware().loginfo(f'Saved task graph at {file_name}.')
        return Status.SUCCESS
