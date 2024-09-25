import re
from typing import List, Dict, Tuple, Optional

import pydot
from line_profiler import profile
from py_trees import Status

import giskard_msgs.msg as giskard_msgs
from giskard_msgs.msg import ExecutionState
from giskardpy.data_types.data_types import LifeCycleState
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.motion_graph.monitors.monitors import EndMotion
from giskardpy.motion_graph.monitors.payload_monitors import CancelMotion
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import create_path
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.behaviors.publish_feedback import giskard_state_to_execution_state
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard, GiskardBlackboard

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


def extract_node_names_from_condition(condition: str) -> List[str]:
    return re.findall(r"'(.*?)'", condition)


def search_for_monitor(monitor_name: str, execution_state: ExecutionState) -> giskard_msgs.MotionGraphNode:
    return [m for m in execution_state.monitors if m.name == monitor_name][0]


node_state_to_color: Dict[Tuple[LifeCycleState, int], Tuple[str, str]] = {
    (LifeCycleState.not_started, 1): (NotStartedColor, MonitorTrueGreen),
    (LifeCycleState.running, 1): (MyBLUE, MonitorTrueGreen),
    (LifeCycleState.on_hold, 1): (MyORANGE, MonitorTrueGreen),
    (LifeCycleState.succeeded, 1): (MyGREEN, MonitorTrueGreen),
    (LifeCycleState.failed, 1): ('red', MonitorTrueGreen),

    (LifeCycleState.not_started, 0): (NotStartedColor, MonitorFalseRed),
    (LifeCycleState.running, 0): (MyBLUE, MonitorFalseRed),
    (LifeCycleState.on_hold, 0): (MyORANGE, MonitorFalseRed),
    (LifeCycleState.succeeded, 0): (MyGREEN, MonitorFalseRed),
    (LifeCycleState.failed, 0): ('red', MonitorFalseRed),
}


def format_condition(condition: str) -> str:
    condition = condition.replace(' and ', '<BR/>       and ')
    condition = condition.replace(' or ', '<BR/>       nor ')
    condition = condition.replace('1.0', 'True')
    condition = condition.replace('0.0', 'False')
    return condition


def format_motion_graph_node_msg(msg: giskard_msgs.MotionGraphNode, color: str) -> str:
    start_condition = format_condition(msg.start_condition)
    pause_condition = format_condition(msg.pause_condition)
    end_condition = format_condition(msg.end_condition)
    if msg.class_name in {EndMotion.__name__, CancelMotion.__name__}:
        pause_condition = None
        end_condition = None
    return conditions_to_str(msg.name, start_condition, pause_condition, end_condition, color)


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


def escape_name(name: str) -> str:
    return f'"{name}"'


def is_goal(name: str, execution_state: ExecutionState) -> bool:
    for goal in execution_state.goals:
        if goal.name == name:
            return True
    return False


def get_cluster_of_node(node_name: str, graph: pydot.Graph) -> Optional[str]:
    node_cluster = None
    for cluster in graph.get_subgraph_list():
        if len(cluster.get_node(escape_name(node_name))) == 1:
            node_cluster = cluster.get_name()
            break
    return node_cluster


def is_cluster_root_node(node_name: str, graph: pydot.Graph) -> bool:
    cluster_name = get_cluster_of_node(node_name, graph)
    if cluster_name is None:
        return True
    return escape_name(node_name) == graph.get_subgraph(cluster_name)[0].get_nodes()[0].get_name()


def belong_to_same_cluster(node1: str, node2: str, graph: pydot.Graph) -> bool:
    node1_cluster = get_cluster_of_node(node1, graph)
    node2_cluster = get_cluster_of_node(node2, graph)
    return node1_cluster == node2_cluster or node1_cluster is None and node2_cluster is None


def execution_state_to_dot_graph(execution_state: ExecutionState, use_state_color: bool = False) -> pydot.Dot:
    graph = pydot.Dot(graph_type='digraph', ranksep=1., compound=True)

    def add_node(node_msg: giskard_msgs.MotionGraphNode, style: str, color: str, bg_color: str) \
            -> pydot.Node:
        num_extra_boarders = 0
        node_id = str(node_msg.name)
        boarder_style = 'rounded'
        if node_msg.class_name == EndMotion.__name__:
            num_extra_boarders = 1
            boarder_style = 'rounded'
        elif node_msg.class_name == CancelMotion.__name__:
            num_extra_boarders = 1
            boarder_style = 'dashed, rounded'
        label = format_motion_graph_node_msg(node_msg, color)
        node = pydot.Node(node_id,
                          label=label,
                          shape='rectangle',
                          color=color,
                          style=style,
                          margin=0,
                          fillcolor=bg_color,
                          fontname=FONT,
                          fontsize=Fontsize,
                          penwidth=LineWidth)
        add_boarder_to_node(graph=graph, node=node, num=num_extra_boarders, color=color, style=boarder_style)
        graph.add_node(node)
        return node

    # %% add all nodes
    # Process monitors and their conditions
    style = 'filled, rounded'
    for i, monitor in enumerate(execution_state.monitors):
        if use_state_color:
            color, bg_color = node_state_to_color[(execution_state.monitor_life_cycle_state[i],
                                                   execution_state.monitor_state[i])]
        else:
            color, bg_color = 'black', 'white'
        add_node(node_msg=monitor, style=style, color=color, bg_color=bg_color)

    # Process tasks and their connections
    style = 'filled, diagonals'
    for i, task in enumerate(execution_state.tasks):
        # TODO add one collision avoidance task?
        if use_state_color:
            color, bg_color = node_state_to_color[(execution_state.task_life_cycle_state[i],
                                                   execution_state.task_state[i])]
        else:
            color, bg_color = 'black', MyGRAY
        add_node(node_msg=task, style=style, color=color, bg_color=bg_color)

    # Process goals and their connections
    style = 'filled, diagonals'
    for i, goal in enumerate(execution_state.goals):
        goal_cluster = pydot.Cluster(graph_name=goal.name,
                                     fontname=FONT,
                                     fontsize=Fontsize,
                                     penwidth=LineWidth)
        goal_cluster_node = add_node(goal, style, 'black', 'white')
        goal_cluster.add_node(goal_cluster_node)
        goal_obj = god_map.motion_graph_manager.goal_state.get_node(goal.name)
        for node in goal_obj.tasks + goal_obj.monitors:
            graph_node = graph.get_node(f'"{node.name}"')[0]
            goal_cluster.add_node(graph_node)
            goal_cluster.add_edge(pydot.Edge(goal_cluster_node, graph_node, style='invis'))
        graph.add_subgraph(goal_cluster)

    # %% add edges
    add_edges(graph, execution_state)
    return graph


def should_draw_edge(node1_name: str, node1_cluster: str, node2_name: str, node2_cluster: str, graph: pydot.Graph) \
        -> bool:
    return (node1_cluster is None and node2_cluster is None  # both outside of cluster
            or node1_cluster is None and is_cluster_root_node(node2_name, graph)  # one in cluster
            or is_cluster_root_node(node1_name, graph) and node2_cluster is None  # one in cluster
            or is_cluster_root_node(node1_name, graph) and is_cluster_root_node(node2_name, graph)  # both cluster root
            or (node1_cluster == node2_cluster and  # both in cluster but both not roots
                (not is_cluster_root_node(node1_name, graph) and not is_cluster_root_node(node2_name, graph))))

def add_edges(graph: pydot.Graph, execution_state: ExecutionState) -> pydot.Graph:
    for i, node in enumerate(execution_state.monitors + execution_state.tasks + execution_state.goals):
        node_name = node.name
        node_cluster = get_cluster_of_node(node_name, graph)
        free_symbols = extract_node_names_from_condition(node.start_condition)
        for sub_node_name in free_symbols:
            sub_node_cluster = get_cluster_of_node(sub_node_name, graph)
            if should_draw_edge(node_name, node_cluster, sub_node_name, sub_node_cluster, graph):
                if (node_cluster is not None and is_cluster_root_node(node_name, graph)
                        and sub_node_cluster is not None and is_cluster_root_node(sub_node_name, graph)):
                    kwargs = {'lhead': node_cluster, 'ltail': sub_node_cluster}
                else:
                    kwargs = {}
                graph.add_edge(pydot.Edge(src=sub_node_name, dst=node_name, penwidth=LineWidth, color=MyGREEN,
                                          arrowsize=ArrowSize, **kwargs))
        free_symbols = extract_node_names_from_condition(node.pause_condition)
        for sub_node_name in free_symbols:
            sub_node_cluster = get_cluster_of_node(sub_node_name, graph)
            if should_draw_edge(node_name, node_cluster, sub_node_name, sub_node_cluster, graph):
                graph.add_edge(pydot.Edge(sub_node_name, node_name, penwidth=LineWidth, color=MyORANGE,
                                          arrowsize=ArrowSize))
        free_symbols = extract_node_names_from_condition(node.end_condition)
        for sub_node_name in free_symbols:
            sub_node_cluster = get_cluster_of_node(sub_node_name, graph)
            if should_draw_edge(node_name, node_cluster, sub_node_name, sub_node_cluster, graph):
                graph.add_edge(pydot.Edge(node_name, sub_node_name, color=MyRED, penwidth=LineWidth,
                                          arrowhead='none',
                                          arrowtail='normal',
                                          dir='both', arrowsize=ArrowSize))

    return graph


class PlotMotionGraph(GiskardBehavior):

    @profile
    def __init__(self, name: str = 'plot task graph'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        file_name = god_map.tmp_folder + f'task_graphs/goal_{GiskardBlackboard().move_action_server.goal_id}.pdf'
        execution_state = giskard_state_to_execution_state()
        graph = execution_state_to_dot_graph(execution_state)
        create_path(file_name)
        graph.write_pdf(file_name)
        # graph.write_dot(file_name + '.dot')
        get_middleware().loginfo(f'Saved task graph at {file_name}.')
        return Status.SUCCESS
