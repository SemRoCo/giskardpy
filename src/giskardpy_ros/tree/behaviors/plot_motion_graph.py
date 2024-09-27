import re
from typing import List, Dict, Tuple, Optional, Union

import pydot
from line_profiler import profile
from py_trees import Status

import giskard_msgs.msg as giskard_msgs
from giskard_msgs.msg import ExecutionState, MotionGraphNode
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


def extract_node_names_from_condition(condition: str) -> List[str]:
    return re.findall(r"'(.*?)'", condition)


def format_condition(condition: str) -> str:
    condition = condition.replace(' and ', '<BR/>       and ')
    condition = condition.replace(' or ', '<BR/>       nor ')
    condition = condition.replace('1.0', 'True')
    condition = condition.replace('0.0', 'False')
    return condition


NotStartedColor = '#9F9F9F'
MyBLUE = '#0000DD'
MyGREEN = '#006600'
MyORANGE = '#996900'
MyRED = '#993000'
MyGRAY = '#E0E0E0'

ChatGPTGreen = '#28A745'
ChatGPTOrange = '#E6AC00'
ChatGPTRed = '#DC3545'
ChatGPTBlue = '#007BFF'
ChatGPTGray = '#6C757D'

StartCondColor = ChatGPTGreen
PauseCondColor = ChatGPTOrange
EndCondColor = ChatGPTRed
ResetCondColor = ChatGPTGray

MonitorTrueGreen = '#B6E5A0'
MonitorFalseRed = '#FF8961'
FONT = 'sans-serif'
LineWidth = 4
NodeSep = 0.5
RankSep = 1
ArrowSize = 1
Fontsize = 15
GoalNodeStyle = 'filled'
GoalNodeShape = 'none'
GoalClusterStyle = 'filled'
MonitorStyle = 'filled, rounded'
MonitorShape = 'rectangle'
TaskStyle = 'filled, diagonals'
TaskShape = 'rectangle'
ConditionFont = 'monospace'
LiftCycleStateToColor: Dict[Tuple[LifeCycleState, int], Tuple[str, str]] = {
    (LifeCycleState.not_started, 1): (ResetCondColor, MonitorTrueGreen),
    (LifeCycleState.running, 1): (StartCondColor, MonitorTrueGreen),
    (LifeCycleState.paused, 1): (PauseCondColor, MonitorTrueGreen),
    (LifeCycleState.succeeded, 1): (EndCondColor, MonitorTrueGreen),
    (LifeCycleState.failed, 1): ('red', MonitorTrueGreen),

    (LifeCycleState.not_started, 0): (ResetCondColor, MonitorFalseRed),
    (LifeCycleState.running, 0): (StartCondColor, MonitorFalseRed),
    (LifeCycleState.paused, 0): (PauseCondColor, MonitorFalseRed),
    (LifeCycleState.succeeded, 0): (EndCondColor, MonitorFalseRed),
    (LifeCycleState.failed, 0): ('red', MonitorFalseRed),
}


class ExecutionStateToDotParser:
    graph: pydot.Graph

    def __init__(self, execution_state: ExecutionState):
        self.execution_state = execution_state
        self.graph = pydot.Dot(graph_type='digraph', graph_name='', ranksep=RankSep, nodesep=NodeSep, compound=True)

    def search_for_monitor(self, monitor_name: str) -> giskard_msgs.MotionGraphNode:
        return [m for m in self.execution_state.monitors if m.name == monitor_name][0]

    def format_motion_graph_node_msg(self, msg: giskard_msgs.MotionGraphNode, color: str) -> str:
        start_condition = format_condition(msg.start_condition)
        pause_condition = format_condition(msg.pause_condition)
        end_condition = format_condition(msg.end_condition)
        reset_condition = format_condition(msg.reset_condition)
        if msg.class_name in {EndMotion.__name__, CancelMotion.__name__}:
            pause_condition = None
            end_condition = None
        return self.conditions_to_str(name=msg.name,
                                      start_condition=start_condition,
                                      pause_condition=pause_condition,
                                      end_condition=end_condition,
                                      reset_condition=reset_condition,
                                      color=color)

    def conditions_to_str(self, name: str, start_condition: str, pause_condition: Optional[str],
                          end_condition: Optional[str], reset_condition: Optional[str],
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
        if reset_condition is not None:
            label += (f'<TR><TD WIDTH="100%" BGCOLOR="{color}" HEIGHT="{LineWidth}"></TD></TR>'
                      f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">reset:{reset_condition}</FONT></TD></TR>')
        label += f'</TABLE>>'
        return label

    def add_boarder_to_node(self, graph: Union[pydot.Graph, pydot.Cluster], node: pydot.Node, num: int, color: str,
                            style: str) -> None:
        child = node
        for i in range(num):
            c = pydot.Cluster(graph_name=f'{node.get_name()}', penwidth=LineWidth, style=style, color=color)
            if i == 0:
                c.add_node(child)
            else:
                c.add_subgraph(child)
            child = c
        if num > 0:
            graph.add_subgraph(c)

    def escape_name(self, name: str) -> str:
        return f'"{name}"'

    def is_goal(self, name: str) -> bool:
        for goal in self.execution_state.goals:
            if goal.name == name:
                return True
        return False

    def get_cluster_of_node(self, node_name: str, graph: Union[pydot.Graph, pydot.Cluster]) -> Optional[pydot.Cluster]:
        node_cluster = None
        for cluster in graph.get_subgraphs():
            if len(cluster.get_node(self.escape_name(node_name))) == 1 or len(cluster.get_node(node_name)) == 1:
                node_cluster = cluster
                break
        return node_cluster

    def add_node(self, graph: pydot.Graph, node_msg: giskard_msgs.MotionGraphNode, style: str, color: str, shape: str,
                 bg_color: str) \
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
        label = self.format_motion_graph_node_msg(node_msg, color)
        node = pydot.Node(node_id,
                          label=label,
                          shape=shape,
                          color=color,
                          style=style,
                          margin=0,
                          fillcolor=bg_color,
                          fontname=FONT,
                          fontsize=Fontsize,
                          penwidth=LineWidth)
        self.add_boarder_to_node(graph=graph, node=node, num=num_extra_boarders, color=color, style=boarder_style)
        if num_extra_boarders == 0:
            graph.add_node(node)
        return node

    def cluster_name_to_goal_name(self, name: str) -> str:
        if name == '':
            return name
        return name[9:-1]

    def to_dot_graph(self, use_state_color: bool = False) -> pydot.Graph:
        self.add_goal_cluster(self.graph, use_state_color)
        return self.graph

    def add_goal_cluster(self, parent_cluster: Union[pydot.Graph, pydot.Cluster], use_state_color: bool = False):
        my_tasks = []
        for i, task in enumerate(self.execution_state.tasks):
            # TODO add one collision avoidance task?
            if self.execution_state.task_parents[i] == self.cluster_name_to_goal_name(parent_cluster.get_name()):
                if use_state_color:
                    color, bg_color = LiftCycleStateToColor[(self.execution_state.task_life_cycle_state[i],
                                                             self.execution_state.task_state[i])]
                else:
                    color, bg_color = 'black', MyGRAY
                self.add_node(parent_cluster, node_msg=task, style=TaskStyle, color=color, bg_color=bg_color,
                              shape=TaskShape)
                my_tasks.append(task)
        my_monitors = []
        for i, monitor in enumerate(self.execution_state.monitors):
            if self.execution_state.monitor_parents[i] == self.cluster_name_to_goal_name(parent_cluster.get_name()):
                if use_state_color:
                    color, bg_color = LiftCycleStateToColor[(self.execution_state.monitor_life_cycle_state[i],
                                                             self.execution_state.monitor_state[i])]
                else:
                    color, bg_color = 'black', 'white'
                self.add_node(parent_cluster, node_msg=monitor, style=MonitorStyle, color=color, bg_color=bg_color,
                              shape=MonitorShape)
                my_monitors.append(monitor)
        my_goals = []
        for i, goal in enumerate(self.execution_state.goals):
            if self.execution_state.goal_parents[i] == self.cluster_name_to_goal_name(parent_cluster.get_name()):
                if use_state_color:
                    color, bg_color = LiftCycleStateToColor[(self.execution_state.monitor_life_cycle_state[i],
                                                             self.execution_state.monitor_state[i])]
                else:
                    color, bg_color = 'black', 'white'
                goal_cluster = pydot.Cluster(graph_name=goal.name,
                                             fontname=FONT,
                                             fontsize=Fontsize,
                                             style=GoalClusterStyle,
                                             color=color,
                                             fillcolor=bg_color,
                                             penwidth=LineWidth)
                self.add_node(graph=goal_cluster, node_msg=goal, style=GoalNodeStyle, color=color, bg_color=bg_color,
                              shape=GoalNodeShape)
                parent_cluster.add_subgraph(goal_cluster)
                self.add_goal_cluster(goal_cluster)
                my_goals.append(goal)
        # %% add edges
        self.add_edges(parent_cluster, my_tasks, my_monitors, my_goals)

    def add_edges(self,
                  graph: Union[pydot.Graph, pydot.Cluster],
                  tasks: List[MotionGraphNode],
                  monitors: List[MotionGraphNode],
                  goals: List[MotionGraphNode]) -> pydot.Graph:
        all_nodes = tasks + monitors + goals
        all_node_name = [node.name for node in all_nodes] + [self.cluster_name_to_goal_name(graph.get_name())]
        for node in all_nodes:
            node_name = node.name
            node_cluster = self.get_cluster_of_node(node_name, graph)
            for sub_node_name in extract_node_names_from_condition(node.start_condition):
                if sub_node_name not in all_node_name:
                    continue
                sub_node_cluster = self.get_cluster_of_node(sub_node_name, graph)
                kwargs = {}
                if node_cluster is not None:
                    kwargs['lhead'] = node_cluster.get_name()
                if sub_node_cluster is not None:
                    kwargs['ltail'] = sub_node_cluster.get_name()
                graph.add_edge(pydot.Edge(src=sub_node_name, dst=node_name, penwidth=LineWidth, color=StartCondColor,
                                          arrowsize=ArrowSize, **kwargs))
            for sub_node_name in extract_node_names_from_condition(node.pause_condition):
                if sub_node_name not in all_node_name:
                    continue
                sub_node_cluster = self.get_cluster_of_node(sub_node_name, graph)
                kwargs = {}
                if node_cluster is not None:
                    kwargs['lhead'] = node_cluster.get_name()
                if sub_node_cluster is not None:
                    kwargs['ltail'] = sub_node_cluster.get_name()
                graph.add_edge(pydot.Edge(sub_node_name, node_name, penwidth=LineWidth, color=PauseCondColor,
                                          minlen=0,
                                          arrowsize=ArrowSize, **kwargs))
            for sub_node_name in extract_node_names_from_condition(node.end_condition):
                if sub_node_name not in all_node_name:
                    continue
                sub_node_cluster = self.get_cluster_of_node(sub_node_name, graph)
                kwargs = {}
                if node_cluster is not None:
                    kwargs['ltail'] = node_cluster.get_name()
                if sub_node_cluster is not None:
                    kwargs['lhead'] = sub_node_cluster.get_name()
                graph.add_edge(pydot.Edge(node_name, sub_node_name, color=EndCondColor, penwidth=LineWidth,
                                          arrowhead='none',
                                          arrowtail='normal',
                                          dir='both', arrowsize=ArrowSize, **kwargs))
            for sub_node_name in extract_node_names_from_condition(node.reset_condition):
                if sub_node_name not in all_node_name:
                    continue
                sub_node_cluster = self.get_cluster_of_node(sub_node_name, graph)
                kwargs = {}
                if node_cluster is not None:
                    kwargs['ltail'] = node_cluster.get_name()
                if sub_node_cluster is not None:
                    kwargs['lhead'] = sub_node_cluster.get_name()
                graph.add_edge(pydot.Edge(node_name, sub_node_name, color=ResetCondColor, penwidth=LineWidth,
                                          arrowhead='none',
                                          arrowtail='normal',
                                          constraint=False,
                                          dir='both', arrowsize=ArrowSize, **kwargs))

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
        parser = ExecutionStateToDotParser(execution_state)
        graph = parser.to_dot_graph()
        create_path(file_name)
        graph.write_pdf(file_name)
        # graph.write_dot(file_name + '.dot')
        get_middleware().loginfo(f'Saved task graph at {file_name}.')
        return Status.SUCCESS
