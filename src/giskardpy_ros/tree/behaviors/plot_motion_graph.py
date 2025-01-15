# -*- coding: utf-8 -*-

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

import pydot
from line_profiler import profile
from py_trees import Status

import giskard_msgs.msg as giskard_msgs
from giskard_msgs.msg import ExecutionState, MotionStatechartNode
from giskardpy.data_types.data_types import LifeCycleState, ObservationState
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.monitors.monitors import EndMotion, CancelMotion
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import create_path
from giskardpy_ros.ros1.msg_converter import json_str_to_ros_kwargs
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.behaviors.publish_feedback import giskard_state_to_execution_state
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard, GiskardBlackboard


def extract_node_names_from_condition(condition: str) -> List[str]:
    return set(re.findall(r"'(.*?)'", condition))


def format_condition(condition: str) -> str:
    condition = condition.replace(' and ', '<BR/>       and ')
    condition = condition.replace(' or ', '<BR/>       or ')
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
ChatGPTGray = '#8F959E'

StartCondColor = ChatGPTGreen
PauseCondColor = ChatGPTOrange
EndCondColor = ChatGPTRed
ResetCondColor = ChatGPTGray

MonitorTrueGreen = '#B6E5A0'
MonitorFalseRed = '#FF5024'
FONT = 'sans-serif'
LineWidth = 4
NodeSep = 1
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

ResetSymbol = '⟲'

ObservationStateToColor: Dict[ObservationState, str] = {
    ObservationState.unknown: ResetCondColor,
    ObservationState.true: MonitorTrueGreen,
    ObservationState.false: MonitorFalseRed
}

ObservationStateToSymbol: Dict[ObservationState, str] = {
    ObservationState.unknown: '?',
    ObservationState.true: 'True',
    ObservationState.false: 'False'
}

ObservationStateToEdgeStyle: Dict[ObservationState, Dict[str, str]] = {
    ObservationState.unknown: {'penwidth': (LineWidth * 1.5) / 2,
                               # 'label': '<<FONT FACE="monospace"><B>?</B></FONT>>',
                               'fontsize': Fontsize * 1.333},
    ObservationState.true: {'penwidth': LineWidth * 1.5},
    ObservationState.false: {'style': 'dashed',
                             'penwidth': LineWidth * 1.5}
}

LiftCycleStateToColor: Dict[LifeCycleState, str] = {
    LifeCycleState.not_started: ResetCondColor,
    LifeCycleState.running: StartCondColor,
    LifeCycleState.paused: PauseCondColor,
    LifeCycleState.succeeded: EndCondColor,
    LifeCycleState.failed: 'red',
}

LiftCycleStateToSymbol: Dict[LifeCycleState, str] = {
    # LifeCycleState.not_started: '○',
    LifeCycleState.not_started: '—',
    LifeCycleState.running: '▶',
    # LifeCycleState.paused: '⏸',
    LifeCycleState.paused: '<B>||</B>',
    LifeCycleState.succeeded: '■',
    LifeCycleState.failed: 'red',
}


class ExecutionStateToDotParser:
    graph: pydot.Graph
    compact: bool

    def __init__(self, execution_state: ExecutionState, compact: bool = False):
        self.compact = compact
        self.execution_state = execution_state
        self.graph = pydot.Dot(graph_type='digraph', graph_name='',
                               ranksep=RankSep if not self.compact else RankSep*0.5,
                               nodesep=NodeSep if not self.compact else NodeSep*0.5,
                               compound=True,
                               ratio='compress')

    def search_for_monitor(self, monitor_name: str) -> giskard_msgs.MotionStatechartNode:
        return [m for m in self.execution_state.monitors if m.name == monitor_name][0]

    def format_motion_graph_node_msg(self, msg: giskard_msgs.MotionStatechartNode,
                                     obs_state: ObservationState, life_cycle_state: LifeCycleState) -> str:
        start_condition = format_condition(msg.start_condition)
        pause_condition = format_condition(msg.pause_condition)
        end_condition = format_condition(msg.end_condition)
        reset_condition = format_condition(msg.reset_condition)
        if msg.class_name in {EndMotion.__name__, CancelMotion.__name__}:
            pause_condition = None
            end_condition = None
            reset_condition = None
        return self.conditions_to_str(name=msg.name,
                                      start_condition=start_condition,
                                      pause_condition=pause_condition,
                                      end_condition=end_condition,
                                      reset_condition=reset_condition,
                                      obs_state=obs_state,
                                      life_cycle_state=life_cycle_state)

    def conditions_to_str(self, name: str, start_condition: str, pause_condition: Optional[str],
                          end_condition: Optional[str], reset_condition: Optional[str],
                          obs_state: ObservationState, life_cycle_state: LifeCycleState) -> str:
        line_color = 'black'
        obs_color = ObservationStateToColor[obs_state]
        obs_text = ObservationStateToSymbol[obs_state]
        life_color = LiftCycleStateToColor[life_cycle_state]
        life_symbol = LiftCycleStateToSymbol[life_cycle_state]
        label = (f'<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                 f'<TR>'
                 f'  <TD WIDTH="100%" HEIGHT="{LineWidth}"></TD>'
                 f'</TR>'
                 f'<TR>'
                 f'  <TD><B> {name} </B></TD>'
                 f'</TR>'
                 f'<TR>'
                 f'  <TD CELLPADDING="0">'
                 f'    <TABLE BORDER="0" CELLBORDER="2" CELLSPACING="0" WIDTH="100%">'
                 f'      <TR>'
                 f'        <TD BGCOLOR="{obs_color}" WIDTH="50%" FIXEDSIZE="FALSE"><FONT FACE="monospace">{obs_text}</FONT></TD>'
                 f'        <VR/>'
                 f'        <TD BGCOLOR="{life_color}" WIDTH="50%" FIXEDSIZE="FALSE"><FONT FACE="monospace">{life_symbol}</FONT></TD>'
                 f'      </TR>'
                 f'    </TABLE>'
                 f'  </TD>'
                 f'</TR>')
        if self.compact:
            label += (f'<TR>'
                      f'  <TD WIDTH="100%" HEIGHT="{LineWidth*2.5}"></TD>'
                      f'</TR>')
        else:
            if start_condition is not None:
                label += (f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
                          f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">start:{start_condition}</FONT></TD></TR>')
            if pause_condition is not None:
                label += (f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
                          f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">pause:{pause_condition}</FONT></TD></TR>')
            if end_condition is not None:
                label += (f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
                          f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">end  :{end_condition}</FONT></TD></TR>')
            if reset_condition is not None:
                label += (f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
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

    def add_node(self, graph: pydot.Graph, node_msg: giskard_msgs.MotionStatechartNode, style: str, shape: str,
                 obs_state: ObservationState, life_cycle_state: LifeCycleState) \
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
        label = self.format_motion_graph_node_msg(msg=node_msg, obs_state=obs_state, life_cycle_state=life_cycle_state)
        node = pydot.Node(node_id,
                          label=label,
                          shape=shape,
                          color='black',
                          style=style,
                          margin=0,
                          fillcolor='white',
                          fontname=FONT,
                          fontsize=Fontsize,
                          penwidth=LineWidth)
        self.add_boarder_to_node(graph=graph, node=node, num=num_extra_boarders, color='black', style=boarder_style)
        if num_extra_boarders == 0:
            graph.add_node(node)
        return node

    def cluster_name_to_goal_name(self, name: str) -> str:
        if name == '':
            return name
        if '"' in name:
            return name[9:-1]
        return name[8:]

    def to_dot_graph(self) -> pydot.Graph:
        obs_states: Dict[str, ObservationState] = {}
        self.add_goal_cluster(self.graph, obs_states)
        return self.graph

    def add_goal_cluster(self, parent_cluster: Union[pydot.Graph, pydot.Cluster],
                         obs_states: Dict[str, ObservationState]):
        my_tasks: List[MotionStatechartNode] = []
        for i, task in enumerate(self.execution_state.tasks):
            # TODO add one collision avoidance task?
            if self.execution_state.task_parents[i] == self.cluster_name_to_goal_name(parent_cluster.get_name()):
                obs_state = self.execution_state.task_state[i]
                self.add_node(parent_cluster, node_msg=task, style=TaskStyle, shape=TaskShape,
                              obs_state=obs_state,
                              life_cycle_state=self.execution_state.task_life_cycle_state[i])
                my_tasks.append(task)
                obs_states[task.name] = obs_state
        my_monitors: List[MotionStatechartNode] = []
        for i, monitor in enumerate(self.execution_state.monitors):
            if self.execution_state.monitor_parents[i] == self.cluster_name_to_goal_name(parent_cluster.get_name()):
                obs_state = self.execution_state.monitor_state[i]
                life_cycle_state = self.execution_state.monitor_life_cycle_state[i]
                self.add_node(parent_cluster, node_msg=monitor, style=MonitorStyle, shape=MonitorShape,
                              obs_state=obs_state,
                              life_cycle_state=life_cycle_state)
                my_monitors.append(monitor)
                obs_states[monitor.name] = obs_state
        my_goals: List[MotionStatechartNode] = []
        for i, goal in enumerate(self.execution_state.goals):
            if self.execution_state.goal_parents[i] == self.cluster_name_to_goal_name(parent_cluster.get_name()):
                obs_state = self.execution_state.goal_state[i]
                goal_cluster = pydot.Cluster(graph_name=goal.name,
                                             fontname=FONT,
                                             fontsize=Fontsize,
                                             style=GoalClusterStyle,
                                             color='black',
                                             fillcolor='white',
                                             penwidth=LineWidth)
                self.add_node(graph=goal_cluster, node_msg=goal, style=GoalNodeStyle, shape=GoalNodeShape,
                              obs_state=obs_state,
                              life_cycle_state=self.execution_state.goal_life_cycle_state[i])
                obs_states[goal.name] = obs_state
                parent_cluster.add_subgraph(goal_cluster)
                self.add_goal_cluster(goal_cluster, obs_states)
                my_goals.append(goal)
        # %% add edges
        self.add_edges(parent_cluster, my_tasks, my_monitors, my_goals, obs_states)

    def add_edges(self,
                  graph: Union[pydot.Graph, pydot.Cluster],
                  tasks: List[MotionStatechartNode],
                  monitors: List[MotionStatechartNode],
                  goals: List[MotionStatechartNode],
                  obs_states: Dict[str, ObservationState]) -> pydot.Graph:
        all_nodes = tasks + monitors + goals
        all_node_name = [node.name for node in all_nodes]  # + [self.cluster_name_to_goal_name(graph.get_name())]
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
                kwargs.update(ObservationStateToEdgeStyle[obs_states[sub_node_name]])
                graph.add_edge(pydot.Edge(src=sub_node_name, dst=node_name, color=StartCondColor,
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
                kwargs.update(ObservationStateToEdgeStyle[obs_states[sub_node_name]])
                graph.add_edge(pydot.Edge(sub_node_name, node_name, color=PauseCondColor,
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
                kwargs.update(ObservationStateToEdgeStyle[obs_states[sub_node_name]])
                graph.add_edge(pydot.Edge(node_name, sub_node_name, color=EndCondColor,
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
                kwargs.update(ObservationStateToEdgeStyle[obs_states[sub_node_name]])
                graph.add_edge(pydot.Edge(node_name, sub_node_name, color=ResetCondColor,
                                          arrowhead='none',
                                          arrowtail='normal',
                                          minlen=0,
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
