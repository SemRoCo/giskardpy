from __future__ import annotations

import re
from dataclasses import dataclass, field

import pydot
from typing_extensions import (
    List,
    Dict,
    Optional,
    Union,
    Set,
    TYPE_CHECKING,
)

from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.graph_node import (
    Goal,
    TrinaryCondition,
)
from giskardpy.motion_statechart.plotters.plot_specs import (
    TRANSITION_SPECS,
    EdgeSpec,
    StateSelector,
)
from giskardpy.motion_statechart.plotters.styles import (
    RankSep,
    NodeSep,
    ObservationStateToColor,
    ObservationStateToSymbol,
    LiftCycleStateToColor,
    LiftCycleStateToSymbol,
    LineWidth,
    ConditionFont,
    FONT,
    Fontsize,
    GoalClusterStyle,
    ObservationStateToEdgeStyle,
    ArrowSize,
)

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart


def extract_node_names_from_condition(condition: str) -> Set[str]:
    matches = re.findall(r'"(.*?)"|\'(.*?)\'', condition)
    return set(match for group in matches for match in group if match)


def format_condition(condition: str) -> str:
    condition = condition.replace(" and ", "<BR/>       and ")
    condition = condition.replace(" or ", "<BR/>       or ")
    condition = condition.replace("1.0", "True")
    condition = condition.replace("0.0", "False")
    return condition


@dataclass
class MotionStatechartGraphviz:
    motion_statechart: MotionStatechart
    graph: pydot.Graph = field(init=False)
    compact: bool = False
    _cluster_map: Dict[MotionStatechartNode, pydot.Cluster] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self):
        self.graph = pydot.Dot(
            graph_type="digraph",
            graph_name="",
            ranksep=RankSep if not self.compact else RankSep * 0.5,
            nodesep=NodeSep if not self.compact else NodeSep * 0.5,
            compound=True,
            ratio="compress",
        )

    def _format_motion_graph_node(
        self,
        node: MotionStatechartNode,
    ) -> str:
        obs_state = self.motion_statechart.observation_state[node]
        life_cycle_state = self.motion_statechart.life_cycle_state[node]
        obs_color = ObservationStateToColor[obs_state]
        obs_text = ObservationStateToSymbol[obs_state]
        life_color = LiftCycleStateToColor[life_cycle_state]
        life_symbol = LiftCycleStateToSymbol[life_cycle_state]
        label = (
            f'<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            f"<TR>"
            f'  <TD WIDTH="100%" HEIGHT="{LineWidth}"></TD>'
            f"</TR>"
            f"<TR>"
            f"  <TD><B> {node.unique_name} </B></TD>"
            f"</TR>"
            f"<TR>"
            f'  <TD CELLPADDING="0">'
            f'    <TABLE BORDER="0" CELLBORDER="2" CELLSPACING="0" WIDTH="100%">'
            f"      <TR>"
            f'        <TD BGCOLOR="{obs_color}" WIDTH="50%" FIXEDSIZE="FALSE"><FONT FACE="monospace">{obs_text}</FONT></TD>'
            f"        <VR/>"
            f'        <TD BGCOLOR="{life_color}" WIDTH="50%" FIXEDSIZE="FALSE"><FONT FACE="monospace">{life_symbol}</FONT></TD>'
            f"      </TR>"
            f"    </TABLE>"
            f"  </TD>"
            f"</TR>"
        )
        if self.compact:
            label += (
                f"<TR>" f'  <TD WIDTH="100%" HEIGHT="{LineWidth*2.5}"></TD>' f"</TR>"
            )
        else:
            label += self._build_condition_block(node)
        label += f"</TABLE>>"
        return label

    def _build_condition_block(
        self, node: MotionStatechartNode, line_color="black"
    ) -> str:
        start_condition = format_condition(str(node._start_condition))
        pause_condition = format_condition(str(node._pause_condition))
        end_condition = format_condition(str(node._end_condition))
        reset_condition = format_condition(str(node._reset_condition))
        label = (
            f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
            f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">start:{start_condition}</FONT></TD></TR>'
        )
        if not isinstance(node, (EndMotion, CancelMotion)):
            label += (
                f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
                f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">pause:{pause_condition}</FONT></TD></TR>'
            )
            label += (
                f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
                f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">end  :{end_condition}</FONT></TD></TR>'
            )
            label += (
                f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
                f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">reset:{reset_condition}</FONT></TD></TR>'
            )
        return label

    def _escape_name(self, name: str) -> str:
        return f'"{name}"'

    def _get_cluster_of_node(
        self, node_name: str, graph: Union[pydot.Graph, pydot.Cluster]
    ) -> Optional[pydot.Cluster]:
        node_cluster = None
        for cluster in graph.get_subgraphs():
            if (
                len(cluster.get_node(self._escape_name(node_name))) == 1
                or len(cluster.get_node(node_name)) == 1
            ):
                node_cluster = cluster
                break
        return node_cluster

    def _add_node(
        self,
        graph: pydot.Graph,
        node: MotionStatechartNode,
    ) -> pydot.Node:
        pydot_node = self._create_pydot_node(node)
        if len(node.plot_specs.extra_border_styles) == 0:
            graph.add_node(pydot_node)
            return pydot_node
        child = pydot_node
        for index, style in enumerate(node.plot_specs.extra_border_styles):
            c = pydot.Cluster(
                graph_name=f"{node.unique_name}",
                penwidth=LineWidth,
                style=node.plot_specs.extra_border_styles[index],
                color="black",
            )
            if index == 0:
                c.add_node(child)
            else:
                c.add_subgraph(child)
            child = c
        if len(node.plot_specs.extra_border_styles) > 0:
            graph.add_subgraph(c)
        return pydot_node

    def _create_pydot_node(self, node: MotionStatechartNode) -> pydot.Node:
        label = self._format_motion_graph_node(node=node)
        pydot_node = pydot.Node(
            str(node.unique_name),
            label=label,
            shape=node.plot_specs.shape,
            color="black",
            style=node.plot_specs.style,
            margin=0,
            fillcolor="white",
            fontname=FONT,
            fontsize=Fontsize,
            penwidth=LineWidth,
        )
        return pydot_node

    def to_dot_graph(self) -> pydot.Graph:
        self._cluster_map[None] = self.graph
        top_level_nodes = [
            node for node in self.motion_statechart.nodes if not node.parent_node
        ]
        self._add_nodes(self.graph, top_level_nodes)
        self._add_edges()
        return self.graph

    def to_dot_graph_pdf(self, file_name: str):
        self.to_dot_graph()
        file_name = file_name
        # create_path(file_name)
        self.graph.write_pdf(file_name)
        print(f"Saved task graph at {file_name}.")

    def _is_visible_in_hierarchy(self, node: MotionStatechartNode) -> bool:
        """Return False if the node or any of its ancestors is marked invisible in plot specs."""
        current = node
        while current is not None:
            if not current.plot_specs.visible:
                return False
            current = current.parent_node
        return True

    def _add_nodes(
        self,
        parent_cluster: Union[pydot.Graph, pydot.Cluster],
        nodes: List[MotionStatechartNode],
    ):
        for i, node in enumerate(nodes):
            # Skip invisible nodes entirely. If a Goal is invisible, skip its children as well.
            if not self._is_visible_in_hierarchy(node):
                continue

            if isinstance(node, Goal):
                goal_cluster = self._add_cluster(node, parent_cluster)
                self._add_node(
                    graph=goal_cluster,
                    node=node,
                )
                # Recurse only into visible children; _add_nodes applies visibility filtering
                self._add_nodes(goal_cluster, node.nodes)

            self._add_node(
                parent_cluster,
                node=node,
            )

    def _add_cluster(
        self,
        node: MotionStatechartNode,
        parent_cluster: Union[pydot.Graph, pydot.Cluster],
    ):
        goal_cluster = pydot.Cluster(
            graph_name=str(node.unique_name),
            fontname=FONT,
            fontsize=Fontsize,
            style=GoalClusterStyle,
            color="black",
            fillcolor="white",
            penwidth=LineWidth,
        )
        parent_cluster.add_subgraph(goal_cluster)
        self._cluster_map[node] = goal_cluster
        return goal_cluster

    def _add_edges(self):
        transition: TrinaryCondition
        for edge_index, (
            parent_node_index,
            child_node_index,
            transition,
        ) in self.motion_statechart.rx_graph.edge_index_map().items():
            parent_node = self.motion_statechart.rx_graph.get_node_data(
                parent_node_index
            )
            child_node = self.motion_statechart.rx_graph.get_node_data(child_node_index)

            # Skip edges if either endpoint (or one of its ancestors) is invisible
            if not self._is_visible_in_hierarchy(parent_node):
                continue
            if not self._is_visible_in_hierarchy(child_node):
                continue

            if not self._are_nodes_in_same_cluster(parent_node, child_node):
                continue
            spec = TRANSITION_SPECS.get(transition.kind)
            if spec is None:
                raise ValueError(f"Unhandled transition kind: {transition.kind}")
            self._add_condition_edge(parent_node, child_node, spec)

    def _are_nodes_in_same_cluster(
        self, parent_node: MotionStatechartNode, child_node: MotionStatechartNode
    ) -> bool:
        parent_node_parent = parent_node.parent_node
        child_node_parent = child_node.parent_node

        if parent_node_parent is None or child_node_parent is None:
            return parent_node_parent is child_node_parent

        return parent_node_parent.name == child_node_parent.name

    def _edge_clusters_kwargs(
        self,
        graph: Union[pydot.Graph, pydot.Cluster],
        src_name: str,
        dst_name: str,
    ) -> Dict[str, object]:
        kwargs: Dict[str, object] = {}
        dst_cluster = self._get_cluster_of_node(dst_name, graph)
        src_cluster = self._get_cluster_of_node(src_name, graph)
        if dst_cluster is not None:
            kwargs["lhead"] = dst_cluster.get_name()
        if src_cluster is not None:
            kwargs["ltail"] = src_cluster.get_name()
        return kwargs

    def _add_condition_edge(
        self,
        parent_node: MotionStatechartNode,
        child_node: MotionStatechartNode,
        spec: EdgeSpec,
    ) -> None:
        graph = self._cluster_map[parent_node.parent_node]

        def _select_node(
            parent: MotionStatechartNode,
            child: MotionStatechartNode,
            selector: StateSelector,
        ) -> MotionStatechartNode:
            return parent if selector == "parent" else child

        src_node = _select_node(parent_node, child_node, spec.src_selector)
        dst_node = _select_node(parent_node, child_node, spec.dst_selector)
        style_node = _select_node(parent_node, child_node, spec.state_selector)

        src_name = str(src_node.unique_name)
        dst_name = str(dst_node.unique_name)

        kwargs = self._edge_clusters_kwargs(graph, src_name, dst_name)

        observation_state = self.motion_statechart.observation_state[style_node]
        kwargs.update(ObservationStateToEdgeStyle[observation_state])

        kwargs.update(spec.extras())

        graph.add_edge(
            pydot.Edge(
                src=src_name,
                dst=dst_name,
                color=spec.color,
                arrowsize=ArrowSize,
                **kwargs,
            )
        )
