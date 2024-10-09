from typing import Dict, List, TypeVar, Generic, Set

import numpy as np

from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types.data_types import LifeCycleState, ObservationState
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_graph.graph_node import MotionGraphNode
import giskardpy.casadi_wrapper as cas
from line_profiler import profile

T = TypeVar('T', bound=MotionGraphNode)


class MotionGraphNodeStateManager(Generic[T]):
    nodes: List[T]
    key_to_idx: Dict[str, int]
    life_cycle_state: np.ndarray
    observation_state: np.ndarray

    def __init__(self):
        self.nodes = []
        self.key_to_idx = {}

    def get_node_names(self) -> Set[str]:
        return set(self.key_to_idx.keys())

    def get_life_cycle_state(self, key: str) -> float:
        idx = self.key_to_idx[key]
        return self.life_cycle_state[idx]

    def get_life_cycle_state_symbols(self) -> List[cas.Symbol]:
        return [node.get_life_cycle_state_expression() for node in self.nodes]

    def get_observation_state_symbol_map(self) -> Dict[str, cas.Symbol]:
        return {node.name: node.get_observation_state_expression() for node in self.nodes}

    def get_observation_state(self, key: str) -> float:
        idx = self.key_to_idx[key]
        return self.observation_state[idx]

    def get_node(self, key: str) -> T:
        idx = self.key_to_idx[key]
        return self.nodes[idx]

    def append(self, node: T) -> None:
        if node.name in self.key_to_idx:
            raise GoalInitalizationException(f'Node named {node.name} already exists.')
        self.nodes.append(node)
        self.key_to_idx[node.name] = len(self.nodes) - 1

    def init_states(self) -> None:
        self.observation_state = np.ones(len(self.nodes)) * ObservationState.unknown
        self.life_cycle_state = np.zeros(len(self.nodes))
        for node_id, node in enumerate(self.nodes):
            if cas.is_true_symbol(node.start_condition):
                self.life_cycle_state[node_id] = LifeCycleState.running
            else:
                self.life_cycle_state[node_id] = LifeCycleState.not_started

    def get_state_as_dict(self) -> Dict[str, float]:
        """
        Retrieves the current state as a dictionary mapping keys to values.

        :return: Dictionary of current state.
        """
        return {key: self.life_cycle_state[idx] for key, idx in self.key_to_idx.items()}

    def __repr__(self) -> str:
        self_as_dict = self.get_state_as_dict()
        return f"{self_as_dict}"


@profile
def compile_graph_node_state_updater(node_state: MotionGraphNodeStateManager) -> CompiledFunction:
    state_updater = []
    node: MotionGraphNode
    for node in node_state.nodes:
        state_symbol = node.get_life_cycle_state_expression()

        not_started_transitions = cas.if_else(condition=cas.is_true3(node.logic3_start_condition),
                                              if_result=LifeCycleState.running,
                                              else_result=LifeCycleState.not_started)
        running_transitions = cas.if_cases(cases=[(cas.is_true3(node.logic3_reset_condition), LifeCycleState.not_started),
                                                  (cas.is_true3(node.logic3_end_condition), LifeCycleState.succeeded),
                                                  (cas.is_true3(node.logic3_pause_condition), LifeCycleState.paused)],
                                           else_result=LifeCycleState.running)
        pause_transitions = cas.if_cases(cases=[(cas.is_true3(node.logic3_reset_condition), LifeCycleState.not_started),
                                                (cas.is_true3(node.logic3_end_condition), LifeCycleState.succeeded),
                                                (cas.logic_not(cas.is_true3(node.logic3_pause_condition)), LifeCycleState.running)],
                                         else_result=LifeCycleState.paused)
        ended_transitions = cas.if_else(condition=cas.is_true3(node.logic3_reset_condition),
                                        if_result=LifeCycleState.not_started,
                                        else_result=LifeCycleState.succeeded)

        state_machine = cas.if_eq_cases(a=state_symbol,
                                        b_result_cases=[(LifeCycleState.not_started, not_started_transitions),
                                                        (LifeCycleState.running, running_transitions),
                                                        (LifeCycleState.paused, pause_transitions),
                                                        (LifeCycleState.succeeded, ended_transitions)],
                                        else_result=state_symbol)
        state_updater.append(state_machine)
    state_updater = cas.Expression(state_updater)

    symbols = node_state.get_life_cycle_state_symbols() + god_map.motion_graph_manager.get_observation_state_symbols()
    return state_updater.compile(symbols)
