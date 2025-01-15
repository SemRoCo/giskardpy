from typing import Dict, List, TypeVar, Generic, Set, Optional

import numpy as np

from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types.data_types import LifeCycleState, ObservationState
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
import giskardpy.casadi_wrapper as cas
from line_profiler import profile

from giskardpy.symbol_manager import symbol_manager

T = TypeVar('T', bound=MotionStatechartNode)


class MotionGraphNodeStateManager(Generic[T]):
    god_map_path: str

    nodes: List[T]
    key_to_idx: Dict[str, int]
    _life_cycle_state: np.ndarray
    life_cycle_history: List[np.ndarray]
    observation_state: np.ndarray

    substitution_values: Dict[str, Dict[str, float]]  # node name -> (old_symbol, value)

    def __init__(self, god_map_path: str):
        self.nodes = []
        self.key_to_idx = {}
        self.god_map_path = god_map_path
        self.substitution_values = {}

    @property
    def life_cycle_state(self) -> np.ndarray:
        return self._life_cycle_state

    @life_cycle_state.setter
    def life_cycle_state(self, new_life_cycle: np.ndarray) -> None:
        self._life_cycle_state = new_life_cycle
        self.life_cycle_history.append(new_life_cycle.copy())


    def get_node_names(self) -> Set[str]:
        return set(self.key_to_idx.keys())

    def get_life_cycle_state(self, key: str) -> float:
        idx = self.key_to_idx[key]
        return self.life_cycle_state[idx]

    def get_substitution_value(self, key: str, symbol: str) -> float:
        return self.substitution_values[key][symbol]

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
        self.life_cycle_history = []
        self.observation_state = np.ones(len(self.nodes)) * ObservationState.unknown
        life_cycle_state = np.zeros(len(self.nodes))
        for node_id, node in enumerate(self.nodes):
            if cas.is_true_symbol(node.start_condition):
                life_cycle_state[node_id] = LifeCycleState.running
            else:
                life_cycle_state[node_id] = LifeCycleState.not_started
        self.life_cycle_state = life_cycle_state

    def get_state_as_dict(self) -> Dict[str, float]:
        """
        Retrieves the current state as a dictionary mapping keys to values.

        :return: Dictionary of current state.
        """
        return {key: self.life_cycle_state[idx] for key, idx in self.key_to_idx.items()}

    @profile
    def register_expression_updater(self, node: MotionStatechartNode, expression: cas.PreservedCasType) \
            -> cas.PreservedCasType:
        """
        Expression is updated when all monitors are 1 at the same time, but only once.
        """
        old_symbols = []
        new_symbols = []
        for i, symbol in enumerate(expression.free_symbols()):
            old_symbols.append(symbol)
            new_symbols.append(self.get_substitution_key(node.name, str(symbol)))
        new_expression = cas.substitute(expression, old_symbols, new_symbols)
        self.update_substitution_values(node.name, old_symbols)
        return new_expression

    @profile
    def update_substitution_values(self, node_name: str, keys: Optional[List[cas.Symbol]] = None) -> None:
        if keys is None:
            keys = list(self.substitution_values[node_name].keys())
        else:
            keys = [str(s) for s in keys]
        values = symbol_manager.resolve_symbols(keys)
        self.substitution_values[node_name] = {key: value for key, value in zip(keys, values)}

    @profile
    def get_substitution_key(self, node_name: str, original_symbol: str) -> cas.Symbol:
        return symbol_manager.get_symbol(
            f'{self.god_map_path}.substitution_values["{node_name}"]["{original_symbol}"]')

    def trigger_update_triggers(self):
        prev_life_cycle_state = self.life_cycle_history[-2]
        life_cycle_state = self.life_cycle_history[-1]
        condition = (prev_life_cycle_state == LifeCycleState.not_started) & (life_cycle_state == LifeCycleState.running)
        for idx in np.argwhere(condition).flatten():
            node_name = self.nodes[idx].name
            if node_name in self.substitution_values:
                self.update_substitution_values(node_name=node_name)

    def __repr__(self) -> str:
        self_as_dict = self.get_state_as_dict()
        return f"{self_as_dict}"


@profile
def compile_graph_node_state_updater(node_state: MotionGraphNodeStateManager) -> CompiledFunction:
    state_updater = []
    node: MotionStatechartNode
    for node in node_state.nodes:
        state_symbol = node.get_life_cycle_state_expression()

        not_started_transitions = cas.if_else(condition=cas.is_true3(node.logic3_start_condition),
                                              if_result=LifeCycleState.running,
                                              else_result=LifeCycleState.not_started)
        running_transitions = cas.if_cases(
            cases=[(cas.is_true3(node.logic3_reset_condition), LifeCycleState.not_started),
                   (cas.is_true3(node.logic3_end_condition), LifeCycleState.succeeded),
                   (cas.is_true3(node.logic3_pause_condition), LifeCycleState.paused)],
            else_result=LifeCycleState.running)
        pause_transitions = cas.if_cases(cases=[(cas.is_true3(node.logic3_reset_condition), LifeCycleState.not_started),
                                                (cas.is_true3(node.logic3_end_condition), LifeCycleState.succeeded),
                                                (cas.is_false3(node.logic3_pause_condition), LifeCycleState.running)],
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
