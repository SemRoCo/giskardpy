from typing import Dict

from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types.data_types import PrefixName, LifeCycleState
from giskardpy.god_map import god_map
from giskardpy.motion_graph.graph_node import MotionGraphNode
import giskardpy.casadi_wrapper as cas
from line_profiler import profile


@profile
def compile_graph_node_state_updater(graph_nodes: Dict[PrefixName, MotionGraphNode]) -> CompiledFunction:
    symbols = []
    for node in sorted(graph_nodes.values(), key=lambda x: x.id):
        symbols.append(node.get_life_cycle_state_expression())
    node_state = cas.Expression(symbols)
    observation_state_symbols = [x.s for x in god_map.motion_graph_manager.get_observation_state_symbols()]

    state_updater = []
    for node in sorted(graph_nodes.values(), key=lambda x: x.id):
        state_symbol = node_state[node.id]

        if cas.is_true(node.start_condition):
            start_if = LifeCycleState.running  # start right away
        else:
            start_if = cas.if_else(node.start_condition,
                                   if_result=LifeCycleState.running,
                                   else_result=LifeCycleState.not_started)
        if cas.is_false(node.pause_condition):
            pause_if = LifeCycleState.running  # never pause
        else:
            pause_if = cas.if_else(node.pause_condition,
                                   if_result=LifeCycleState.on_hold,
                                   else_result=LifeCycleState.running)
        if cas.is_false(node.end_condition):
            else_result = pause_if  # never end
        else:
            else_result = cas.if_else(node.end_condition,
                                      if_result=LifeCycleState.succeeded,
                                      else_result=pause_if)

        state_f = cas.if_eq_cases(a=state_symbol,
                                  b_result_cases=[(LifeCycleState.not_started, start_if),
                                                  (LifeCycleState.succeeded, LifeCycleState.succeeded)],
                                  else_result=else_result)  # running or paused
        state_updater.append(state_f)
    state_updater = cas.Expression(state_updater)
    symbols = node_state.free_symbols() + observation_state_symbols
    return state_updater.compile(symbols)
