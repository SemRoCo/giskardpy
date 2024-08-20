from typing import Dict

from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types import PrefixName, TaskState
from giskardpy.god_map import god_map
from giskardpy.motion_graph.graph_node import MotionGraphNode
import giskardpy.casadi_wrapper as cas


@profile
def compile_graph_node_state_updater(graph_nodes: Dict[PrefixName, MotionGraphNode]) -> CompiledFunction:
    symbols = []
    for node in sorted(graph_nodes.values(), key=lambda x: x.id):
        symbols.append(node.get_life_cycle_state_expression())
    node_state = cas.Expression(symbols)
    monitor_state = god_map.monitor_manager.get_monitor_state_expr()

    state_updater = []
    for node in sorted(graph_nodes.values(), key=lambda x: x.id):
        state_symbol = node_state[node.id]

        if cas.is_true(node.start_condition):
            start_if = TaskState.running  # start right away
        else:
            start_if = cas.if_else(node.start_condition,
                                   if_result=TaskState.running,
                                   else_result=TaskState.not_started)
        if cas.is_false(node.hold_condition):
            hold_if = TaskState.running  # never hold
        else:
            hold_if = cas.if_else(node.hold_condition,
                                  if_result=TaskState.on_hold,
                                  else_result=TaskState.running)
        if cas.is_false(node.end_condition):
            else_result = hold_if  # never end
        else:
            else_result = cas.if_else(node.end_condition,
                                      if_result=TaskState.succeeded,
                                      else_result=hold_if)

        state_f = cas.if_eq_cases(a=state_symbol,
                                  b_result_cases=[(TaskState.not_started, start_if),
                                                  (TaskState.succeeded, TaskState.succeeded)],
                                  else_result=else_result)  # running or on_hold
        state_updater.append(state_f)
    state_updater = cas.Expression(state_updater)
    symbols = node_state.free_symbols() + monitor_state.free_symbols()
    return state_updater.compile(symbols)