from copy import deepcopy
from typing import Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
from giskardpy.data_types.data_types import Derivatives
from line_profiler import profile

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import JointStates, ColorRGBA
from giskardpy.god_map import god_map
from giskardpy.model.trajectory import Trajectory
from giskardpy.data_types.data_types import PrefixName
from giskardpy.symbol_manager import symbol_manager
from giskardpy.middleware import get_middleware


class DebugExpressionManager:
    debug_expressions: Dict[PrefixName, cas.Expression]
    compiled_debug_expressions: Dict[PrefixName, cas.CompiledFunction]
    evaluated_debug_expressions: Dict[PrefixName, np.ndarray]
    _raw_debug_trajectory: List[Dict[PrefixName, np.ndarray]]

    def __init__(self):
        self.reset()

    def reset(self):
        self.debug_expressions = {}
        self._raw_debug_trajectory = []
        self.compiled_debug_expressions = {}
        self.evaluated_debug_expressions = {}

    def add_debug_expression(self, name: str, expression: cas.Expression, color: Optional[ColorRGBA] = None,
                             derivative: Derivatives = Derivatives.position,
                             derivatives_to_plot: Optional[List[Derivatives]] = None):
        if derivatives_to_plot is None:
            derivatives_to_plot = [derivative]
        if isinstance(expression, cas.Symbol_):
            expression.color = color
        expression.debug_derivative = derivative
        expression.debug_derivatives_to_plot = derivatives_to_plot
        self.debug_expressions[PrefixName(name, prefix='')] = expression

    def compile_debug_expressions(self):
        for name, expr in self.debug_expressions.items():
            if isinstance(expr, (int, float)):
                self.debug_expressions[name] = cas.Expression(expr)
        self.compiled_debug_expressions = {}
        free_symbols = set()
        for name, expr in self.debug_expressions.items():
            free_symbols.update(expr.free_symbols())
        free_symbols = list(free_symbols)
        for name, expr in self.debug_expressions.items():
            self.compiled_debug_expressions[name] = expr.compile(free_symbols)
        num_debug_expressions = len(self.compiled_debug_expressions)
        if num_debug_expressions > 0:
            get_middleware().loginfo(f'Compiled {len(self.compiled_debug_expressions)} debug expressions.')

    @profile
    def eval_debug_expressions(self, log_traj: bool = True):  # renamed
        self.evaluated_debug_expressions = {}
        for name, f in self.compiled_debug_expressions.items():
            params = symbol_manager.resolve_symbols(f.str_params)
            self.evaluated_debug_expressions[name] = f.fast_call(params).copy()
        if log_traj:
            self.log_debug_expressions()
        return self.evaluated_debug_expressions

    def log_debug_expressions(self):
        if len(self.evaluated_debug_expressions) > 0:
            self._raw_debug_trajectory.append(self.evaluated_debug_expressions)

    def raw_traj_to_traj(self, control_dt: float) -> Trajectory:
        debug_trajectory = Trajectory()
        for control_cycle_counter, evaluated_debug_expressions in enumerate(self._raw_debug_trajectory):
            if control_cycle_counter >= 1:
                last_mjs = debug_trajectory.get_exact(control_cycle_counter - 1)
                js = deepcopy(last_mjs)
            else:
                last_mjs = JointStates()
                js = JointStates()
            for name, value in evaluated_debug_expressions.items():
                if len(value) > 1:
                    if len(value.shape) == 2:
                        for x in range(value.shape[0]):
                            for y in range(value.shape[1]):
                                tmp_name = f'{name}|{x}_{y}'
                                self.evaluated_expr_to_js(tmp_name, last_mjs, js, float(value[x, y]), control_dt,
                                                          self.debug_expressions[name].debug_derivative,
                                                          control_cycle_counter)
                    else:
                        for x in range(value.shape[0]):
                            tmp_name = f'{name}|{x}'
                            self.evaluated_expr_to_js(tmp_name, last_mjs, js, float(value[x]), control_dt,
                                                      self.debug_expressions[name].debug_derivative,
                                                      control_cycle_counter)
                else:
                    self.evaluated_expr_to_js(name, last_mjs, js, float(value), control_dt,
                                              self.debug_expressions[name].debug_derivative, control_cycle_counter)
            debug_trajectory.set(control_cycle_counter, js)
        for control_cycle_counter, evaluated_debug_expressions in enumerate(self._raw_debug_trajectory):
            js = debug_trajectory.get_exact(control_cycle_counter)
            for name in self.debug_expressions:
                for d in Derivatives.range(Derivatives.position, Derivatives.jerk):
                    if d not in self.debug_expressions[name].debug_derivatives_to_plot:
                        js[name][d] = 0

        return debug_trajectory

    def evaluated_expr_to_js(self, name: Union[PrefixName, str], last_js: JointStates, next_js: JointStates, value: float,
                             dt:float, derivative: Derivatives, control_cycle_counter: int):
        next_js[name][derivative] = value

        for i in range(derivative + 1, 4 + min(control_cycle_counter-3, 0)):
            next_js[name][i] = (next_js[name][i - 1] - last_js[name][i - 1]) / dt

        for i in range(derivative - 1, -1, -1):
            next_js[name][i] = next_js[name][i] + next_js[name][i + 1] * dt

    def to_pandas(self):
        p_debug = {}
        for name, value in self.evaluated_debug_expressions.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) == 2:
                    p_debug[str(name)] = value.reshape((value.shape[0] * value.shape[1]))
                else:
                    p_debug[str(name)] = value
            else:
                p_debug[str(name)] = np.array(value)
        self.p_debug = pd.DataFrame.from_dict(p_debug, orient='index').sort_index()
        return self.p_debug
