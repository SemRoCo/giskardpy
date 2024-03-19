import traceback
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from std_msgs.msg import ColorRGBA

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types import JointStates
from giskardpy.god_map import god_map
from giskardpy.model.trajectory import Trajectory
from giskardpy.data_types import PrefixName
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils import logging


class DebugExpressionManager:
    debug_expressions: Dict[PrefixName, cas.Expression]
    compiled_debug_expressions: Dict[str, cas.CompiledFunction]
    evaluated_debug_expressions: Dict[str, np.ndarray]
    _debug_trajectory: Trajectory

    def __init__(self):
        self.debug_expressions = {}
        self._debug_trajectory = Trajectory()

    def add_debug_expression(self, name: str, expression: cas.Expression, color: Optional[ColorRGBA] = None):
        if isinstance(expression, cas.Symbol_):
            expression.color = color
        self.debug_expressions[PrefixName(name, prefix='')] = expression

    @property
    def debug_trajectory(self):
        return self._debug_trajectory

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
            logging.loginfo(f'Compiled {len(self.compiled_debug_expressions)} debug expressions.')

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
            control_cycle_counter = god_map.control_cycle_counter - 1
            last_mjs = None
            if control_cycle_counter >= 1:
                last_mjs = self._debug_trajectory.get_exact(control_cycle_counter - 1)
            js = JointStates()
            for name, value in self.evaluated_debug_expressions.items():
                if len(value) > 1:
                    if len(value.shape) == 2:
                        for x in range(value.shape[0]):
                            for y in range(value.shape[1]):
                                tmp_name = f'{name}|{x}_{y}'
                                self.evaluated_expr_to_js(tmp_name, last_mjs, js, value[x, y])
                    else:
                        for x in range(value.shape[0]):
                            tmp_name = f'{name}|{x}'
                            self.evaluated_expr_to_js(tmp_name, last_mjs, js, value[x])
                else:
                    self.evaluated_expr_to_js(name, last_mjs, js, value)
            self._debug_trajectory.set(control_cycle_counter, js)

    def evaluated_expr_to_js(self, name, last_js, next_js: JointStates, value):
        if last_js is not None:
            velocity = value - last_js[name].position
        else:
            if isinstance(value, np.ndarray):
                velocity = np.zeros(value.shape)
            else:
                velocity = 0
        next_js[name].position = value
        next_js[name].velocity = velocity / god_map.qp_controller_config.sample_period

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
