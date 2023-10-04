import traceback
from typing import List, Dict

import numpy as np
import pandas as pd
import giskardpy.casadi_wrapper as cas
from giskardpy import identifier
from giskardpy.data_types import JointStates
from giskardpy.god_map_interpreter import god_map
from giskardpy.model.trajectory import Trajectory
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils import logging


class DebugExpressionManager:
    debug_expressions: Dict[str, cas.Expression]
    compiled_debug_expressions: Dict[str, cas.CompiledFunction]
    evaluated_debug_expressions: Dict[str, np.ndarray]
    debug_trajectory: Trajectory

    def __init__(self):
        self.debug_expressions = {}
        self.debug_trajectory = Trajectory()

    def add_debug_expression(self, name: str, expression: cas.Expression):
        self.debug_expressions[name] = expression

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
    def eval_debug_exprs(self):
        self.evaluated_debug_expressions = {}
        for name, f in self.compiled_debug_expressions.items():
            params = symbol_manager.resolve_symbols(f.str_params)
            self.evaluated_debug_expressions[name] = f.fast_call(params).copy()
        self.log_debug_expressions()
        return self.evaluated_debug_expressions

    def log_debug_expressions(self):
        if len(self.evaluated_debug_expressions) > 0:
            time = god_map.time - 1
            last_mjs = None
            if time >= 1:
                last_mjs = self.debug_trajectory.get_exact(time-1)
            js = JointStates()
            for name, value in self.evaluated_debug_expressions.items():
                if len(value) > 1:
                    continue
                if last_mjs is not None:
                    velocity = value - last_mjs[name].position
                else:
                    if isinstance(value, np.ndarray):
                        velocity = np.zeros(value.shape)
                    else:
                        velocity = 0
                js[name].position = value
                js[name].velocity = velocity/god_map.qp_controller_config.sample_period
            self.debug_trajectory.set(time, js)

    def to_pandas(self):
        p_debug = {}
        for name, value in self.evaluated_debug_expressions.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) == 2:
                    p_debug[name] = value.reshape((value.shape[0] * value.shape[1]))
                else:
                    p_debug[name] = value
            else:
                p_debug[name] = np.array(value)
        self.p_debug = pd.DataFrame.from_dict(p_debug, orient='index').sort_index()
        return self.p_debug
