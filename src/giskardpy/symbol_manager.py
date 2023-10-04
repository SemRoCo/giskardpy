from typing import Dict, Callable

from giskardpy.god_map_interpreter import god_map
import numpy as np
import giskardpy.casadi_wrapper as cas
from giskardpy.utils.singleton import SingletonMeta


class SymbolManager(metaclass=SingletonMeta):
    symbol_str_to_lambda: Dict[str, Callable[[], float]]
    symbol_str_to_symbol: Dict[str, cas.Symbol]

    def __init__(self):
        self.symbol_str_to_lambda = {}
        self.symbol_str_to_symbol = {}
        self.last_hash = -1

    def get_symbol(self, symbol_reference):
        # TODO check for start with god_map?
        if symbol_reference not in self.symbol_str_to_lambda:
            lambda_expr = eval(f'lambda: {symbol_reference}')
            self.symbol_str_to_lambda[symbol_reference] = lambda_expr
            self.symbol_str_to_symbol[symbol_reference] = cas.Symbol(symbol_reference)
        return self.symbol_str_to_symbol[symbol_reference]

    def resolve_symbols(self, symbols):
        return np.array([self.symbol_str_to_lambda[s]() for s in symbols])

    def compile_resolve_symbols(self, symbols):
        self.c = eval('lambda: np.array([' + ', '.join(symbols) + '])')

    def resolve_symbols2(self):
        return self.c()

    def resolve_symbols3(self, symbols):
        h = hash(tuple(symbols))
        if h != self.last_hash:
            self.compile_resolve_symbols(symbols)
            self.last_hash = h
        return self.resolve_symbols2()

    def evaluate_expr(self, expr: cas.Expression):
        if isinstance(expr, (int, float)):
            return expr
        f = expr.compile()
        if len(f.str_params) == 0:
            return expr.evaluate()
        result = f.fast_call(self.resolve_symbols(f.str_params))
        if len(result) == 1:
            return result[0]
        else:
            return result

    # def to_expr(self, identifier):
    #     try:
    #         data = self.get_data(identifier)
    #     except KeyError as e:
    #         raise KeyError(f'to_expr only works, when there is already data at the path: {e}')
    #     if isinstance(data, np.ndarray):
    #         data = data.tolist()
    #     if isinstance(data, numbers.Number):
    #         return self.to_symbol(identifier)
    #     if isinstance(data, Pose):
    #         return self.pose_msg_to_frame(identifier)
    #     elif isinstance(data, PoseStamped):
    #         return self.pose_msg_to_frame(identifier + ['pose'])
    #     elif isinstance(data, Point):
    #         return self.point_msg_to_point3(identifier)
    #     elif isinstance(data, PointStamped):
    #         return self.point_msg_to_point3(identifier + ['point'])
    #     elif isinstance(data, Vector3):
    #         return self.vector_msg_to_vector3(identifier)
    #     elif isinstance(data, Vector3Stamped):
    #         return self.vector_msg_to_vector3(identifier + ['vector'])
    #     elif isinstance(data, list):
    #         return self.list_to_symbol_matrix(identifier, data)
    #     elif isinstance(data, Quaternion):
    #         return self.quaternion_msg_to_rotation(identifier)
    #     elif isinstance(data, QuaternionStamped):
    #         return self.quaternion_msg_to_rotation(identifier + ['quaternion'])
    #     elif isinstance(data, np.ndarray):
    #         return self.list_to_symbol_matrix(identifier, data)
    #     else:
    #         raise NotImplementedError('to_expr not implemented for type {}.'.format(type(data)))

    @property
    def time(self):
        return self.get_symbol('god_map.time')

    @property
    def hack(self):
        return self.get_symbol('god_map.hack')


symbol_manager = SymbolManager()
