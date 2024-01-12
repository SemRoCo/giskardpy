import ast
import numbers
from typing import Dict, Callable

from giskardpy.exceptions import GiskardException
from giskardpy.god_map import god_map
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

    def get_symbol(self, symbol_reference: str) -> cas.Symbol:
        """
        Returns a symbol reference to the input parameter. If the symbol doesn't exist yet, it will be created.
        :param symbol_reference: e.g. 'god_map.monitor_manager.monitors[0]'
        :return: symbol reference
        """
        if symbol_reference not in self.symbol_str_to_lambda:
            lambda_expr = eval(f'lambda: {symbol_reference}')
            self.symbol_str_to_lambda[symbol_reference] = lambda_expr
            self.symbol_str_to_symbol[symbol_reference] = cas.Symbol(symbol_reference)
        return self.symbol_str_to_symbol[symbol_reference]

    def resolve_symbols(self, symbols):
        return np.array([self.symbol_str_to_lambda[s]() for s in symbols], dtype=float)

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

    def get_expr(self, expr, input_type_hint=None, output_type_hint=None):
        if input_type_hint is None:
            try:
                data = eval(expr)
            except KeyError as e:
                raise KeyError(f'to_expr only works, when there is already data at the path: {e}')
        if output_type_hint == cas.TransMatrix:
            return cas.TransMatrix(
                [
                    [
                        self.get_symbol(expr + '[0, 0]'),
                        self.get_symbol(expr + '[0, 1]'),
                        self.get_symbol(expr + '[0, 2]'),
                        self.get_symbol(expr + '[0, 3]')
                    ],
                    [
                        self.get_symbol(expr + '[1, 0]'),
                        self.get_symbol(expr + '[1, 1]'),
                        self.get_symbol(expr + '[1, 2]'),
                        self.get_symbol(expr + '[1, 3]')
                    ],
                    [
                        self.get_symbol(expr + '[2, 0]'),
                        self.get_symbol(expr + '[2, 1]'),
                        self.get_symbol(expr + '[2, 2]'),
                        self.get_symbol(expr + '[2, 3]')
                    ],
                    [
                        0, 0, 0, 1
                    ],
                ]
            )
        if output_type_hint == cas.Point3:
            return cas.Point3((self.get_symbol(expr + '[0]'),
                               self.get_symbol(expr + '[1]'),
                               self.get_symbol(expr + '[2]')))
        if output_type_hint == cas.Vector3:
            return cas.Vector3((self.get_symbol(expr + '[0]'),
                                self.get_symbol(expr + '[1]'),
                                self.get_symbol(expr + '[2]')))

        # if input_type_hint == np.ndarray:
        #     data = data.tolist()
        # if input_type_hint == numbers.Number:
        #     return self.get_symbol(expr)
        # if input_type_hint == Pose:
        #     return self.pose_msg_to_frame(identifier)
        # elif input_type_hint == PoseStamped:
        #     return self.pose_msg_to_frame(identifier + ['pose'])
        # elif input_type_hint == Point:
        #     return self.point_msg_to_point3(identifier)
        # elif input_type_hint == PointStamped:
        #     return self.point_msg_to_point3(identifier + ['point'])
        # elif input_type_hint == Vector3:
        #     return self.vector_msg_to_vector3(identifier)
        # elif input_type_hint == Vector3Stamped:
        #     return self.vector_msg_to_vector3(identifier + ['vector'])
        # elif input_type_hint == list:
        #     return self.list_to_symbol_matrix(identifier, data)
        # elif input_type_hint == Quaternion:
        #     return self.quaternion_msg_to_rotation(identifier)
        # elif input_type_hint == QuaternionStamped:
        #     return self.quaternion_msg_to_rotation(identifier + ['quaternion'])
        # elif input_type_hint == np.ndarray:
        #     return self.list_to_symbol_matrix(identifier, data)
        # else:
        raise NotImplementedError('to_expr not implemented for type {}.'.format(type(data)))

    @property
    def time(self):
        return self.get_symbol('god_map.time')

    @property
    def hack(self):
        return self.get_symbol('god_map.hack')


symbol_manager = SymbolManager()
