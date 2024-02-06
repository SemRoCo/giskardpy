from typing import Dict, Callable, Type, Optional, overload, Union

from giskardpy.exceptions import GiskardException
from giskardpy.god_map import god_map
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point, Vector3Stamped, Vector3, QuaternionStamped, \
    Quaternion

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
        try:
            return np.array([self.symbol_str_to_lambda[s]() for s in symbols], dtype=float)
        except Exception as e:
            for s in symbols:
                try:
                    self.symbol_str_to_lambda[s]()
                except Exception as e2:
                    raise GiskardException(f'Cannot resolve {s} ({e2.__class__.__name__}: {str(e2)})')

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

    @overload
    def get_expr(self,
                 expr: str,
                 input_type_hint: Union[Type[Point], Type[PointStamped]] = None,
                 output_type_hint: type(None) = None) -> cas.Point3:
        ...

    @overload
    def get_expr(self,
                 expr: str,
                 input_type_hint: Optional[Union[list, tuple, np.ndarray]] = None,
                 output_type_hint: Type[cas.Point3] = None) -> cas.Point3:
        ...

    @overload
    def get_expr(self,
                 expr: str,
                 input_type_hint: Union[Type[Vector3], Type[Vector3Stamped]] = None,
                 output_type_hint: type(None) = None) -> cas.Vector3:
        ...

    @overload
    def get_expr(self,
                 expr: str,
                 input_type_hint: Optional[Union[list, tuple, np.ndarray]] = None,
                 output_type_hint: Type[cas.Vector3] = None) -> cas.Vector3:
        ...

    @overload
    def get_expr(self,
                 expr: str,
                 input_type_hint: Union[Type[Quaternion], Type[QuaternionStamped]] = None,
                 output_type_hint: Optional[Type[cas.RotationMatrix]] = None) -> cas.RotationMatrix:
        ...

    @overload
    def get_expr(self,
                 expr: str,
                 input_type_hint: Union[Type[Quaternion], Type[QuaternionStamped]] = None,
                 output_type_hint: Type[cas.Quaternion] = None) -> cas.Quaternion:
        ...

    @overload
    def get_expr(self,
                 expr: str,
                 input_type_hint: Optional[Union[list, tuple, np.ndarray]] = None,
                 output_type_hint: Type[cas.Quaternion] = None) -> cas.Quaternion:
        ...

    @overload
    def get_expr(self,
                 expr: str,
                 input_type_hint: Optional[Union[list, tuple, np.ndarray]] = None,
                 output_type_hint: Type[cas.RotationMatrix] = None) -> cas.RotationMatrix:
        ...

    @overload
    def get_expr(self,
                 expr: str,
                 input_type_hint: Optional[Union[list, tuple, np.ndarray]] = None,
                 output_type_hint: Type[cas.TransMatrix] = None) -> cas.TransMatrix:
        ...

    def get_expr(self, variable_ref_str, input_type_hint=None, output_type_hint=None):
        """

        :param variable_ref_str: A string expression referring to a variable (on the god_map), e.g. 'god_map.time'
        :param input_type_hint: Use this to specify the type of the input. If None, this function will try to evaluate
                                expr and use its type.
        :param output_type_hint: when input_type_hint is a list, typle or numpy array, set the output type.
                                    will be ignored if input_type_hint is a ROS message.
        :return:
        """
        if input_type_hint is None:
            try:
                data = eval(variable_ref_str)
                input_type_hint = type(data)
            except Exception as e:
                raise type(e)(f'No data in \'{variable_ref_str}\' ({e}), '
                              f'can\'t determine input_type_hint, please set it.')

        if input_type_hint in {list, tuple, np.ndarray}:
            if output_type_hint == cas.TransMatrix:
                return self._list_to_transmatrix(variable_ref_str)
            if output_type_hint == cas.Point3:
                return self._list_to_point3(variable_ref_str)
            if output_type_hint == cas.Vector3:
                return self._list_to_vector3(variable_ref_str)
            if output_type_hint == cas.Quaternion:
                return self._list_to_quaternion(variable_ref_str)
            if output_type_hint == cas.RotationMatrix:
                return self._list_to_rotation_matrix(variable_ref_str)
            raise ValueError(f'If input_type_hint is [list, tuple, np.ndarray], please specify output_type_hint out of'
                             f'[cas.TransMatrix, cas.Point3, cas.Vector3, cas.Quaternion, cas.RotationMatrix]')

        if input_type_hint == PointStamped:
            return self._point_msg_to_point3(f'{variable_ref_str}.point')
        if input_type_hint == Point:
            return self._point_msg_to_point3(variable_ref_str)
        if input_type_hint == Vector3Stamped:
            return self._vector_msg_to_vector3(f'{variable_ref_str}.vector')
        if input_type_hint == Vector3:
            return self._vector_msg_to_vector3(variable_ref_str)
        if input_type_hint == QuaternionStamped:
            variable_ref_str += 'quaternion'
            input_type_hint = Quaternion
        if input_type_hint == Quaternion:
            if output_type_hint == cas.Quaternion:
                return self._quaternion_msg_to_quaternion(variable_ref_str)
            else:
                return self._quaternion_msg_to_rotation(variable_ref_str)
        if input_type_hint == PoseStamped:
            return self._pose_to_transmatrix(f'{variable_ref_str}.pose')
        if input_type_hint == Pose:
            return self._pose_to_transmatrix(variable_ref_str)

        raise NotImplementedError(f'to_expr not implemented for type {input_type_hint}.')

    def _quaternion_msg_to_rotation(self, variable_ref_str: str) -> cas.RotationMatrix:
        return self._quaternion_msg_to_quaternion(variable_ref_str).to_rotation_matrix()

    def _quaternion_msg_to_quaternion(self, variable_ref_str: str) -> cas.Quaternion:
        return cas.Quaternion((self.get_symbol(variable_ref_str + '.x'),
                               self.get_symbol(variable_ref_str + '.y'),
                               self.get_symbol(variable_ref_str + '.z'),
                               self.get_symbol(variable_ref_str + '.w')))

    def _point_msg_to_point3(self, variable_ref_str: str) -> cas.Point3:
        return cas.Point3((self.get_symbol(variable_ref_str + '.x'),
                           self.get_symbol(variable_ref_str + '.y'),
                           self.get_symbol(variable_ref_str + '.z')))

    def _vector_msg_to_vector3(self, variable_ref_str: str) -> cas.Vector3:
        return cas.Vector3((self.get_symbol(variable_ref_str + '.x'),
                            self.get_symbol(variable_ref_str + '.y'),
                            self.get_symbol(variable_ref_str + '.z')))

    def _pose_to_transmatrix(self, variable_ref_str: str) -> cas.TransMatrix:
        p = self._point_msg_to_point3(f'{variable_ref_str}.position')
        r = self._quaternion_msg_to_rotation(f'{variable_ref_str}.orientation')
        return cas.TransMatrix.from_point_rotation_matrix(p, r)

    def _list_to_transmatrix(self, variable_ref_str: str) -> cas.TransMatrix:
        return cas.TransMatrix(
            [
                [
                    self.get_symbol(variable_ref_str + '[0, 0]'),
                    self.get_symbol(variable_ref_str + '[0, 1]'),
                    self.get_symbol(variable_ref_str + '[0, 2]'),
                    self.get_symbol(variable_ref_str + '[0, 3]')
                ],
                [
                    self.get_symbol(variable_ref_str + '[1, 0]'),
                    self.get_symbol(variable_ref_str + '[1, 1]'),
                    self.get_symbol(variable_ref_str + '[1, 2]'),
                    self.get_symbol(variable_ref_str + '[1, 3]')
                ],
                [
                    self.get_symbol(variable_ref_str + '[2, 0]'),
                    self.get_symbol(variable_ref_str + '[2, 1]'),
                    self.get_symbol(variable_ref_str + '[2, 2]'),
                    self.get_symbol(variable_ref_str + '[2, 3]')
                ],
                [
                    0, 0, 0, 1
                ],
            ]
        )

    def _list_to_rotation_matrix(self, variable_ref_str: str) -> cas.RotationMatrix:
        return cas.RotationMatrix(
            [
                [
                    self.get_symbol(variable_ref_str + '[0, 0]'),
                    self.get_symbol(variable_ref_str + '[0, 1]'),
                    self.get_symbol(variable_ref_str + '[0, 2]'),
                    0
                ],
                [
                    self.get_symbol(variable_ref_str + '[1, 0]'),
                    self.get_symbol(variable_ref_str + '[1, 1]'),
                    self.get_symbol(variable_ref_str + '[1, 2]'),
                    0
                ],
                [
                    self.get_symbol(variable_ref_str + '[2, 0]'),
                    self.get_symbol(variable_ref_str + '[2, 1]'),
                    self.get_symbol(variable_ref_str + '[2, 2]'),
                    0
                ],
                [
                    0, 0, 0, 1
                ],
            ]
        )

    def _list_to_point3(self, variable_ref_str: str) -> cas.Point3:
        return cas.Point3((self.get_symbol(variable_ref_str + '[0]'),
                           self.get_symbol(variable_ref_str + '[1]'),
                           self.get_symbol(variable_ref_str + '[2]')))

    def _list_to_vector3(self, variable_ref_str: str) -> cas.Vector3:
        return cas.Vector3((self.get_symbol(variable_ref_str + '[0]'),
                            self.get_symbol(variable_ref_str + '[1]'),
                            self.get_symbol(variable_ref_str + '[2]')))

    def _list_to_quaternion(self, variable_ref_str: str) -> cas.Quaternion:
        return cas.Quaternion((self.get_symbol(variable_ref_str + '[0]'),
                               self.get_symbol(variable_ref_str + '[1]'),
                               self.get_symbol(variable_ref_str + '[2]'),
                               self.get_symbol(variable_ref_str + '[3]')))

    @property
    def time(self):
        return self.get_symbol('god_map.time')

    @property
    def hack(self):
        return self.get_symbol('god_map.hack')


symbol_manager = SymbolManager()
