import copy
import numbers
from collections import defaultdict
from copy import copy, deepcopy
from multiprocessing import RLock
from typing import Sequence, Union, Any, List

import numpy as np
from geometry_msgs.msg import Pose, Point, Vector3, PoseStamped, PointStamped, Vector3Stamped, QuaternionStamped, \
    Quaternion

from giskardpy import casadi_wrapper as w
from giskardpy.data_types import KeyDefaultDict
from giskardpy.utils import logging
from giskardpy.utils.singleton import SingletonMeta


class _GodMap(metaclass=SingletonMeta):
    """
    Data structure used by tree to exchange information.
    """

    key_to_expr: dict
    expr_to_key: dict
    shortcuts: dict

    def __init__(self):
        self.key_to_expr = {}
        self.expr_to_key = {}
        self.shortcuts = {}
        self.expr_separator = '_'
        self.lock = RLock()

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

    def clear_cache(self):
        self.shortcuts = {}

    def to_symbol(self, identifier) -> w.Symbol:
        """
        All registered identifiers will be included in self.get_symbol_map().
        :type identifier: list
        :return: the symbol corresponding to the identifier
        :rtype: sw.Symbol
        """
        assert isinstance(identifier, list) or isinstance(identifier, tuple)
        identifier = tuple(identifier)
        identifier_parts = identifier
        if identifier not in self.key_to_expr:
            expr = w.Symbol(self.expr_separator.join([str(x) for x in identifier]))
            if expr in self.expr_to_key:
                raise Exception(f'{self.expr_separator} not allowed in key')
            self.key_to_expr[identifier] = expr
            self.expr_to_key[str(expr)] = identifier_parts
        return self.key_to_expr[identifier]


    def list_to_symbol_matrix(self, identifier, data):
        def replace_nested_list(l, f, start_index=None):
            if start_index is None:
                start_index = []
            result = []
            for i, entry in enumerate(l):
                index = start_index + [i]
                if isinstance(entry, list):
                    result.append(replace_nested_list(entry, f, index))
                else:
                    result.append(f(index))
            return result

        return w.Expression(replace_nested_list(data, lambda index: self.to_symbol(identifier + index)))

    def list_to_point3(self, identifier) -> w.Point3:
        return w.Point3((self.to_symbol(identifier + [0]),
                         self.to_symbol(identifier + [1]),
                         self.to_symbol(identifier + [2])))

    def list_to_vector3(self, identifier) -> w.Vector3:
        return w.Vector3((self.to_symbol(identifier + [0]),
                          self.to_symbol(identifier + [1]),
                          self.to_symbol(identifier + [2])))

    def list_to_translation3(self, identifier) -> w.TransMatrix:
        return w.TransMatrix.from_xyz_rpy(
            x=self.to_symbol(identifier + [0]),
            y=self.to_symbol(identifier + [1]),
            z=self.to_symbol(identifier + [2]),
        )

    def list_to_frame(self, identifier):
        return w.TransMatrix(
            [
                [
                    self.to_symbol(identifier + [0, 0]),
                    self.to_symbol(identifier + [0, 1]),
                    self.to_symbol(identifier + [0, 2]),
                    self.to_symbol(identifier + [0, 3])
                ],
                [
                    self.to_symbol(identifier + [1, 0]),
                    self.to_symbol(identifier + [1, 1]),
                    self.to_symbol(identifier + [1, 2]),
                    self.to_symbol(identifier + [1, 3])
                ],
                [
                    self.to_symbol(identifier + [2, 0]),
                    self.to_symbol(identifier + [2, 1]),
                    self.to_symbol(identifier + [2, 2]),
                    self.to_symbol(identifier + [2, 3])
                ],
                [
                    0, 0, 0, 1
                ],
            ]
        )

    def pose_msg_to_frame(self, identifier):
        p = w.Point3.from_xyz(x=self.to_symbol(identifier + ['position', 'x']),
                              y=self.to_symbol(identifier + ['position', 'y']),
                              z=self.to_symbol(identifier + ['position', 'z']))
        q = w.Quaternion.from_xyzw(x=self.to_symbol(identifier + ['orientation', 'x']),
                                   y=self.to_symbol(identifier + ['orientation', 'y']),
                                   z=self.to_symbol(identifier + ['orientation', 'z']),
                                   w=self.to_symbol(identifier + ['orientation', 'w'])).to_rotation_matrix()
        return w.TransMatrix.from_point_rotation_matrix(p, q)

    def quaternion_msg_to_rotation(self, identifier):
        return w.Quaternion.from_xyzw(x=self.to_symbol(identifier + ['x']),
                                      y=self.to_symbol(identifier + ['y']),
                                      z=self.to_symbol(identifier + ['z']),
                                      w=self.to_symbol(identifier + ['w'])).to_rotation_matrix()

    def point_msg_to_point3(self, identifier):
        return w.Point3.from_xyz(
            x=self.to_symbol(identifier + ['x']),
            y=self.to_symbol(identifier + ['y']),
            z=self.to_symbol(identifier + ['z']),
        )

    def vector_msg_to_vector3(self, identifier):
        return w.Vector3.from_xyz(
            x=self.to_symbol(identifier + ['x']),
            y=self.to_symbol(identifier + ['y']),
            z=self.to_symbol(identifier + ['z']),
        )
