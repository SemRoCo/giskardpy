import unittest

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager


class TestSymbolManager(unittest.TestCase):
    def test_register_symbol(self):
        symbol1 = symbol_manager.register_symbol('test', lambda: 1)
        symbol2 = symbol_manager.register_symbol('test', lambda: 1)
        assert id(symbol1) == id(symbol2)
        symbol2 = symbol_manager.get_symbol('test')
        assert id(symbol1) == id(symbol2)

    def test_to_expr_point3(self):
        god_map.array = np.array([1, 2, 3, 0])

        expr = symbol_manager.get_expr('god_map.array', output_type_hint=cas.Point3)
        assert isinstance(expr, cas.Point3)



    def test_expr_vector3(self):
        god_map.array = np.array([1, 2, 3, 0])

        expr = symbol_manager.get_expr('god_map.array', output_type_hint=cas.Vector3)
        assert isinstance(expr, cas.Vector3)



    def test_to_expr_quaternion(self):
        god_map.array = np.array([1, 2, 3, 0])
        god_map.m = np.eye(3)

        expr = symbol_manager.get_expr('god_map.m', output_type_hint=cas.RotationMatrix)
        assert isinstance(expr, cas.RotationMatrix)
        expr = symbol_manager.get_expr('god_map.array', output_type_hint=cas.Quaternion)
        assert isinstance(expr, cas.Quaternion)


    def test_to_expr_pose(self):
        god_map.m = np.eye(4)

        expr = symbol_manager.get_expr('god_map.m', output_type_hint=cas.TransMatrix)
        assert isinstance(expr, cas.TransMatrix)

    def test_to_expr_exceptions(self):
        god_map.array = [1, 2, 3]
        try:
            symbol_manager.get_expr('god_map.muh', output_type_hint=cas.Point3)
            assert False
        except AttributeError as e:
            pass

        try:
            symbol_manager.get_expr('god_map.array')
            assert False
        except ValueError as e:
            pass

        try:
            symbol_manager.get_expr('god_map.muh', input_type_hint=str, output_type_hint=cas.Point3)
            assert False
        except NotImplementedError as e:
            pass
