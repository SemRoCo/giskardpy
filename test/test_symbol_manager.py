import unittest

import numpy as np
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, QuaternionStamped, Pose

import giskardpy.casadi_wrapper as cas
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager


class TestSymbolManager(unittest.TestCase):
    def test_to_expr_point3(self):
        god_map.array = np.array([1, 2, 3, 0])
        god_map.msg = PointStamped()

        expr = symbol_manager.get_expr('god_map.array', output_type_hint=cas.Point3)
        assert isinstance(expr, cas.Point3)

        expr = symbol_manager.get_expr('god_map.msg')
        assert isinstance(expr, cas.Point3)
        expr = symbol_manager.get_expr('god_map.msg.point')
        assert isinstance(expr, cas.Point3)

        expr = symbol_manager.get_expr('god_map.muh', input_type_hint=PointStamped)
        assert isinstance(expr, cas.Point3)

    def test_expr_vector3(self):
        god_map.array = np.array([1, 2, 3, 0])
        god_map.msg = Vector3Stamped()

        expr = symbol_manager.get_expr('god_map.array', output_type_hint=cas.Vector3)
        assert isinstance(expr, cas.Vector3)

        expr = symbol_manager.get_expr('god_map.msg')
        assert isinstance(expr, cas.Vector3)

        expr = symbol_manager.get_expr('god_map.msg.vector')
        assert isinstance(expr, cas.Vector3)

        expr = symbol_manager.get_expr('god_map.muh', input_type_hint=Vector3Stamped)
        assert isinstance(expr, cas.Vector3)

    def test_to_expr_quaternion(self):
        god_map.array = np.array([1, 2, 3, 0])
        god_map.m = np.eye(3)
        god_map.msg = QuaternionStamped()

        expr = symbol_manager.get_expr('god_map.m', output_type_hint=cas.RotationMatrix)
        assert isinstance(expr, cas.RotationMatrix)
        expr = symbol_manager.get_expr('god_map.array', output_type_hint=cas.Quaternion)
        assert isinstance(expr, cas.Quaternion)

        expr = symbol_manager.get_expr('god_map.msg')
        assert isinstance(expr, cas.RotationMatrix)

        expr = symbol_manager.get_expr('god_map.msg.quaternion')
        assert isinstance(expr, cas.RotationMatrix)
        expr = symbol_manager.get_expr('god_map.msg.quaternion', output_type_hint=cas.Quaternion)
        assert isinstance(expr, cas.Quaternion)
        expr = symbol_manager.get_expr('god_map.msg.quaternion', output_type_hint=cas.RotationMatrix)
        assert isinstance(expr, cas.RotationMatrix)

        expr = symbol_manager.get_expr('god_map.muh', input_type_hint=QuaternionStamped)
        assert isinstance(expr, cas.RotationMatrix)

    def test_to_expr_pose(self):
        god_map.m = np.eye(4)
        god_map.msg = PoseStamped()

        expr = symbol_manager.get_expr('god_map.m', output_type_hint=cas.TransMatrix)
        assert isinstance(expr, cas.TransMatrix)

        expr = symbol_manager.get_expr('god_map.msg')
        assert isinstance(expr, cas.TransMatrix)

        expr = symbol_manager.get_expr('god_map.msg.pose')
        assert isinstance(expr, cas.TransMatrix)

        expr = symbol_manager.get_expr('god_map.muh', input_type_hint=PoseStamped)
        assert isinstance(expr, cas.TransMatrix)
        expr = symbol_manager.get_expr('god_map.muh', input_type_hint=Pose)
        assert isinstance(expr, cas.TransMatrix)

    def test_to_expr(self):
        god_map.point = PointStamped()
        god_map.vector = Vector3Stamped()
        god_map.quaternion = QuaternionStamped()
        god_map.pose = PoseStamped()

        references = {PointStamped: 'god_map.point',
                      Vector3Stamped: 'god_map.vector',
                      QuaternionStamped: 'god_map.quaternion',
                      PoseStamped: 'god_map.pose'}

        god_map.m = np.eye(4)
        god_map.array = np.zeros(4)

        type_map = {PointStamped: cas.Point3,
                    Vector3Stamped: cas.Vector3,
                    QuaternionStamped: cas.RotationMatrix,
                    PoseStamped: cas.TransMatrix}

        for msg, cas_type in type_map.items():
            # %% output type should be ignored if input is ros message
            for msg2, reference in references.items():
                expr = symbol_manager.get_expr(reference, output_type_hint=cas_type)
                if not isinstance(expr, type_map[msg2]):
                    assert False
            # %% test input type
            expr = symbol_manager.get_expr('god_map.muh', input_type_hint=msg)
            if not isinstance(expr, cas_type):
                assert False

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
