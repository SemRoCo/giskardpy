import numpy as np

import giskardpy
from giskardpy.utils import KeyDefaultDict

giskardpy.WORLD_IMPLEMENTATION = None
import unittest
from collections import namedtuple

from geometry_msgs.msg import PoseStamped
from hypothesis import given
import hypothesis.strategies as st
from giskardpy import identifier
from giskardpy import cas_wrapper as w
from giskardpy.god_map import GodMap
from utils_for_tests import variable_name, keys_values, lists_of_same_length, pr2_urdf
from giskardpy.world import World
from giskardpy.world_object import WorldObject

PKG = u'giskardpy'


class TestGodMap(unittest.TestCase):
    @given(variable_name(),
           st.integers())
    def test_set_get_integer(self, key, number):
        db = GodMap()
        db.safe_set_data([key], number)
        self.assertEqual(db.get_data([key]), number, msg=u'key={}, number={}'.format(key, number))

    @given(variable_name(),
           st.floats(allow_nan=False))
    def test_set_get_float(self, key, number):
        db = GodMap()
        db.safe_set_data([key], number)
        db.safe_set_data([key], number)

        self.assertEqual(db.get_data([key]), number)

    @given(variable_name(),
           variable_name(),
           keys_values())
    def test_namedtuple(self, key, tuple_name, key_values):
        Frame = namedtuple(tuple_name, key_values[0])
        db = GodMap()
        db.safe_set_data([key], Frame(*key_values[1]))
        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), v)

    def test_namedtuple1(self):
        Frame = namedtuple(u'Frame', [u'pos'])
        db = GodMap()
        db.safe_set_data([u'f12'], Frame(pos=2))
        with self.assertRaises(AttributeError):
            db.safe_set_data([u'f12', u'pos'], 42)

    @given(variable_name(),
           variable_name(),
           keys_values())
    def test_class1(self, key, class_name, key_values):
        c = type(str(class_name), (object,), {})()
        for k, v in zip(*key_values):
            setattr(c, k, None)

        db = GodMap()
        db.safe_set_data([key], c)

        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), None)
        for k, v in zip(*key_values):
            db.safe_set_data([key, k], v)
        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), v)

    @given(st.lists(variable_name(), unique=True))
    def test_class3(self, class_names):
        db = GodMap()
        for i, name in enumerate(class_names):
            c = type(str(name), (object,), {})()
            db.safe_set_data(class_names[:i + 1], c)

        for i, name in enumerate(class_names):
            c = db.get_data(class_names[:i + 1])
            self.assertEqual(type(c).__name__, name)

    @given(variable_name(),
           keys_values())
    def test_dict1(self, key, key_values):
        d = {k: v for k, v in zip(*key_values)}
        db = GodMap()
        db.safe_set_data([key], d)
        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), v)

    @given(variable_name(),
           keys_values())
    def test_dict2(self, key, key_values):
        d = {}
        db = GodMap()
        db.safe_set_data([key], d)
        for k, v in zip(*key_values):
            db.safe_set_data([key, k], v)
        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), v)

    @given(variable_name(),
           st.lists(variable_name(), min_size=1),
           st.floats(allow_nan=False))
    def test_dict3(self, key, tuple_key, value):
        tuple_key = tuple(tuple_key)
        d = {tuple_key: value}
        db = GodMap()
        db.safe_set_data([key], d)
        self.assertEqual(db.get_data([key, tuple_key]), value)

    @given(variable_name(),
           st.lists(st.floats(allow_nan=False), min_size=1))
    def test_list1(self, key, value):
        db = GodMap()
        db.safe_set_data([key], value)
        for i, v in enumerate(value):
            self.assertEqual(db.get_data([key, i]), v)

    def test_list_double_index(self):
        key = 'asdf'
        value = np.array([[0, 1], [2, 3]])
        db = GodMap()
        db.safe_set_data([key], value)
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                self.assertEqual(db.get_data([key, i, j]), value[i, j])

    @given(variable_name(),
           st.lists(st.floats(allow_nan=False), min_size=1))
    def test_tuple1(self, key, value):
        value = tuple(value)
        db = GodMap()
        db.safe_set_data([key], value)
        for i, v in enumerate(value):
            self.assertEqual(db.get_data([key, i]), v)

    @given(variable_name(),
           lists_of_same_length([st.floats(allow_nan=False), st.floats(allow_nan=False)]))
    def test_list_overwrite_entry(self, key, lists):
        first_values, second_values = lists
        db = GodMap()
        db.safe_set_data([key], first_values)
        for i, v in enumerate(first_values):
            self.assertEqual(db.get_data([key, i]), v)
            db.safe_set_data([key, i], second_values[i])

        for i, v in enumerate(second_values):
            self.assertEqual(db.get_data([key, i]), v)

    @given(variable_name(),
           st.lists(st.floats(allow_nan=False), min_size=1))
    def test_list_index_error(self, key, l):
        db = GodMap()
        db.safe_set_data([key], l)
        with self.assertRaises(IndexError):
            db.safe_set_data([key, len(l) + 1], 0)

    @given(variable_name(),
           st.lists(st.floats(allow_nan=False), min_size=1))
    def test_list_negative_index(self, key, l):
        db = GodMap()
        db.safe_set_data([key], l)
        for i in range(len(l)):
            self.assertEqual(db.get_data([key, -i]), l[-i])

    @given(variable_name(),
           variable_name())
    def test_function_1param_lambda(self, key, key2):
        db = GodMap()
        f = lambda x: x
        db.safe_set_data([key], f)
        self.assertEqual(db.get_data([key, [key2]]), key2)

    @given(variable_name(),
           variable_name(),
           variable_name(),
           variable_name())
    def test_function_2param_call(self, key1, key2, key3, key4):
        db = GodMap()

        class MUH(object):
            def __call__(self, next_member, next_next_member):
                return next_next_member

        a = MUH()
        d = {key2: a}
        db.safe_set_data([key1], d)
        self.assertEqual(db.get_data([key1, key2, (key3, key4)]), key4)

    @given(variable_name(),
           variable_name(),
           variable_name(),
           variable_name(),
           variable_name())
    def test_function3(self, key1, key2, key3, key4, key5):
        db = GodMap()

        class MUH(object):
            def __call__(self, next_member):
                return [key5]

        a = MUH()
        d = {key2: a}
        db.safe_set_data([key1], d)
        try:
            db.get_data([key1, key2, (key3, key4), 0])
            assert False
        except TypeError:
            assert True

    @given(variable_name(),
           variable_name(),
           variable_name(),
           variable_name(),
           variable_name())
    def test_function4(self, key1, key2, key3, key4, key5):
        db = GodMap()

        class MUH(object):
            def __call__(self, next_member, next_next_member):
                return [key5]

        a = MUH()
        d = {key2: a}
        db.safe_set_data([key1], d)
        self.assertEqual(key5, db.get_data([key1, key2, (key3, key4), 0]))

    @given(variable_name(),
           variable_name(),
           variable_name())
    def test_function_no_param(self, key1, key2, key3):
        db = GodMap()

        class MUH(object):
            def __call__(self):
                return [key3]

        a = MUH()
        d = {key2: a}
        db.safe_set_data([key1], d)
        self.assertEqual(key3, db.get_data([key1, key2, [], 0]))

    @given(variable_name(),
           st.integers())
    def test_to_symbol(self, key, value):
        gm = GodMap()
        gm.safe_set_data([key], value)
        self.assertTrue(w.is_symbol(gm.to_symbol([key])))
        self.assertTrue(key in str(gm.to_symbol([key])))

    @given(lists_of_same_length([variable_name(), st.floats()], unique=True))
    def test_get_symbol_map(self, keys_values):
        keys, values = keys_values
        gm = GodMap()
        for key, value in zip(keys, values):
            gm.safe_set_data([key], value)
            gm.to_symbol([key])
        self.assertEqual(len(gm.get_values(keys)), len(keys))

    def test_god_map_with_world(self):
        gm = GodMap()
        w = World()
        r = WorldObject(pr2_urdf())
        w.add_robot(robot=r,
                    base_pose=PoseStamped(),
                    controlled_joints=[],
                    ignored_pairs=set(),
                    added_pairs=set())
        gm.safe_set_data([u'world'], w)
        gm_robot = gm.get_data(identifier.robot)
        assert 'pr2' == gm_robot.get_name()


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestDataBus',
                    test=TestGodMap)
