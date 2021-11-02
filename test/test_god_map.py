import numpy as np

import giskardpy

giskardpy.WORLD_IMPLEMENTATION = None
import unittest
from collections import namedtuple

from geometry_msgs.msg import PoseStamped
from hypothesis import given, assume
import hypothesis.strategies as st
from giskardpy import casadi_wrapper as w
from giskardpy.god_map import GodMap
from utils_for_tests import variable_name, keys_values, lists_of_same_length

PKG = u'giskardpy'


class TestGodMap(unittest.TestCase):
    @given(variable_name(),
           variable_name(),
           st.integers())
    def test_god_map_key_error(self, key, wrong_key, number):
        assume(key != wrong_key)
        db = GodMap()
        db.set_data([key], number)
        try:
            db.get_data([wrong_key])
            assert False
        except KeyError as e:
            assert True

    @given(variable_name(),
           st.integers())
    def test_set_get_integer(self, key, number):
        db = GodMap()
        db.set_data([key], number)
        self.assertEqual(db.get_data([key]), number, msg=u'key={}, number={}'.format(key, number))

    @given(variable_name(),
           st.integers())
    def test_set_get_integer2(self, key, number):
        db = GodMap()
        db.set_data([key], number)
        self.assertEqual(db.get_data([key]), number, msg=u'key={}, number={}'.format(key, number))
        self.assertEqual(db.get_data([key]), number, msg=u'key={}, number={}'.format(key, number))

    @given(variable_name(),
           st.floats(allow_nan=False))
    def test_set_get_float(self, key, number):
        db = GodMap()
        db.set_data([key], number)
        db.set_data([key], number)

        self.assertEqual(db.get_data([key]), number)

    @given(variable_name(),
           variable_name(),
           keys_values())
    def test_namedtuple(self, key, tuple_name, key_values):
        Frame = namedtuple(tuple_name, key_values[0])
        db = GodMap()
        db.set_data([key], Frame(*key_values[1]))
        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), v)

    def test_namedtuple1(self):
        Frame = namedtuple(u'Frame', [u'pos'])
        db = GodMap()
        db.set_data([u'f12'], Frame(pos=2))
        with self.assertRaises(AttributeError):
            db.set_data([u'f12', u'pos'], 42)

    @given(variable_name(),
           variable_name(),
           keys_values())
    def test_class1(self, key, class_name, key_values):
        c = type(str(class_name), (object,), {})()
        for k, v in zip(*key_values):
            setattr(c, k, None)

        db = GodMap()
        db.set_data([key], c)

        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), None)
        for k, v in zip(*key_values):
            db.set_data([key, k], v)
        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), v)

    @given(st.lists(variable_name(), unique=True))
    def test_class3(self, class_names):
        db = GodMap()
        for i, name in enumerate(class_names):
            c = type(str(name), (object,), {})()
            db.set_data(class_names[:i + 1], c)

        for i, name in enumerate(class_names):
            c = db.get_data(class_names[:i + 1])
            self.assertEqual(type(c).__name__, name)

    def test_attribute_error(self):
        db = GodMap()

        class C(object):
            asdf = 1

        db.unsafe_set_data(['c'], C())
        try:
            db.get_data(['c', 'a'])
            assert False
        except KeyError as e:
            assert True

    def test_index_error(self):
        db = GodMap()
        db.unsafe_set_data(['l'], [1, 2, 3])
        try:
            db.get_data(['l', '5'])
            assert False
        except KeyError as e:
            assert True

    @given(variable_name(),
           keys_values())
    def test_dict1(self, key, key_values):
        d = {k: v for k, v in zip(*key_values)}
        db = GodMap()
        db.set_data([key], d)
        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), v)

    @given(variable_name(),
           keys_values())
    def test_dict2(self, key, key_values):
        d = {}
        db = GodMap()
        db.set_data([key], d)
        for k, v in zip(*key_values):
            db.set_data([key, k], v)
        for k, v in zip(*key_values):
            self.assertEqual(db.get_data([key, k]), v)

    @given(variable_name(),
           st.lists(variable_name(), min_size=1),
           st.floats(allow_nan=False))
    def test_dict3(self, key, tuple_key, value):
        tuple_key = tuple(tuple_key)
        d = {tuple_key: value}
        db = GodMap()
        db.set_data([key], d)
        self.assertEqual(db.get_data([key, tuple_key]), value)

    @given(variable_name(),
           st.lists(st.floats(allow_nan=False), min_size=1))
    def test_list1(self, key, value):
        db = GodMap()
        db.set_data([key], value)
        for i, v in enumerate(value):
            self.assertEqual(db.get_data([key, i]), v)

    def test_list_double_index(self):
        key = 'asdf'
        value = np.array([[0, 1], [2, 3]])
        db = GodMap()
        db.set_data([key], value)
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                self.assertEqual(db.get_data([key, i, j]), value[i, j])

    @given(variable_name(),
           st.lists(st.floats(allow_nan=False), min_size=1))
    def test_tuple1(self, key, value):
        value = tuple(value)
        db = GodMap()
        db.set_data([key], value)
        for i, v in enumerate(value):
            self.assertEqual(db.get_data([key, i]), v)

    @given(variable_name(),
           lists_of_same_length([st.floats(allow_nan=False), st.floats(allow_nan=False)]))
    def test_list_overwrite_entry(self, key, lists):
        first_values, second_values = lists
        db = GodMap()
        db.set_data([key], first_values)
        for i, v in enumerate(first_values):
            self.assertEqual(db.get_data([key, i]), v)
            db.set_data([key, i], second_values[i])

        for i, v in enumerate(second_values):
            self.assertEqual(db.get_data([key, i]), v)

    @given(variable_name(),
           st.lists(st.floats(allow_nan=False), min_size=1))
    def test_list_index_error(self, key, l):
        db = GodMap()
        db.set_data([key], l)
        with self.assertRaises(IndexError):
            db.set_data([key, len(l) + 1], 0)

    @given(variable_name(),
           st.lists(st.floats(allow_nan=False), min_size=1))
    def test_list_negative_index(self, key, l):
        db = GodMap()
        db.set_data([key], l)
        for i in range(len(l)):
            self.assertEqual(db.get_data([key, -i]), l[-i])

    @given(variable_name(),
           variable_name())
    def test_function_1param_lambda(self, key, key2):
        db = GodMap()
        f = lambda x: x
        db.set_data([key], f)
        self.assertEqual(db.get_data([key, (key2,)]), key2)

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
        db.set_data([key1], d)
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
        db.set_data([key1], d)
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
        db.set_data([key1], d)
        self.assertEqual(key5, db.get_data([key1, key2, (key3, key4), 0]))
        self.assertEqual(key5, db.get_data([key1, key2, (key3, key4), 0]))
        self.assertEqual(key5, db.get_data([key1, key2, (key3, key4), 0]))

    def test_clear_cache(self):
        db = GodMap()

        d = {'b': 'c'}
        db.set_data(['a'], d)
        self.assertEqual('c', db.get_data(['a', 'b']))
        db.clear_cache()

        class C(object):
            b = 'c'

        db.set_data(['a'], C())
        self.assertEqual('c', db.get_data(['a', 'b']))

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
        db.set_data([key1], d)
        self.assertEqual(key3, db.get_data([key1, key2, tuple(), 0]))

    @given(variable_name(),
           st.integers())
    def test_to_symbol(self, key, value):
        gm = GodMap()
        gm.set_data([key], value)
        self.assertTrue(w.is_symbol(gm.to_symbol([key])))
        self.assertTrue(key in str(gm.to_symbol([key])))

    def test_to_symbol_pose_stamped(self):
        gm = GodMap()
        pose = PoseStamped()
        gm.set_data(['muh'], pose)
        result = gm.to_expr(['muh'])
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(result.shape[1], 4)

    @given(lists_of_same_length([variable_name(), st.floats()], unique=True))
    def test_get_symbol_map(self, keys_values):
        keys, values = keys_values
        gm = GodMap()
        for key, value in zip(keys, values):
            gm.set_data([key], value)
            gm.to_symbol([key])
        self.assertEqual(len(gm.get_values(keys)), len(keys))

    def test_to_expr_ndarray(self):
        gm = GodMap()
        data = np.zeros((5, 5))
        gm.set_data(['muh'], data)
        expr = gm.to_expr(['muh'])
        assert expr.shape == data.shape

    def test_to_expr_list(self):
        gm = GodMap()
        data = [1, 2, 3]
        gm.set_data(['muh'], data)
        expr = gm.to_expr(['muh'])
        data[0]
        data[1]
        data[2]


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestDataBus',
                    test=TestGodMap)
