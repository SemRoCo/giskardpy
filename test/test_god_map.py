import unittest
from collections import namedtuple
from hypothesis import given, reproduce_failure, assume
import hypothesis.strategies as st
import giskardpy.symengine_wrappers as sw
from giskardpy.god_map import GodMap
from giskardpy.test_utils import variable_name, keys_values, lists_of_same_length

PKG = u'giskardpy'


class TestGodMap(unittest.TestCase):
    @given(variable_name(),
           st.integers())
    def test_set_get_integer(self, key, number):
        db = GodMap()
        db.set_data([key], number)
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
            db.set_data([key, len(l)+1], 0)

    @given(variable_name(),
           st.lists(st.floats(allow_nan=False), min_size=1))
    def test_list_negative_index(self, key, l):
        db = GodMap()
        db.set_data([key], l)
        for i in range(len(l)):
            self.assertEqual(db.get_data([key, -i]), l[-i])

    @given(variable_name(),
           st.floats(allow_nan=False))
    def test_function1(self, key, value):
        # TODO not clean that i try to call every function
        db = GodMap()
        f = lambda gm: value
        db.set_data([key], f)
        self.assertEqual(db.get_data([key]), value)

    @given(variable_name(), variable_name(), st.floats(allow_nan=False))
    def test_function2(self, key, dict_key, return_value):
        db = GodMap()

        class MUH(object):
            def __call__(self, god_map):
                return return_value

        a = MUH()
        d = {dict_key: a}
        db.set_data([key], d)
        self.assertEqual(db.get_data([key, dict_key]), return_value)

    @given(variable_name(),
           st.integers())
    def test_to_symbol(self, key, value):
        gm = GodMap()
        gm.set_data([key], value)
        self.assertTrue(isinstance(gm.to_symbol([key]), sw.Symbol))
        self.assertTrue(key in str(gm.to_symbol([key])))

    @given(lists_of_same_length([variable_name(), st.floats()], unique=True))
    def test_get_symbol_map(self, keys_values):
        keys, values = keys_values
        gm = GodMap()
        for key, value in zip(keys, values):
            gm.set_data([key], value)
            gm.to_symbol([key])
        self.assertEqual(len(gm.get_symbol_map()), len(keys))


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestDataBus',
                    test=TestGodMap)
