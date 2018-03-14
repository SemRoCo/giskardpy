import unittest
from collections import namedtuple

from giskardpy.god_map import GodMap

PKG = 'giskardpy'


class TestDataBus(unittest.TestCase):
    def test_set1(self):
        db = GodMap()
        db.set_data('asdf', 2)
        self.assertEqual(db.get_data('asdf'), 2)

    def test_set2(self):
        db = GodMap()
        db.set_data('asdf', 2)
        db.set_data('asdf', 2)
        self.assertEqual(db.get_data('asdf'), 2)

    def test_namedtuple(self):
        Frame = namedtuple('Frame', ['pos'])
        db = GodMap()
        db.set_data('f12', Frame(pos=2))
        self.assertEqual(db.get_data('f12/pos'), 2)

    def test_namedtuple1(self):
        Frame = namedtuple('Frame', ['pos'])
        db = GodMap()
        db.set_data('f12', Frame(pos=2))
        with self.assertRaises(AttributeError):
            db.set_data('f12/pos', 42)

    def test_class1(self):
        class C1(object):
            asdf = 32
        db = GodMap()
        db.set_data('asdf', C1())
        self.assertEqual(db.get_data('asdf/asdf'), 32)

    def test_class2(self):
        class C1(object):
            asdf = 32
        db = GodMap()
        db.set_data('asdf', C1())
        db.set_data('asdf/asdf', 2)
        self.assertEqual(db.get_data('asdf/asdf'), 2)

    def test_class4(self):
        class C1(object):
            asdf = 32
        db = GodMap()
        db.set_data('asdf', C1())
        db.set_data('asdf/asdff', 2)
        self.assertEqual(db.get_data('asdf/asdff'), 2)

    def test_class3(self):
        class C1(object):
            asdf = 32
        db = GodMap()
        db.set_data('asdf', C1())
        db.set_data('asdf/asdf', C1)
        self.assertEqual(db.get_data('asdf/asdf/asdf'), 32)

    def test_dict1(self):
        d = {'fu': 1, 'ba': 2}
        db = GodMap()
        db.set_data('asdf', d)
        self.assertEqual(db.get_data('asdf/fu'), 1)
        self.assertEqual(db.get_data('asdf/ba'), 2)

    def test_dict2(self):
        d = {'fu': 1, 'ba': 2}
        db = GodMap()
        db.set_data('asdf', d)
        db.set_data('asdf/fu', 42)
        db.set_data('asdf/lulu', 1337)
        self.assertEqual(db.get_data('asdf/fu'), 42)
        self.assertEqual(db.get_data('asdf/ba'), 2)
        self.assertEqual(db.get_data('asdf/lulu'), 1337)

    def test_list1(self):
        l = range(10)
        db = GodMap()
        db.set_data('asdf', l)
        for i in l:
            self.assertEqual(db.get_data('asdf/{}'.format(i)), i)

    def test_tuple1(self):
        l = (0,1,2,3,4)
        db = GodMap()
        db.set_data('asdf', l)
        for i in l:
            self.assertEqual(db.get_data('asdf/{}'.format(i)), i)

    def test_list2(self):
        l = range(10)
        db = GodMap()
        db.set_data('asdf', l)
        db.set_data('asdf/0', 42)
        self.assertEqual(db.get_data('asdf/0'), 42)

    def test_list3(self):
        l = range(10)
        db = GodMap()
        db.set_data('asdf', l)
        with self.assertRaises(IndexError):
            db.set_data('asdf/11111', 42)


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestDataBus',
                    test=TestDataBus)
