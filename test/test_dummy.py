#!/usr/bin/env python
import unittest

PKG = 'giskardpy'


class TestDummy(unittest.TestCase):
    def test_1(self):
        self.assertEqual(1, 1, '1!=1')


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestDummy',
                    test=TestDummy)
