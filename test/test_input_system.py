import unittest
from itertools import chain

from hypothesis import given
import hypothesis.strategies as st

from giskardpy.god_map import GodMap
from giskardpy.input_system import JointStatesInput, Point3Input, Vector3Input, FrameInput
from utils_for_tests import variable_name

PKG = u'giskardpy'



class TestInputSystem(unittest.TestCase):

    @given(st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()))
    def test_joint_states_input(self, joint_names, prefix, suffix):
        gm = GodMap()
        js_input = JointStatesInput(gm.to_symbol, joint_names, prefix, suffix)
        for i, (joint_name, joint_symbol) in enumerate(js_input.joint_map.items()):
            self.assertTrue(joint_name in joint_names)
            symbol_str = str(joint_symbol)
            for e in prefix:
                self.assertTrue(e in symbol_str)
            for e in suffix:
                self.assertTrue(e in symbol_str)

    @given(st.lists(variable_name(), min_size=1),
           st.lists(variable_name(), min_size=1),
           st.lists(variable_name(), min_size=1),
           st.lists(variable_name()),
           st.lists(variable_name()))
    def test_point3_input(self, x, y, z, prefix, suffix):
        gm = GodMap()
        input = Point3Input(gm.to_symbol, prefix, suffix, x, y, z)
        x_symbol = str(input.x)
        y_symbol = str(input.y)
        z_symbol = str(input.z)
        for e in chain(x, prefix, suffix):
            self.assertTrue(e in x_symbol)
        for e in chain(y, prefix, suffix):
            self.assertTrue(e in y_symbol)
        for e in chain(z, prefix, suffix):
            self.assertTrue(e in z_symbol)

    @given(st.lists(variable_name(), min_size=1),
           st.lists(variable_name(), min_size=1),
           st.lists(variable_name(), min_size=1),
           st.lists(variable_name()),
           st.lists(variable_name()))
    def test_vector3_input(self, x, y, z, prefix, suffix):
        gm = GodMap()
        input = Vector3Input(gm.to_symbol, prefix, suffix, x, y, z)
        x_symbol = str(input.x)
        y_symbol = str(input.y)
        z_symbol = str(input.z)
        for e in chain(x, prefix, suffix):
            self.assertTrue(e in x_symbol)
        for e in chain(y, prefix, suffix):
            self.assertTrue(e in y_symbol)
        for e in chain(z, prefix, suffix):
            self.assertTrue(e in z_symbol)

    @given(st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()),
           st.lists(variable_name()))
    def test_frame_input(self, translation_prefix, translation_suffix, rotation_prefix, rotation_suffix,
                         x, y, z, qx, qy, qz, qw):
        gm = GodMap()
        input = FrameInput(gm.to_symbol, translation_prefix, translation_suffix, rotation_prefix, rotation_suffix,
                           x, y, z, qx, qy, qz, qw)
        x_symbol = str(input.x)
        y_symbol = str(input.y)
        z_symbol = str(input.z)
        qx_symbol = str(input.qx)
        qy_symbol = str(input.qy)
        qz_symbol = str(input.qz)
        qw_symbol = str(input.qw)
        for e in chain(x, translation_prefix, translation_suffix):
            self.assertTrue(e in x_symbol)
        for e in chain(y, translation_prefix, translation_suffix):
            self.assertTrue(e in y_symbol)
        for e in chain(z, translation_prefix, translation_suffix):
            self.assertTrue(e in z_symbol)
        for e in chain(qx, rotation_prefix, rotation_suffix):
            self.assertTrue(e in qx_symbol)
        for e in chain(qy, rotation_prefix, rotation_suffix):
            self.assertTrue(e in qy_symbol)
        for e in chain(qz, rotation_prefix, rotation_suffix):
            self.assertTrue(e in qz_symbol)
        for e in chain(qw, rotation_prefix, rotation_suffix):
            self.assertTrue(e in qw_symbol)



if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestDataBus',
                    test=TestInputSystem)
