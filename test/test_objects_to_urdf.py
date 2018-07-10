import unittest

from giskardpy.object import WorldObject, to_urdf_string, MeshShape, ColorRgba, MaterialProperty, VisualProperty, \
    BoxShape, FixedJoint
from giskardpy.data_types import Transform, Point, Quaternion

PKG = 'giskardpy'

class TestObjectUrdfGen(unittest.TestCase):
    def test_named_empty_object(self):
        my_obj = WorldObject(name='foo')
        urdf_string = to_urdf_string(my_obj)
        self.assertEqual(urdf_string, '<robot name="foo"><link name="foo_link"/></robot>')

    def test_named_empty_object_without_robot_tag(self):
        my_obj = WorldObject(name='foo')
        urdf_string = to_urdf_string(my_obj, skip_robot_tag=True)
        self.assertEqual(urdf_string, '<link name="foo_link"/>')

    def test_transform_with_translation(self):
        my_obj = Transform(translation=Point(1.1, 2.2, 3.3))
        urdf_string = to_urdf_string(my_obj)
        self.assertEqual(urdf_string, '<origin rpy="0.0 -0.0 0.0" xyz="1.1 2.2 3.3"/>')

    def test_transform_with_rotation(self):
        my_obj = Transform(rotation=Quaternion(0.707, 0.0, 0.707, 0.0))
        urdf_string = to_urdf_string(my_obj)
        self.assertEqual(urdf_string, '<origin rpy="0.0 -1.57079632679 -3.14159265359" xyz="0.0 0.0 0.0"/>')

    def test_mesh_shape(self):
        my_obj = MeshShape(filename='foo.bar', scale=[0.1, 0.2, 0.3])
        urdf_string = to_urdf_string(my_obj)
        self.assertEqual(urdf_string, '<geometry><mesh filename="foo.bar" scale="0.1 0.2 0.3"/></geometry>')

    def test_color_red(self):
        my_obj = MaterialProperty(name='Red', color=ColorRgba(0.9, 0.0, 0.0, 1.0))
        urdf_string = to_urdf_string(my_obj)
        self.assertEqual(urdf_string, '<material name="Red"><color rgba="0.9 0.0 0.0 1.0"/></material>')

    def test_obj_with_box_visual(self):
        my_obj = WorldObject(name='my_box', visual_props=[VisualProperty(geometry=BoxShape(0.5, 1.5, 2.5))])
        urdf_string = to_urdf_string(my_obj)
        self.assertEqual(urdf_string, '<robot name="my_box"><link name="my_box_link"><visual><origin rpy="0.0 -0.0 0.0" xyz="0.0 0.0 0.0"/><geometry><box size="0.5 1.5 2.5"/></geometry></visual></link></robot>')

    def test_fixed_joint(self):
        my_joint = FixedJoint('a_joint', Transform(), 'from_link', 'to_link')
        urdf_string = to_urdf_string(my_joint)
        self.assertEqual('<joint name="a_joint" type="fixed"><origin rpy="0.0 -0.0 0.0" xyz="0.0 0.0 0.0"/><parent link="from_link"/><child link="to_link"/></joint>', urdf_string)