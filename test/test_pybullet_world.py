import unittest

from hypothesis.strategies import composite

from giskardpy.exceptions import UnknownBodyException, RobotExistsException, DuplicateNameException
from giskardpy.object import UrdfObject, Box, Sphere, Cylinder
from giskardpy.pybullet_world import PyBulletWorld
import pybullet as p
import hypothesis.strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, invariant
from giskardpy.data_types import SingleJointState, Transform, Point, Quaternion
from giskardpy.test_utils import variable_name, robot_urdfs


def small_float():
    return st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)

@composite
def transform(draw):
    p = Point(draw(small_float()), draw(small_float()), draw(small_float()))
    q = Quaternion(draw(small_float()), draw(small_float()), draw(small_float()), draw(small_float()))
    return Transform(p, q)


class TestPyBulletWorld(RuleBasedStateMachine):
    # FIXME
    def __init__(self):
        super(TestPyBulletWorld, self).__init__()
        self.world = PyBulletWorld()
        self.world.setup()

    object_names = Bundle(u'object_names')
    robot_names = Bundle(u'robot_names')

    @invariant()
    def keeping_track_of_bodies(self):
        assert len(self.world.get_object_names()) + self.world.has_robot() == p.getNumBodies()

    @rule(target=object_names,
          name=variable_name(),
          length=small_float(),
          width=small_float(),
          height=small_float(),
          base_pose=transform())
    def add_box(self, name, length, width, height, base_pose):
        robot_existed = self.world.has_robot()
        object_existed = name in self.world.get_object_names()

        object = Box(name, length, width, height)
        try:
            self.world.spawn_urdf_object(object, base_pose)
            assert name in self.world.get_object_names()
        except DuplicateNameException:
            assert object_existed or robot_existed
        return name

    @rule(target=object_names,
          name=variable_name(),
          radius=small_float(),
          base_pose=transform())
    def add_sphere(self, name, radius, base_pose):
        robot_existed = self.world.has_robot()
        object_existed = self.world.has_object(name)

        object = Sphere(name, radius)
        try:
            self.world.spawn_urdf_object(object, base_pose)
            assert self.world.has_object(name)
        except DuplicateNameException:
            assert object_existed or robot_existed
        return name

    @rule(target=object_names,
          name=variable_name(),
          radius=small_float(),
          length=small_float(),
          base_pose=transform())
    def add_cylinder(self, name, radius, length, base_pose):
        robot_existed = self.world.has_robot()
        object_existed = self.world.has_object(name)
        object = Cylinder(name, radius, length)
        try:
            self.world.spawn_urdf_object(object, base_pose)
            assert self.world.has_object(name)
        except DuplicateNameException:
            assert object_existed or robot_existed
        return name

    @rule(target=robot_names,
          name=variable_name(),
          robot_urdf=robot_urdfs(),
          base_pose=transform())
    def spawn_robot(self, name, robot_urdf, base_pose):
        robot_existed = self.world.has_robot()
        object_existed = self.world.has_object(name)
        try:
            self.world.spawn_robot_from_urdf_file(name, robot_urdf, base_pose)
            assert self.world.has_robot()
            assert self.world.get_robot().name == name
        except RobotExistsException:
            assert robot_existed
            assert self.world.has_robot()
        except DuplicateNameException:
            assert object_existed
            assert not self.world.has_robot()
        return name

    @rule()
    def delete_robot(self):
        self.world.remove_robot()
        assert not self.world.has_robot()
        assert self.world.get_robot() is None

    @rule(name=object_names)
    def delete_object(self, name):
        object_existed = self.world.has_object(name)
        try:
            self.world.delete_object(name),
        except UnknownBodyException:
            assert not object_existed

        assert not self.world.has_object(name)

    @rule(remaining_objects=st.lists(object_names))
    def delete_all_objects(self, remaining_objects):
        old_objects = set(self.world.get_object_names())
        self.world.delete_all_objects(remaining_objects)
        for object_name in remaining_objects:
            if object_name in old_objects:
                assert self.world.has_object(object_name)
        assert len(self.world.get_object_names()) == len(old_objects.intersection(set(remaining_objects)))

    @rule()
    def clear_world(self):

        self.world.soft_reset()
        assert 1 == p.getNumBodies()

    def teardown(self):
        self.world.deactivate_viewer()


TestTrees = TestPyBulletWorld.TestCase

if __name__ == '__main__':
    unittest.main()
