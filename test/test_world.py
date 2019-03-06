import os
import shutil

import pytest
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from giskard_msgs.msg import WorldBody

import test_urdf_object
from giskardpy.exceptions import DuplicateNameException
from giskardpy.test_utils import pr2_urdf, donbot_urdf, boxy_urdf, base_bot_urdf
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import make_world_body_box, make_world_body_sphere, make_world_body_cylinder, make_urdf_world_body
from giskardpy.world import World
from giskardpy.world_object import WorldObject


@pytest.fixture(scope=u'module')
def module_setup(request):
    pass

@pytest.fixture()
def function_setup(request, module_setup):
    """
    :rtype: WorldObject
    """
    pass

@pytest.fixture()
def parsed_pr2(function_setup):
    """
    :rtype: WorldObject
    """
    return WorldObject(pr2_urdf())


@pytest.fixture()
def parsed_base_bot(function_setup):
    """
    :rtype: WorldObject
    """
    return WorldObject(base_bot_urdf())

@pytest.fixture()
def parsed_donbot(function_setup):
    """
    :rtype: Robot
    """
    return WorldObject(donbot_urdf())

@pytest.fixture()
def parsed_boxy(function_setup):
    """
    :rtype: Robot
    """
    return WorldObject(boxy_urdf())

@pytest.fixture()
def empty_world(function_setup):
    """
    :rtype: World
    """
    return World()

@pytest.fixture()
def world_with_pr2(parsed_pr2):
    """
    :rtype: World
    """
    w = World()
    w.add_robot(parsed_pr2)
    return w

@pytest.fixture()
def test_folder(request):
    """
    :rtype: World
    """
    folder_name = u'tmp_data/'
    def kill_pybullet():
        shutil.rmtree(folder_name)

    request.addfinalizer(kill_pybullet)
    return folder_name

class TestWorldObj(test_urdf_object.TestUrdfObject):
    cls = WorldObject
    def test_from_urdf_file(self, parsed_pr2):
        assert isinstance(parsed_pr2, WorldObject)

    def test_safe_load_collision_matrix(self, parsed_pr2, test_folder):
        parsed_pr2.init_self_collision_matrix()
        scm = parsed_pr2.get_self_collision_matrix()
        parsed_pr2.safe_self_collision_matrix(test_folder)
        parsed_pr2.load_self_collision_matrix(test_folder)
        assert scm == parsed_pr2.get_self_collision_matrix()

class TestWorld(object):
    cls = WorldObject
    def test_add_robot(self, empty_world):
        assert len(empty_world.get_objects()) == 0
        assert not empty_world.has_robot()
        pr2 = self.cls(pr2_urdf())
        empty_world.add_robot(pr2)
        assert empty_world.has_robot()
        assert pr2 == empty_world.get_robot()

    def test_add_object(self, empty_world):
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        empty_world.add_object(box)
        assert empty_world.has_object(name)
        assert len(empty_world.get_objects()) == 1
        assert empty_world.get_object(box.get_name()) == box

    def test_add_object_twice(self, empty_world):
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        empty_world.add_object(box)
        try:
            empty_world.add_object(box)
            assert False, u'expected exception'
        except DuplicateNameException:
            assert True
        assert empty_world.has_object(name)
        assert len(empty_world.get_objects()) == 1
        assert empty_world.get_object(box.get_name()) == box

    def test_add_object_with_robot_name(self, world_with_pr2):
        name = u'pr2'
        box = self.cls.from_world_body(make_world_body_box(name))
        try:
            world_with_pr2.add_object(box)
            assert False, u'expected exception'
        except DuplicateNameException:
            assert True
        assert world_with_pr2.has_robot()
        assert len(world_with_pr2.get_objects()) == 0

    def test_attach_existing_obj_to_robot(self, world_with_pr2):
        box = self.cls.from_world_body(make_world_body_box())
        world_with_pr2.add_object(box)
        p = Pose()
        p.orientation.w = 1
        links_before = set(world_with_pr2.get_robot().get_link_names())
        joints_before = set(world_with_pr2.get_robot().get_joint_names())
        world_with_pr2.attach_existing_obj_to_robot(u'box', u'l_gripper_tool_frame', p)
        assert u'box' not in world_with_pr2.get_object_names()
        assert set(world_with_pr2.get_robot().get_link_names()).difference(links_before) == {u'box'}
        assert set(world_with_pr2.get_robot().get_joint_names()).difference(joints_before) == {u'box'}

    def test_hard_reset1(self, world_with_pr2):
        world_with_pr2.hard_reset()
        assert not world_with_pr2.has_robot()

    def test_hard_reset2(self, world_with_pr2):
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_pr2.add_object(box)
        world_with_pr2.hard_reset()
        assert not world_with_pr2.has_robot()
        assert len(world_with_pr2.get_objects()) == 0

    def test_soft_reset1(self, world_with_pr2):
        world_with_pr2.soft_reset()
        assert world_with_pr2.has_robot()

    def test_soft_reset2(self, world_with_pr2):
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_pr2.add_object(box)
        world_with_pr2.soft_reset()
        assert world_with_pr2.has_robot()
        assert len(world_with_pr2.get_objects()) == 0

    def test_remove_object1(self, world_with_pr2):
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_pr2.add_object(box)
        world_with_pr2.remove_object(name)
        assert not world_with_pr2.has_object(name)
        assert len(world_with_pr2.get_objects()) == 0
        assert world_with_pr2.has_robot()

    def test_remove_object2(self, world_with_pr2):
        name1 = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name1))
        world_with_pr2.add_object(box)
        name2 = u'muh2'
        box = self.cls.from_world_body(make_world_body_box(name2))
        world_with_pr2.add_object(box)
        world_with_pr2.remove_object(name1)
        assert not world_with_pr2.has_object(name1)
        assert world_with_pr2.has_object(name2)
        assert len(world_with_pr2.get_objects()) == 1
        assert world_with_pr2.has_robot()


