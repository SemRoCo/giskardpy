import pybullet as p
import shutil

import pytest

from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.pybullet_world_object import PyBulletWorldObject
import giskardpy.pybullet_wrapper as pbw
from utils_for_tests import pr2_urdf, base_bot_urdf, donbot_urdf, boxy_urdf
from giskardpy.world_object import WorldObject
import test_world




@pytest.fixture(scope=u'module')
def module_setup(request):
    print(u'starting pybullet')
    pbw.start_pybullet(False)

    def kill_pybullet():
        print(u'shutdown pybullet')
        pbw.stop_pybullet()

    request.addfinalizer(kill_pybullet)

@pytest.fixture()
def function_setup(request, module_setup):
    """
    :rtype: WorldObject
    """
    pbw.clear_pybullet()
    def kill_pybullet():
        print(u'resetting pybullet')
        pbw.clear_pybullet()

    request.addfinalizer(kill_pybullet)

@pytest.fixture()
def parsed_pr2(function_setup):
    """
    :rtype: WorldObject
    """
    return PyBulletWorldObject(pr2_urdf())


@pytest.fixture()
def parsed_base_bot(function_setup):
    """
    :rtype: WorldObject
    """
    return PyBulletWorldObject(base_bot_urdf())

@pytest.fixture()
def parsed_donbot(function_setup):
    """
    :rtype: Robot
    """
    return PyBulletWorldObject(donbot_urdf())

@pytest.fixture()
def parsed_boxy(function_setup):
    """
    :rtype: Robot
    """
    return PyBulletWorldObject(boxy_urdf())

@pytest.fixture()
def empty_world(function_setup):
    """
    :rtype: PyBulletWorld
    """
    pbw.clear_pybullet()
    pw = PyBulletWorld(path_to_data_folder=u'../data')
    return pw

@pytest.fixture()
def world_with_pr2(function_setup):
    """
    :rtype: World
    """
    w = PyBulletWorld(path_to_data_folder=u'../data')
    pr2 = WorldObject(pr2_urdf())
    w.add_robot(pr2, None, pr2.controlled_joints, 0, 0)
    return w

@pytest.fixture()
def world_with_donbot(parsed_donbot):
    """
    :type parsed_donbot: WorldObject
    :rtype: World
    """
    w = PyBulletWorld(path_to_data_folder=u'../data')
    donbot = WorldObject(pr2_urdf())
    w.add_robot(donbot, None, donbot.controlled_joints, 0, 0)
    return w

@pytest.fixture()
def test_folder(request):
    """
    :rtype: World
    """
    folder_name = u'tmp_data/'
    def kill_pybullet():
        try:
            shutil.rmtree(folder_name)
        except Exception:
            pass

    request.addfinalizer(kill_pybullet)
    return folder_name

def assert_num_pybullet_objects(num):
    assert p.getNumBodies() == num, pbw.print_body_names()

class TestPyBulletWorldObject(test_world.TestWorldObj):
    cls = PyBulletWorldObject
    def test_create_object(self, parsed_base_bot):
        assert_num_pybullet_objects(1)
        assert u'pointy' in pbw.get_body_names()


class TestPyBulletWorld(test_world.TestWorld):
    cls = WorldObject

    def test_add_robot(self, empty_world):
        super(TestPyBulletWorld, self).test_add_robot(empty_world)
        assert_num_pybullet_objects(3)

    def test_add_object(self, empty_world):
        super(TestPyBulletWorld, self).test_add_object(empty_world)
        assert_num_pybullet_objects(3)

    def test_add_object_twice(self, empty_world):
        super(TestPyBulletWorld, self).test_add_object_twice(empty_world)
        assert_num_pybullet_objects(3)

    def test_add_object_with_robot_name(self, world_with_pr2):
        super(TestPyBulletWorld, self).test_add_object_with_robot_name(world_with_pr2)
        assert_num_pybullet_objects(3)

    def test_hard_reset1(self, world_with_pr2):
        super(TestPyBulletWorld, self).test_hard_reset1(world_with_pr2)
        assert_num_pybullet_objects(2)

    def test_hard_reset2(self, world_with_pr2):
        super(TestPyBulletWorld, self).test_hard_reset2(world_with_pr2)
        assert_num_pybullet_objects(2)

    def test_soft_reset1(self, world_with_pr2):
        super(TestPyBulletWorld, self).test_soft_reset1(world_with_pr2)
        assert_num_pybullet_objects(3)

    def test_soft_reset2(self, world_with_pr2):
        super(TestPyBulletWorld, self).test_soft_reset2(world_with_pr2)
        assert_num_pybullet_objects(3)

    def test_remove_object1(self, world_with_pr2):
        super(TestPyBulletWorld, self).test_remove_object1(world_with_pr2)
        assert_num_pybullet_objects(3)

    def test_remove_object2(self, world_with_pr2):
        super(TestPyBulletWorld, self).test_remove_object2(world_with_pr2)
        assert_num_pybullet_objects(4)

    def test_attach_existing_obj_to_robot(self, world_with_pr2):
        super(TestPyBulletWorld, self).test_attach_existing_obj_to_robot1(world_with_pr2)
        assert_num_pybullet_objects(3)















