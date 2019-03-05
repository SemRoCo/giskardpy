import pybullet as p
import pytest

import test_urdf_object
from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.pybullet_world_object import PyBulletWorldObject
import giskardpy.pybullet_wrapper as pbw
from giskardpy.test_utils import pr2_urdf, base_bot_urdf, donbot_urdf, boxy_urdf
from giskardpy.world_object import WorldObject
import test_world




@pytest.fixture(scope=u'module')
def pybullet(request):
    print(u'starting pybullet')
    pbw.start_pybullet(False)

    def kill_pybullet():
        print(u'shutdown pybullet')
        pbw.stop_pybullet()

    request.addfinalizer(kill_pybullet)

@pytest.fixture()
def resetted_pybullet(pybullet):
    """
    :rtype: WorldObject
    """
    pbw.clear_pybullet()
    return pybullet

@pytest.fixture()
def parsed_pr2(resetted_pybullet):
    """
    :rtype: WorldObject
    """
    return PyBulletWorldObject(pr2_urdf())


@pytest.fixture()
def parsed_base_bot(resetted_pybullet):
    """
    :rtype: WorldObject
    """
    return PyBulletWorldObject(base_bot_urdf())

@pytest.fixture()
def parsed_donbot(resetted_pybullet):
    """
    :rtype: Robot
    """
    return PyBulletWorldObject(donbot_urdf())

@pytest.fixture()
def parsed_boxy(resetted_pybullet):
    """
    :rtype: Robot
    """
    return PyBulletWorldObject(boxy_urdf())

@pytest.fixture()
def empty_world(pybullet):
    """
    :rtype: PyBulletWorld
    """
    pbw.clear_pybullet()
    pw = PyBulletWorld(path_to_data_folder=u'../data')
    pw.setup()
    return pw

@pytest.fixture()
def world_with_pr2(empty_world, parsed_pr2):
    """
    :rtype: PyBulletWorld
    """
    empty_world.add_robot(parsed_pr2)
    return empty_world

def assert_num_pybullet_objects(num):
    assert p.getNumBodies() == num, pbw.print_body_names()

class TestPyBulletWorldObject(test_world.TestWorldObj):
    cls = PyBulletWorldObject
    def test_create_object(self, parsed_base_bot):
        assert_num_pybullet_objects(1)
        assert u'pointy' in pbw.get_body_names()

    def test_detach_object(self, parsed_base_bot):
        super(TestPyBulletWorldObject, self).test_detach_object(parsed_base_bot)


# def test_robot(self, pybullet):
    #     pr2 =
    #     assert len(empty_world.get_objects()) == 0
    #     assert not empty_world.has_robot()
    #     pr2 = WorldObject(pr2_urdf())
    #     empty_world.add_robot(pr2)
    #     assert empty_world.has_robot()
    #     assert pr2 == empty_world.get_robot()


class TestPyBulletWorld(test_world.TestWorld):
    cls = PyBulletWorldObject

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
        assert_num_pybullet_objects(3)
















