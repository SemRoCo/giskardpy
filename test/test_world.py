import pytest
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from giskard_msgs.msg import WorldBody

from giskardpy.exceptions import DuplicateNameException
from giskardpy.test_utils import pr2_urdf
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import make_world_body_box, make_world_body_sphere, make_world_body_cylinder, make_urdf_world_body
from giskardpy.world import World
from giskardpy.world_object import WorldObject


@pytest.fixture()
def parsed_pr2():
    """
    :rtype: WorldObject
    """
    return WorldObject.from_urdf_file(u'urdfs/pr2.urdf')


@pytest.fixture()
def parsed_base_bot():
    """
    :rtype: WorldObject
    """
    return WorldObject.from_urdf_file(u'urdfs/2d_base_bot.urdf')

@pytest.fixture()
def empty_world():
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

class TestWorldObj(object):
    def test_from_urdf_file(self, parsed_pr2):
        assert isinstance(parsed_pr2, WorldObject)

class TestWorld(object):
    def test_add_robot(self, empty_world):
        assert len(empty_world.get_objects()) == 0
        assert not empty_world.has_robot()
        pr2 = WorldObject(pr2_urdf())
        empty_world.add_robot(pr2)
        assert empty_world.has_robot()
        assert pr2 == empty_world.get_robot()

    def test_add_object(self, empty_world):
        name = u'muh'
        box = WorldObject.from_world_body(make_world_body_box(name))
        empty_world.add_object(box)
        assert empty_world.has_object(name)
        assert len(empty_world.get_objects()) == 1
        assert empty_world.get_object(box.get_name()) == box

    def test_add_object_with_robot_name(self, world_with_pr2):
        name = u'pr2'
        box = WorldObject.from_world_body(make_world_body_box(name))
        try:
            world_with_pr2.add_object(box)
            assert False, u'expected exception'
        except DuplicateNameException:
            assert True
        assert world_with_pr2.has_robot()
        assert len(world_with_pr2.get_objects()) == 0

    def test_hard_reset1(self, world_with_pr2):
        world_with_pr2.hard_reset()
        assert not world_with_pr2.has_robot()

    def test_hard_reset2(self, world_with_pr2):
        name = u'muh'
        box = WorldObject.from_world_body(make_world_body_box(name))
        world_with_pr2.add_object(box)
        world_with_pr2.hard_reset()
        assert not world_with_pr2.has_robot()
        assert len(world_with_pr2.get_objects()) == 0

    def test_soft_reset1(self, world_with_pr2):
        world_with_pr2.soft_reset()
        assert world_with_pr2.has_robot()

    def test_soft_reset2(self, world_with_pr2):
        name = u'muh'
        box = WorldObject.from_world_body(make_world_body_box(name))
        world_with_pr2.add_object(box)
        world_with_pr2.soft_reset()
        assert world_with_pr2.has_robot()
        assert len(world_with_pr2.get_objects()) == 0

    def test_remove_object1(self, world_with_pr2):
        name = u'muh'
        box = WorldObject.from_world_body(make_world_body_box(name))
        world_with_pr2.add_object(box)
        world_with_pr2.remove_object(name)
        assert not world_with_pr2.has_object(name)
        assert len(world_with_pr2.get_objects()) == 0
        assert world_with_pr2.has_robot()

    def test_remove_object2(self, world_with_pr2):
        name1 = u'muh'
        box = WorldObject.from_world_body(make_world_body_box(name1))
        world_with_pr2.add_object(box)
        name2 = u'muh2'
        box = WorldObject.from_world_body(make_world_body_box(name2))
        world_with_pr2.add_object(box)
        world_with_pr2.remove_object(name1)
        assert not world_with_pr2.has_object(name1)
        assert world_with_pr2.has_object(name2)
        assert len(world_with_pr2.get_objects()) == 1
        assert world_with_pr2.has_robot()


