import shutil
import giskardpy
giskardpy.WORLD_IMPLEMENTATION = None

import pytest
from geometry_msgs.msg import Pose
from giskard_msgs.msg import CollisionEntry

import test_urdf_object
from giskardpy.exceptions import DuplicateNameException
from utils_for_tests import pr2_urdf, donbot_urdf, boxy_urdf, base_bot_urdf
from giskardpy.utils import make_world_body_box
from giskardpy.world import World
from giskardpy.world_object import WorldObject
import numpy as np

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
    w.add_robot(parsed_pr2, None, parsed_pr2.controlled_joints, 0, 0)
    return w

@pytest.fixture()
def world_with_donbot(parsed_donbot):
    """
    :type parsed_donbot: WorldObject
    :rtype: World
    """
    w = World()
    w.add_robot(parsed_donbot, None, parsed_donbot.controlled_joints, 0, 0)
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

    def test_base_pose1(self, parsed_pr2):
        p = Pose()
        p.orientation.w = 1
        parsed_pr2.base_pose = p
        assert parsed_pr2.base_pose == p

    def test_base_pose2(self, parsed_pr2):
        p = Pose()
        p.orientation.w = 10
        parsed_pr2.base_pose = p
        orientation = parsed_pr2.base_pose.orientation
        orientation_vector = [orientation.x,
                              orientation.y,
                              orientation.z,
                              orientation.w]
        assert np.linalg.norm(orientation_vector) == 1

    def test_joint_state(self, parsed_pr2):
        js = parsed_pr2.get_zero_joint_state()
        parsed_pr2.joint_state = js
        assert parsed_pr2.joint_state == js

class TestWorld(object):
    cls = WorldObject
    def test_add_robot(self, empty_world):
        assert len(empty_world.get_objects()) == 0
        assert not empty_world.has_robot()
        pr2 = self.cls(pr2_urdf())
        empty_world.add_robot(pr2, None, pr2.controlled_joints, 0, 0)
        assert empty_world.has_robot()
        assert pr2 == empty_world.robot

    def test_add_object(self, empty_world):
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        empty_world.add_object(box)
        assert empty_world.has_object(name)
        assert len(empty_world.get_objects()) == 1
        assert len(empty_world.get_object_names()) == 1
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

    def test_attach_existing_obj_to_robot1(self, world_with_pr2):
        box = self.cls.from_world_body(make_world_body_box())
        world_with_pr2.add_object(box)
        links_before = set(world_with_pr2.robot.get_link_names())
        joints_before = set(world_with_pr2.robot.get_joint_names())
        p = Pose()
        p.orientation.w = 1
        world_with_pr2.attach_existing_obj_to_robot(u'box', u'l_gripper_tool_frame', p)
        assert u'box' not in world_with_pr2.get_object_names()
        assert set(world_with_pr2.robot.get_link_names()).difference(links_before) == {u'box'}
        assert set(world_with_pr2.robot.get_joint_names()).difference(joints_before) == {u'box'}

    def test_attach_existing_obj_to_robot2(self, world_with_pr2):
        box = self.cls.from_world_body(make_world_body_box())
        world_with_pr2.add_object(box)
        p = Pose()
        p.orientation.w = 1
        try:
            world_with_pr2.attach_existing_obj_to_robot(u'box2', u'l_gripper_tool_frame', p)
            assert False
        except KeyError:
            assert True

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

    def test_collision_goals_to_collision_matrix1(self, world_with_donbot):
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], 0.05)
        assert len(collision_matrix) == 0

    def test_collision_goals_to_collision_matrix2(self, world_with_donbot):
        name = u'muh'
        min_dist = 0.05
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], min_dist)
        assert len(collision_matrix) == len(world_with_donbot.robot.get_controlled_links())
        robot_link_names = world_with_donbot.robot.get_link_names()
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist
            assert body_b == name
            assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix3(self, world_with_donbot):
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_ALL_COLLISIONS
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, 0.05)
        assert len(collision_matrix) == len(world_with_donbot.robot.get_controlled_links())
        robot_link_names = world_with_donbot.robot.get_link_names()
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == ce.min_dist
            assert body_b == name
            assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix4(self, world_with_donbot):
        name = u'muh'

        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        ces.append(ce)
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, 0.05)

        assert len(collision_matrix) == 0

    def test_collision_goals_to_collision_matrix5(self, world_with_donbot):
        name = u'muh'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())

        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        ces.append(ce)
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [robot_link_names[0]]
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, 0.05)

        assert len(collision_matrix) == 1
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == ce.min_dist
            assert body_b == name
            assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix6(self, world_with_donbot):
        name = u'muh'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())
        min_dist = 0.05

        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [robot_link_names[0]]
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)

        assert len(collision_matrix) == len(robot_link_names)-1
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist
            assert body_b == name
            assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix7(self, world_with_donbot):
        name = u'muh'
        name2 = u'muh2'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())
        min_dist = 0.05

        box = self.cls.from_world_body(make_world_body_box(name))
        box2 = self.cls.from_world_body(make_world_body_box(name2))
        world_with_donbot.add_object(box)
        world_with_donbot.add_object(box2)

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.body_b = name2
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)

        assert len(collision_matrix) == len(robot_link_names)
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist
            assert body_b == name
            assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix8(self, world_with_donbot):
        name = u'muh'
        name2 = u'muh2'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())
        min_dist = 0.05

        box = self.cls.from_world_body(make_world_body_box(name))
        box2 = self.cls.from_world_body(make_world_body_box(name2))
        world_with_donbot.add_object(box)
        world_with_donbot.add_object(box2)

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [robot_link_names[0]]
        ce.body_b = name2
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)

        assert len(collision_matrix) == len(robot_link_names)*2-1
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist
            assert body_b == body_b_link
            assert robot_link in robot_link_names
            if body_b == name2:
                assert robot_link != robot_link_names[0]
