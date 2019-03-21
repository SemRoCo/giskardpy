import shutil
import giskardpy

giskardpy.WORLD_IMPLEMENTATION = None

from giskardpy.symengine_robot import Robot
import pytest
from geometry_msgs.msg import Pose, Point, Quaternion
from giskard_msgs.msg import CollisionEntry
import test_urdf_object
from giskardpy.exceptions import DuplicateNameException
from utils_for_tests import pr2_urdf, donbot_urdf, boxy_urdf, base_bot_urdf, compare_poses
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

    def test_safe_load_collision_matrix(self, test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        r.update_self_collision_matrix()
        scm = r.get_self_collision_matrix()
        r.safe_self_collision_matrix(test_folder)
        r.load_self_collision_matrix(test_folder)
        assert scm == r.get_self_collision_matrix()

    def test_safe_load_collision_matrix2(self, test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        r.update_self_collision_matrix()
        scm = r.get_self_collision_matrix()

        box = self.cls.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        r.attach_urdf_object(box, u'gripper_tool_frame', p)
        r.update_self_collision_matrix()
        scm_with_obj = r.get_self_collision_matrix()

        r.detach_sub_tree(box.get_name())
        r.load_self_collision_matrix(test_folder)
        assert scm == r.get_self_collision_matrix()

        r.attach_urdf_object(box, u'gripper_tool_frame', p)
        r.load_self_collision_matrix(test_folder)
        assert scm_with_obj == r.get_self_collision_matrix()

    def test_base_pose1(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        p = Pose()
        p.orientation.w = 1
        parsed_pr2.base_pose = p
        assert parsed_pr2.base_pose == p

    def test_base_pose2(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        p = Pose()
        p.orientation.w = 10
        parsed_pr2.base_pose = p
        orientation = parsed_pr2.base_pose.orientation
        orientation_vector = [orientation.x,
                              orientation.y,
                              orientation.z,
                              orientation.w]
        assert np.linalg.norm(orientation_vector) == 1

    def test_joint_state(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        js = parsed_pr2.get_zero_joint_state()
        parsed_pr2.joint_state = js
        assert parsed_pr2.joint_state == js

    def test_controlled_joints(self, function_setup):
        controlled_joints = [u'torso_lift_joint']
        wo = self.cls(pr2_urdf(), controlled_joints=controlled_joints)
        assert wo.controlled_joints == controlled_joints


class TestRobot(TestWorldObj):
    cls = Robot

    def test_safe_load_collision_matrix(self, test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder, calc_self_collision_matrix=True)
        scm = r.get_self_collision_matrix()
        assert len(scm) == 0

class TestWorld(object):
    cls = WorldObject
    world_cls = World

    def make_world_with_pr2(self, path_to_data_folder=None):
        """
        :rtype: World
        """
        w = self.world_cls(path_to_data_folder=path_to_data_folder)
        r = self.cls(pr2_urdf())
        w.add_robot(r, None, r.controlled_joints, 0, 0, path_to_data_folder is not None)
        return w

    def make_world_with_donbot(self, path_to_data_folder=None):
        """
        :rtype: World
        """
        w = self.world_cls(path_to_data_folder=path_to_data_folder)
        r = self.cls(donbot_urdf())
        w.add_robot(r, None, r.controlled_joints, 0, 0, path_to_data_folder is not None)
        return w

    def test_add_robot(self, function_setup):
        empty_world = self.world_cls()
        assert len(empty_world.get_objects()) == 0
        assert not empty_world.has_robot()
        pr2 = self.cls(pr2_urdf())
        empty_world.add_robot(pr2, None, pr2.controlled_joints, 0, 0, False)
        assert empty_world.has_robot()
        assert pr2 == empty_world.robot
        return empty_world

    def test_add_object(self, function_setup):
        empty_world = self.world_cls()
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        empty_world.add_object(box)
        assert empty_world.has_object(name)
        assert len(empty_world.get_objects()) == 1
        assert len(empty_world.get_object_names()) == 1
        assert empty_world.get_object(box.get_name()) == box
        return empty_world

    def test_add_object_twice(self, function_setup):
        empty_world = self.world_cls()
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
        return empty_world

    def test_add_object_with_robot_name(self, function_setup):
        world_with_pr2 = self.make_world_with_pr2()
        name = u'pr2'
        box = self.cls.from_world_body(make_world_body_box(name))
        try:
            world_with_pr2.add_object(box)
            assert False, u'expected exception'
        except DuplicateNameException:
            assert True
        assert world_with_pr2.has_robot()
        assert len(world_with_pr2.get_objects()) == 0
        return world_with_pr2

    def test_attach_existing_obj_to_robot1(self, function_setup):
        world_with_pr2 = self.make_world_with_pr2()
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
        return world_with_pr2

    def test_attach_existing_obj_to_robot2(self, function_setup):
        world_with_pr2 = self.make_world_with_pr2()
        box = self.cls.from_world_body(make_world_body_box())
        world_with_pr2.add_object(box)
        p = Pose()
        p.orientation.w = 1
        try:
            world_with_pr2.attach_existing_obj_to_robot(u'box2', u'l_gripper_tool_frame', p)
            assert False
        except KeyError:
            assert True
        return world_with_pr2

    def test_attach_detach_existing_obj_to_robot1(self, function_setup):
        obj_name = u'box'
        world_with_pr2 = self.make_world_with_pr2()
        box = self.cls.from_world_body(make_world_body_box(name=obj_name))
        world_with_pr2.add_object(box)
        links_before = set(world_with_pr2.robot.get_link_names())
        joints_before = set(world_with_pr2.robot.get_joint_names())
        p = Pose()
        p.orientation.w = 1
        world_with_pr2.attach_existing_obj_to_robot(obj_name, u'l_gripper_tool_frame', p)
        assert obj_name not in world_with_pr2.get_object_names()
        assert set(world_with_pr2.robot.get_link_names()).difference(links_before) == {obj_name}
        assert set(world_with_pr2.robot.get_joint_names()).difference(joints_before) == {obj_name}

        world_with_pr2.detach(obj_name)
        assert set(world_with_pr2.robot.get_link_names()).symmetric_difference(links_before) == set()
        assert set(world_with_pr2.robot.get_joint_names()).symmetric_difference(joints_before) == set()
        assert obj_name in world_with_pr2.get_object_names()
        compare_poses(world_with_pr2.robot.get_fk(world_with_pr2.robot.get_root(), u'l_gripper_tool_frame').pose,
                      world_with_pr2.get_object(obj_name).base_pose)
        return world_with_pr2

    def test_hard_reset1(self, function_setup):
        world_with_pr2 = self.make_world_with_pr2()
        world_with_pr2.hard_reset()
        assert not world_with_pr2.has_robot()
        return world_with_pr2

    def test_hard_reset2(self, function_setup):
        world_with_pr2 = self.make_world_with_pr2()
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_pr2.add_object(box)
        world_with_pr2.hard_reset()
        assert not world_with_pr2.has_robot()
        assert len(world_with_pr2.get_objects()) == 0
        return world_with_pr2

    def test_soft_reset1(self, function_setup):
        world_with_pr2 = self.make_world_with_pr2()
        world_with_pr2.soft_reset()
        assert world_with_pr2.has_robot()
        return world_with_pr2

    def test_soft_reset2(self, function_setup):
        world_with_pr2 = self.make_world_with_pr2()
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_pr2.add_object(box)
        world_with_pr2.soft_reset()
        assert world_with_pr2.has_robot()
        assert len(world_with_pr2.get_objects()) == 0
        return world_with_pr2

    def test_remove_object1(self, function_setup):
        world_with_pr2 = self.make_world_with_pr2()
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_pr2.add_object(box)
        world_with_pr2.remove_object(name)
        assert not world_with_pr2.has_object(name)
        assert len(world_with_pr2.get_objects()) == 0
        assert world_with_pr2.has_robot()
        return world_with_pr2

    def test_remove_object2(self, function_setup):
        world_with_pr2 = self.make_world_with_pr2()
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
        return world_with_pr2

    def test_collision_goals_to_collision_matrix1(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], 0.05)
        assert len(collision_matrix) == 0
        return world_with_donbot

    def test_collision_goals_to_collision_matrix2(self, test_folder):
        min_dist = 0.05
        world_with_donbot = self.make_world_with_donbot(test_folder)
        base_collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], min_dist)
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], min_dist)
        assert len(collision_matrix) == len(base_collision_matrix) + len(world_with_donbot.robot.get_controlled_links())
        robot_link_names = world_with_donbot.robot.get_link_names()
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist
            if body_b == name:
                assert body_b_link == name
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix3(self, test_folder):
        min_dist = 0.05
        world_with_donbot = self.make_world_with_donbot(test_folder)
        base_collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], min_dist)
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_ALL_COLLISIONS
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)
        assert len(collision_matrix) == len(base_collision_matrix) + len(world_with_donbot.robot.get_controlled_links())
        robot_link_names = world_with_donbot.robot.get_link_names()
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == ce.min_dist
            if body_b == name:
                assert body_b_link == name
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix4(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
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
        return world_with_donbot

    def test_collision_goals_to_collision_matrix5(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
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
        ce.body_b = name
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, 0.05)

        assert len(collision_matrix) == 1
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == ce.min_dist
            assert body_b == name
            assert body_b_link == name
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix6(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())
        min_dist = 0.05

        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        allowed_link = robot_link_names[0]

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [allowed_link]
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)

        assert len([x for x in collision_matrix if x[0] == allowed_link]) == 0
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist
            if body_b == name:
                assert body_b_link == name
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix7(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
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

        assert len([x for x in collision_matrix if x[2] == name2]) == 0
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist
            if body_b == name:
                assert body_b_link == name
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix8(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        name2 = u'muh2'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())
        allowed_link = robot_link_names[0]
        min_dist = 0.05

        box = self.cls.from_world_body(make_world_body_box(name))
        box2 = self.cls.from_world_body(make_world_body_box(name2))
        world_with_donbot.add_object(box)
        world_with_donbot.add_object(box2)

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [allowed_link]
        ce.body_b = name2
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)

        assert len([x for x in collision_matrix if x[0] == allowed_link and x[2] == name2]) == 0
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist
            if body_b != world_with_donbot.robot.get_name():
                assert body_b == body_b_link
            assert robot_link in robot_link_names
            if body_b == name2:
                assert robot_link != robot_link_names[0]
        return world_with_donbot
