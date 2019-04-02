import pybullet as p
import shutil

import pytest
from geometry_msgs.msg import Pose, Point, Quaternion
from giskard_msgs.msg import CollisionEntry

from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.pybullet_world_object import PyBulletWorldObject
import giskardpy.pybullet_wrapper as pbw
from giskardpy.symengine_robot import Robot
from giskardpy.utils import make_world_body_box
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
def test_folder(request, function_setup):
    """
    :rtype: World
    """
    folder_name = u'tmp_data/'

    def kill_pybullet():
        try:
            print(u'deleting tmp test folder')
            shutil.rmtree(folder_name)
        except Exception:
            pass

    request.addfinalizer(kill_pybullet)
    return folder_name


def assert_num_pybullet_objects(num):
    assert p.getNumBodies() == num, pbw.print_body_names()


class TestPyBulletWorldObject(test_world.TestWorldObj):
    cls = PyBulletWorldObject

    def test_create_object(self, function_setup):
        parsed_base_bot = self.cls(base_bot_urdf())
        assert_num_pybullet_objects(1)
        assert u'pointy' in pbw.get_body_names()


class TestPyBulletRobot(test_world.TestRobot):
    cls = Robot

    def test_safe_load_collision_matrix(self, test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder, calc_self_collision_matrix=True)
        scm = r.get_self_collision_matrix()
        assert len(scm) == 53

    def test_attach_urdf_object1_2(self, test_folder):
        parsed_pr2 = self.cls(donbot_urdf(), path_to_data_folder=test_folder, calc_self_collision_matrix=True)
        scm = parsed_pr2.get_self_collision_matrix()
        num_of_links_before = len(parsed_pr2.get_link_names())
        num_of_joints_before = len(parsed_pr2.get_joint_names())
        link_chain_before = len(parsed_pr2.get_links_from_sub_tree(u'ur5_shoulder_pan_joint'))
        box = self.cls.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        parsed_pr2.attach_urdf_object(box, u'gripper_tool_frame', p)
        assert box.get_name() in parsed_pr2.get_link_names()
        assert len(parsed_pr2.get_link_names()) == num_of_links_before + 1
        assert len(parsed_pr2.get_joint_names()) == num_of_joints_before + 1
        assert len(parsed_pr2.get_links_from_sub_tree(u'ur5_shoulder_pan_joint')) == link_chain_before + 1
        assert scm.difference(parsed_pr2.get_self_collision_matrix()) == set()
        assert len(scm) < len(parsed_pr2.get_self_collision_matrix())

    def test_detach_object2(self, test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder, calc_self_collision_matrix=True)
        scm = r.get_self_collision_matrix()
        box = WorldObject.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        r.attach_urdf_object(box, u'gripper_tool_frame', p)
        assert len(scm) < len(r.get_self_collision_matrix())
        r.detach_sub_tree(box.get_name())
        assert scm.symmetric_difference(r.get_self_collision_matrix()) == set()

    def test_reset_collision_matrix(self, test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        r.update_self_collision_matrix()
        scm = r.get_self_collision_matrix()

        box = self.cls.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        r.attach_urdf_object(box, u'gripper_tool_frame', p)

        assert scm.symmetric_difference(r.get_self_collision_matrix()) != set()
        r.reset()
        assert scm.symmetric_difference(r.get_self_collision_matrix()) == set()


class TestPyBulletWorld(test_world.TestWorld):
    cls = WorldObject
    world_cls = PyBulletWorld

    def test_add_robot(self, function_setup):
        w = super(TestPyBulletWorld, self).test_add_robot(function_setup)
        assert_num_pybullet_objects(3)

    def test_add_object(self, function_setup):
        w = super(TestPyBulletWorld, self).test_add_object(function_setup)
        assert_num_pybullet_objects(3)

    def test_add_object_twice(self, function_setup):
        w = super(TestPyBulletWorld, self).test_add_object_twice(function_setup)
        assert_num_pybullet_objects(3)

    def test_add_object_with_robot_name(self, function_setup):
        w = super(TestPyBulletWorld, self).test_add_object_with_robot_name(function_setup)
        assert_num_pybullet_objects(3)

    def test_hard_reset1(self, function_setup):
        w = super(TestPyBulletWorld, self).test_hard_reset1(function_setup)
        assert_num_pybullet_objects(2)

    def test_hard_reset2(self, function_setup):
        w = super(TestPyBulletWorld, self).test_hard_reset2(function_setup)
        assert_num_pybullet_objects(2)

    def test_soft_reset1(self, function_setup):
        w = super(TestPyBulletWorld, self).test_soft_reset1(function_setup)
        assert_num_pybullet_objects(3)

    def test_soft_reset2(self, function_setup):
        w = super(TestPyBulletWorld, self).test_soft_reset2(function_setup)
        assert_num_pybullet_objects(3)

    def test_remove_object1(self, function_setup):
        w = super(TestPyBulletWorld, self).test_remove_object1(function_setup)
        assert_num_pybullet_objects(3)

    def test_remove_object2(self, function_setup):
        w = super(TestPyBulletWorld, self).test_remove_object2(function_setup)
        assert_num_pybullet_objects(4)

    def test_attach_existing_obj_to_robot(self, function_setup):
        w = super(TestPyBulletWorld, self).test_attach_existing_obj_to_robot1(function_setup)
        assert_num_pybullet_objects(3)

    def test_collision_goals_to_collision_matrix1(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], 0.05)
        assert len(collision_matrix) == 106
        return world_with_donbot

    def test_attach_detach_existing_obj_to_robot1(self, function_setup):
        w = super(TestPyBulletWorld, self).test_attach_detach_existing_obj_to_robot1(function_setup)
        assert_num_pybullet_objects(4)

    def test_verify_collision_entries_empty(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_empty(test_folder)

    def test_verify_collision_entries_split0(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_split0(test_folder)

    def test_verify_collision_entries_split1(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_split1(test_folder)

    def test_verify_collision_entries_split2(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_split2(test_folder)

    def test_verify_collision_entries_split3(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_split3(test_folder)

    def test_verify_collision_entries_split4(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_split4(test_folder)

    def test_collision_goals_to_collision_matrix3(self, test_folder):
        return super(TestPyBulletWorld, self).test_collision_goals_to_collision_matrix3(test_folder)

    def test_collision_goals_to_collision_matrix5(self, test_folder):
        return super(TestPyBulletWorld, self).test_collision_goals_to_collision_matrix5(test_folder)

    def test_collision_goals_to_collision_matrix6(self, test_folder):
        return super(TestPyBulletWorld, self).test_collision_goals_to_collision_matrix6(test_folder)

    def test_collision_goals_to_collision_matrix7(self, test_folder):
        return super(TestPyBulletWorld, self).test_collision_goals_to_collision_matrix7(test_folder)

    def test_collision_goals_to_collision_matrix8(self, test_folder):
        return super(TestPyBulletWorld, self).test_collision_goals_to_collision_matrix8(test_folder)

    # TODO test that has collision entries of robot links without collision geometry



