import shutil
import time
from collections import defaultdict
from itertools import product
from time import sleep

import pybullet as p
import pytest
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion

import giskardpy.model.pybullet_wrapper as pbw
from giskardpy.model.pybullet_syncer import PyBulletSyncer
from giskardpy.model.pybullet_world import PyBulletWorld
from giskardpy.model.pybullet_world_object import PyBulletWorldObject
from giskardpy.model.robot import Robot
from giskardpy.model.utils import make_world_body_box, make_world_body_sphere, make_world_body_cylinder
from giskardpy.utils.utils import logging
from giskardpy.model.world_object import WorldObject
from test_world import create_world_with_pr2
from utils_for_tests import pr2_urdf, base_bot_urdf, donbot_urdf

# this import has to come last
import test_world

folder_name = u'tmp_data/'


@pytest.fixture(scope=u'module')
def module_setup(request):
    logging.loginfo(u'starting pybullet')
    pbw.start_pybullet(False)

    logging.loginfo(u'deleting tmp test folder')
    try:
        shutil.rmtree(folder_name)
    except:
        pass

    def kill_pybullet():
        logging.loginfo(u'shutdown pybullet')
        pbw.stop_pybullet()

    request.addfinalizer(kill_pybullet)


@pytest.fixture()
def function_setup(request, module_setup):
    """
    :rtype: WorldObject
    """
    pbw.clear_pybullet()

    def kill_pybullet():
        logging.loginfo(u'resetting pybullet')
        pbw.clear_pybullet()

    request.addfinalizer(kill_pybullet)


@pytest.fixture()
def pr2_world(request, function_setup):
    """
    :rtype: World
    """

    world = create_world_with_pr2()
    pbs = PyBulletSyncer(world)
    pbs.sync()
    return pbs


@pytest.fixture()
def delete_test_folder(request):
    """
    :rtype: World
    """
    folder_name = u'tmp_data/'
    try:
        shutil.rmtree(folder_name)
    except:
        pass
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
    class cls(Robot):
        def __init__(self, urdf, base_pose=None, controlled_joints=None, path_to_data_folder=u'', *args, **kwargs):
            super().__init__(urdf, base_pose, controlled_joints, path_to_data_folder, *args, **kwargs)
            self.set_dummy_joint_symbols()

    def test_from_world_body_box(self, function_setup):
        wb = make_world_body_box()
        urdf_obj = self.cls.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert len(urdf_obj.get_joint_names()) == 0

    def test_from_world_body_sphere(self, function_setup):
        wb = make_world_body_sphere()
        urdf_obj = self.cls.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert len(urdf_obj.get_joint_names()) == 0

    def test_from_world_body_cylinder(self, function_setup):
        wb = make_world_body_cylinder()
        urdf_obj = self.cls.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert len(urdf_obj.get_joint_names()) == 0

    def test_safe_load_collision_matrix(self, test_folder, delete_test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        r.init_self_collision_matrix()
        expected = r.get_self_collision_matrix()
        r.safe_self_collision_matrix(test_folder)
        assert r.load_self_collision_matrix(test_folder)
        actual = r.get_self_collision_matrix()
        assert expected == actual

    def test_attach_urdf_object1_2(self, test_folder):
        parsed_pr2 = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        parsed_pr2.init_self_collision_matrix()
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
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        r.init_self_collision_matrix()
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
        r.init_self_collision_matrix()
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
        min_dist = defaultdict(lambda: 0.05)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], min_dist)
        assert len(collision_matrix) == 116
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

    def test_collision_goals_to_collision_matrix9(self, test_folder):
        return super(TestPyBulletWorld, self).test_collision_goals_to_collision_matrix9(test_folder)

    def test_verify_collision_entries_allow_all(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_allow_all(test_folder)

    def test_verify_collision_entries_cut_off1(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_cut_off1(test_folder)

    def test_verify_collision_entries_split5(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_split5(test_folder)

    def test_verify_collision_entries_allow_all_self(self, test_folder):
        super(TestPyBulletWorld, self).test_verify_collision_entries_allow_all_self(test_folder)

    def test_check_collisions(self, test_folder):
        w = self.make_world_with_pr2()
        pr22 = self.cls(pr2_urdf())
        pr22.set_name('pr22')
        w.add_object(pr22)
        base_pose = Pose()
        base_pose.position.x = 10
        base_pose.orientation.w = 1
        w.set_object_pose('pr22', base_pose)
        robot_links = pr22.get_link_names()
        cut_off_distances = {(link1, 'pr22', link2): 0.1 for link1, link2 in product(robot_links, repeat=2)}

        assert len(w.check_collisions(cut_off_distances).all_collisions) == 0

    def test_check_collisions2(self, test_folder):
        w = self.make_world_with_pr2()
        pr22 = self.cls(pr2_urdf())
        pr22.set_name('pr22')
        w.add_object(pr22)
        base_pose = Pose()
        base_pose.position.x = 0.05
        base_pose.orientation.w = 1
        w.set_object_pose('pr22', base_pose)

        pr23 = self.cls(pr2_urdf())
        pr23.set_name('pr23')
        w.add_object(pr23)
        base_pose = Pose()
        base_pose.position.y = 0.05
        base_pose.orientation.w = 1
        w.set_object_pose('pr23', base_pose)

        min_dist = defaultdict(lambda: 0.1)
        cut_off_distances = w.collision_goals_to_collision_matrix([], min_dist)

        for i in range(160):
            assert len(w.check_collisions(cut_off_distances).all_collisions) == 614

    def test_check_collisions3(self, test_folder):
        w = self.make_world_with_pr2()
        pr22 = self.cls(pr2_urdf())
        pr22.set_name('pr22')
        w.add_object(pr22)
        base_pose = Pose()
        base_pose.position.x = 1.5
        base_pose.orientation.w = 1
        w.set_object_pose('pr22', base_pose)
        min_dist = defaultdict(lambda: 0.1)
        cut_off_distances = w.collision_goals_to_collision_matrix([], min_dist)
        robot_links = pr22.get_link_names()
        cut_off_distances.update({(link1, 'pr22', link2): 0.1 for link1, link2 in product(robot_links, repeat=2) if w.robot.has_link_collision(link1)})

        for i in range(160):
            assert len(w.check_collisions(cut_off_distances).all_collisions) == 0

    # TODO test that has collision entries of robot links without collision geometry

    # TODO test that makes sure adding avoid specific self collisions works


class TestPyBulletSyncer(object):
    def test_load_pr2(self, pr2_world):
        assert len(pbw.get_body_names()) == 54

    def test_set_pr2_js(self, pr2_world):
        pr2_world.world.state['torso_lift_link'] = 1
        pr2_world.sync()
        assert len(pbw.get_body_names()) == 54

import pytest
pytest.main(['-s', __file__ + '::TestPyBulletSyncer::test_set_pr2_js'])