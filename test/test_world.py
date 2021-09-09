import shutil
from collections import defaultdict

import numpy as np
import pytest
import urdf_parser_py.urdf as up
from geometry_msgs.msg import Pose, Point, Quaternion
from giskard_msgs.msg import CollisionEntry
from hypothesis import given

import test_urdf_object
from giskardpy import identifier
from giskardpy.data_types import JointStates
from giskardpy.exceptions import DuplicateNameException, PhysicsWorldException, UnknownBodyException
from giskardpy.god_map import GodMap
from giskardpy.model.robot import Robot
from giskardpy.model.urdf_object import hacky_urdf_parser_fix
from giskardpy.model.utils import make_world_body_box
from giskardpy.model.world import World, WorldTree
from giskardpy.model.world_object import WorldObject
from giskardpy.utils.utils import suppress_stderr
from utils_for_tests import pr2_urdf, donbot_urdf, compare_poses, pr2_without_base_urdf, rnd_joint_state


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
    :rtype: str
    """
    folder_name = u'tmp_data/'

    def delete_test_folder():
        try:
            shutil.rmtree(folder_name)
        except OSError:
            print(u'couldn\'t delete test folder')

    request.addfinalizer(delete_test_folder)
    return folder_name


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

    def delete_test_folder():
        try:
            shutil.rmtree(folder_name)
        except OSError:
            print(u'couldn\'t delete test folder')

    request.addfinalizer(delete_test_folder)

    return folder_name


def allow_all_entry():
    ce = CollisionEntry()
    ce.type = CollisionEntry.ALLOW_COLLISION
    ce.robot_links = [CollisionEntry.ALL]
    ce.body_b = CollisionEntry.ALL
    ce.link_bs = [CollisionEntry.ALL]
    ce.min_dist = 0.0
    return ce


def avoid_all_entry(min_dist):
    ce = CollisionEntry()
    ce.type = CollisionEntry.AVOID_COLLISION
    ce.robot_links = [CollisionEntry.ALL]
    ce.body_b = CollisionEntry.ALL
    ce.link_bs = [CollisionEntry.ALL]
    ce.min_dist = min_dist
    return ce


def world_with_robot(urdf):
    god_map = GodMap()
    urdf = pr2_urdf()
    world = WorldTree(god_map)
    god_map.set_data(identifier.world, world)
    world.load_urdf(urdf)
    world.add_group('robot', 'odom_combined')
    return world


def world_with_pr2():
    return world_with_robot(pr2_urdf())


def all_joint_limits(urdf):
    world = world_with_robot(urdf)
    return world.get_all_joint_position_limits()


pr2_joint_limits = all_joint_limits(pr2_urdf())


class TestWorldTree(object):
    def parsed_pr2_urdf(self):
        urdf = pr2_urdf()
        with suppress_stderr():
            return up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))

    def test_load_pr2(self):
        world = world_with_pr2()
        parsed_urdf = self.parsed_pr2_urdf()
        assert set(world.link_names) == set(list(parsed_urdf.link_map.keys()) + [world.root_link_name])
        assert set(world.joint_names) == set(list(parsed_urdf.joint_map.keys()) + [parsed_urdf.name])

    def test_group_pr2_hand(self):
        world = world_with_pr2()
        world.add_group('r_hand', 'r_wrist_roll_link')
        assert world.groups['r_hand'].joint_names == ['r_gripper_palm_joint',
                                                      'r_gripper_led_joint',
                                                      'r_gripper_motor_accelerometer_joint',
                                                      'r_gripper_tool_joint',
                                                      'r_gripper_motor_slider_joint',
                                                      'r_gripper_l_finger_joint',
                                                      'r_gripper_r_finger_joint',
                                                      'r_gripper_motor_screw_joint',
                                                      'r_gripper_l_finger_tip_joint',
                                                      'r_gripper_r_finger_tip_joint',
                                                      'r_gripper_joint']
        assert world.groups['r_hand'].link_names == ['r_wrist_roll_link',
                                                     'r_gripper_palm_link',
                                                     'r_gripper_led_frame',
                                                     'r_gripper_motor_accelerometer_link',
                                                     'r_gripper_tool_frame',
                                                     'r_gripper_motor_slider_link',
                                                     'r_gripper_motor_screw_link',
                                                     'r_gripper_l_finger_link',
                                                     'r_gripper_l_finger_tip_link',
                                                     'r_gripper_r_finger_link',
                                                     'r_gripper_r_finger_tip_link',
                                                     'r_gripper_l_finger_tip_frame']

    def test_get_chain(self):
        world = world_with_pr2()
        parsed_urdf = self.parsed_pr2_urdf()
        root_link = parsed_urdf.get_root()
        tip_link = 'r_gripper_tool_frame'
        real = world.get_chain(root_link, tip_link)
        expected = parsed_urdf.get_chain(root_link, tip_link)
        assert set(real) == set(expected)

    def test_get_chain2(self):
        world = world_with_pr2()
        root_link = 'l_gripper_tool_frame'
        tip_link = 'r_gripper_tool_frame'
        try:
            world.get_chain(root_link, tip_link)
            assert False
        except ValueError:
            pass

    def test_get_chain_group(self):
        root_link = 'r_wrist_roll_link'
        tip_link = 'r_gripper_r_finger_tip_link'
        world = world_with_pr2()
        world.add_group('r_hand', root_link)
        real = world.groups['r_hand'].get_chain(root_link, tip_link)
        assert real == ['r_wrist_roll_link',
                        'r_gripper_palm_joint',
                        'r_gripper_palm_link',
                        'r_gripper_r_finger_joint',
                        'r_gripper_r_finger_link',
                        'r_gripper_r_finger_tip_joint',
                        'r_gripper_r_finger_tip_link']

    def test_get_chain_group2(self):
        root_link = 'r_gripper_l_finger_tip_link'
        tip_link = 'r_gripper_r_finger_tip_link'
        world = world_with_pr2()
        world.add_group('r_hand', 'r_wrist_roll_link')
        try:
            real = world.groups['r_hand'].get_chain(root_link, tip_link)
            assert False
        except ValueError:
            pass

    def test_get_split_chain(self):
        world = world_with_pr2()
        root_link = 'l_gripper_r_finger_tip_link'
        tip_link = 'l_gripper_l_finger_tip_link'
        chain1, connection, chain2 = world.get_split_chain(root_link, tip_link)
        assert chain1 == ['l_gripper_r_finger_tip_link', 'l_gripper_r_finger_tip_joint', 'l_gripper_r_finger_link',
                          'l_gripper_r_finger_joint']
        assert connection == ['l_gripper_palm_link']
        assert chain2 == ['l_gripper_l_finger_joint', 'l_gripper_l_finger_link', 'l_gripper_l_finger_tip_joint',
                          'l_gripper_l_finger_tip_link']

    def test_get_split_chain_group(self):
        root_link = 'r_gripper_l_finger_tip_link'
        tip_link = 'r_gripper_r_finger_tip_link'
        world = world_with_pr2()
        world.add_group('r_hand', 'r_wrist_roll_link')
        chain1, connection, chain2 = world.groups['r_hand'].get_split_chain(root_link, tip_link)
        assert chain1 == ['r_gripper_l_finger_tip_link',
                          'r_gripper_l_finger_tip_joint',
                          'r_gripper_l_finger_link',
                          'r_gripper_l_finger_joint']
        assert connection == ['r_gripper_palm_link']
        assert chain2 == ['r_gripper_r_finger_joint',
                          'r_gripper_r_finger_link',
                          'r_gripper_r_finger_tip_joint',
                          'r_gripper_r_finger_tip_link']

    def test_get_joint_limits2(self):
        world = world_with_pr2()
        lower_limit, upper_limit = world.get_joint_position_limits('l_shoulder_pan_joint')
        assert lower_limit == -0.564601836603
        assert upper_limit == 2.1353981634

    def test_get_all_joint_limits(self):
        world = world_with_pr2()
        assert world.get_all_joint_position_limits() == {'bl_caster_l_wheel_joint': (None, None),
                                                         'bl_caster_r_wheel_joint': (None, None),
                                                         'bl_caster_rotation_joint': (None, None),
                                                         'br_caster_l_wheel_joint': (None, None),
                                                         'br_caster_r_wheel_joint': (None, None),
                                                         'br_caster_rotation_joint': (None, None),
                                                         'fl_caster_l_wheel_joint': (None, None),
                                                         'fl_caster_r_wheel_joint': (None, None),
                                                         'fl_caster_rotation_joint': (None, None),
                                                         'fr_caster_l_wheel_joint': (None, None),
                                                         'fr_caster_r_wheel_joint': (None, None),
                                                         'fr_caster_rotation_joint': (None, None),
                                                         'head_pan_joint': (-2.857, 2.857),
                                                         'head_tilt_joint': (-0.3712, 1.29626),
                                                         'l_elbow_flex_joint': (-2.1213, -0.15),
                                                         'l_forearm_roll_joint': (None, None),
                                                         'l_gripper_joint': (0.0, 0.088),
                                                         'l_gripper_l_finger_joint': (0.0, 0.548),
                                                         'l_gripper_l_finger_tip_joint': (0.0, 0.548),
                                                         'l_gripper_motor_screw_joint': (None, None),
                                                         'l_gripper_motor_slider_joint': (-0.1, 0.1),
                                                         'l_gripper_r_finger_joint': (0.0, 0.548),
                                                         'l_gripper_r_finger_tip_joint': (0.0, 0.548),
                                                         'l_shoulder_lift_joint': (-0.3536, 1.2963),
                                                         'l_shoulder_pan_joint': (-0.564601836603, 2.1353981634),
                                                         'l_upper_arm_roll_joint': (-0.65, 3.75),
                                                         'l_wrist_flex_joint': (-2.0, -0.1),
                                                         'l_wrist_roll_joint': (None, None),
                                                         'laser_tilt_mount_joint': (-0.7354, 1.43353),
                                                         'odom_x_joint': (-1000.0, 1000.0),
                                                         'odom_y_joint': (-1000.0, 1000.0),
                                                         'odom_z_joint': (None, None),
                                                         'r_elbow_flex_joint': (-2.1213, -0.15),
                                                         'r_forearm_roll_joint': (None, None),
                                                         'r_gripper_joint': (0.0, 0.088),
                                                         'r_gripper_l_finger_joint': (0.0, 0.548),
                                                         'r_gripper_l_finger_tip_joint': (0.0, 0.548),
                                                         'r_gripper_motor_screw_joint': (None, None),
                                                         'r_gripper_motor_slider_joint': (-0.1, 0.1),
                                                         'r_gripper_r_finger_joint': (0.0, 0.548),
                                                         'r_gripper_r_finger_tip_joint': (0.0, 0.548),
                                                         'r_shoulder_lift_joint': (-0.3536, 1.2963),
                                                         'r_shoulder_pan_joint': (-2.1353981634, 0.564601836603),
                                                         'r_upper_arm_roll_joint': (-3.75, 0.65),
                                                         'r_wrist_flex_joint': (-2.0, -0.1),
                                                         'r_wrist_roll_joint': (None, None),
                                                         'torso_lift_joint': (0.0115, 0.325),
                                                         'torso_lift_motor_screw_joint': (None, None)}

    def test_get_all_joint_limits_group(self):
        world = world_with_pr2()
        world.add_group('r_hand', 'r_wrist_roll_link')
        assert world.groups['r_hand'].get_all_joint_position_limits() == {'r_gripper_joint': (0.0, 0.088),
                                                                          'r_gripper_l_finger_joint': (0.0, 0.548),
                                                                          'r_gripper_l_finger_tip_joint': (0.0, 0.548),
                                                                          'r_gripper_motor_screw_joint': (None, None),
                                                                          'r_gripper_motor_slider_joint': (-0.1, 0.1),
                                                                          'r_gripper_r_finger_joint': (0.0, 0.548),
                                                                          'r_gripper_r_finger_tip_joint': (0.0, 0.548)}

    @given(rnd_joint_state(pr2_joint_limits))
    def test_pr2_fk1(self, js):
        """
        :type js:
        """
        world = world_with_pr2()
        root = u'odom_combined'
        tips = [u'l_gripper_tool_frame', u'r_gripper_tool_frame']
        for tip in tips:
            mjs = JointStates()
            for joint_name, position in js.items():
                mjs[joint_name].position = position
            world.joint_state = mjs
            fk = world.get_fk_pose(root, tip).pose


class TestWorldObj(test_urdf_object.TestUrdfObject):
    cls = WorldObject

    def test_safe_load_collision_matrix(self, test_folder, delete_test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        r.init_self_collision_matrix()
        scm = r.get_self_collision_matrix()
        r.safe_self_collision_matrix(test_folder)
        r.load_self_collision_matrix(test_folder)
        assert scm == r.get_self_collision_matrix()

    def test_safe_load_collision_matrix2(self, test_folder, delete_test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        r.init_self_collision_matrix()
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
    class cls(Robot):
        def __init__(self, urdf, base_pose=None, controlled_joints=None, path_to_data_folder=u'', *args, **kwargs):
            super().__init__(urdf, base_pose, controlled_joints, path_to_data_folder, *args, **kwargs)
            self.set_dummy_joint_symbols()

    def test_safe_load_collision_matrix(self, test_folder, delete_test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        scm = r.get_self_collision_matrix()
        assert len(scm) == 0

    def test_get_controlled_leaf_joints(self, test_folder, delete_test_folder):
        r = self.cls(pr2_urdf(), path_to_data_folder=test_folder)
        r.controlled_joints = [u'torso_lift_joint',
                               u'r_upper_arm_roll_joint',
                               u'r_shoulder_pan_joint',
                               u'r_shoulder_lift_joint',
                               u'r_forearm_roll_joint',
                               u'r_elbow_flex_joint',
                               u'r_wrist_flex_joint',
                               u'r_wrist_roll_joint',
                               u'l_upper_arm_roll_joint',
                               u'l_shoulder_pan_joint',
                               u'l_shoulder_lift_joint',
                               u'l_forearm_roll_joint',
                               u'l_elbow_flex_joint',
                               u'l_wrist_flex_joint',
                               u'l_wrist_roll_joint',
                               u'head_pan_joint',
                               u'head_tilt_joint',
                               u'odom_x_joint',
                               u'odom_y_joint',
                               u'odom_z_joint']

        r = r.get_controlled_leaf_joints()
        assert r == {
            'l_wrist_roll_joint', 'r_wrist_roll_joint', 'odom_z_joint', 'l_forearm_roll_joint', 'torso_lift_joint',
            'head_tilt_joint', 'r_forearm_roll_joint'
        }

    def test_get_controlled_leaf_joints2(self, test_folder, delete_test_folder):
        r = self.cls(donbot_urdf(), path_to_data_folder=test_folder)
        r.controlled_joints = [u'ur5_shoulder_pan_joint',
                               u'ur5_shoulder_lift_joint',
                               u'ur5_elbow_joint',
                               u'ur5_wrist_1_joint',
                               u'ur5_wrist_2_joint',
                               u'ur5_wrist_3_joint',
                               u'odom_x_joint',
                               u'odom_y_joint',
                               u'odom_z_joint']

        r = r.get_controlled_leaf_joints()
        assert r == {
            'odom_z_joint', 'ur5_wrist_3_joint'
        }

    def test_get_directly_controllable_collision_links(self, test_folder, delete_test_folder):
        r = self.cls(pr2_urdf(), path_to_data_folder=test_folder)
        r.controlled_joints = [u'torso_lift_joint',
                               u'r_upper_arm_roll_joint',
                               u'r_shoulder_pan_joint',
                               u'r_shoulder_lift_joint',
                               u'r_forearm_roll_joint',
                               u'r_elbow_flex_joint',
                               u'r_wrist_flex_joint',
                               u'r_wrist_roll_joint',
                               u'l_upper_arm_roll_joint',
                               u'l_shoulder_pan_joint',
                               u'l_shoulder_lift_joint',
                               u'l_forearm_roll_joint',
                               u'l_elbow_flex_joint',
                               u'l_wrist_flex_joint',
                               u'l_wrist_roll_joint',
                               u'head_pan_joint',
                               u'head_tilt_joint',
                               u'odom_x_joint',
                               u'odom_y_joint',
                               u'odom_z_joint']

        result = r.get_directly_controllable_collision_links(u'odom_x_joint')
        assert result == []
        result = r.get_directly_controllable_collision_links(u'odom_y_joint')
        assert result == []
        result = r.get_directly_controllable_collision_links(u'odom_z_joint')
        assert result == [u'base_link']
        result = r.get_directly_controllable_collision_links(u'l_elbow_flex_joint')
        assert result == [u'l_elbow_flex_link']
        result = r.get_directly_controllable_collision_links(u'r_wrist_roll_joint')
        assert result == [u'r_wrist_roll_link']
        result = r.get_directly_controllable_collision_links(u'br_caster_l_wheel_joint')
        assert result == []


class TestWorld(object):
    cls = WorldObject
    world_cls = World

    def make_world_with_robot(self, urdf, path_to_data_folder):
        w = self.world_cls(path_to_data_folder=path_to_data_folder)
        r = self.cls(urdf)
        w.add_robot(robot=r,
                    base_pose=None,
                    controlled_joints=r.controlled_joints,
                    ignored_pairs=[],
                    added_pairs=[])
        w.robot.set_dummy_joint_symbols()
        w.robot.reinitialize()
        if path_to_data_folder is not None:
            w.robot.init_self_collision_matrix()
        return w

    def make_world_with_pr2(self, path_to_data_folder=None):
        """
        :rtype: World
        """
        return self.make_world_with_robot(pr2_urdf(), path_to_data_folder)

    def make_world_with_pr2_without_base(self, path_to_data_folder=None):
        """
        :rtype: World
        """
        return self.make_world_with_robot(pr2_without_base_urdf(), path_to_data_folder)

    def make_world_with_donbot(self, path_to_data_folder=None):
        """
        :rtype: World
        """
        return self.make_world_with_robot(donbot_urdf(), path_to_data_folder)

    def test_add_robot(self, function_setup):
        empty_world = self.world_cls()
        assert len(empty_world.get_objects()) == 0
        assert not empty_world.has_robot()
        # extracting the urdf turns integers into floats
        pr2 = self.cls(self.cls(pr2_urdf()).get_urdf_str())
        empty_world.add_robot(robot=pr2,
                              base_pose=None,
                              controlled_joints=pr2.controlled_joints,
                              ignored_pairs=[],
                              added_pairs=[])
        assert empty_world.has_robot()
        assert pr2.get_urdf_str() == empty_world.robot.get_urdf_str()
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
        compare_poses(world_with_pr2.robot.get_fk_pose(world_with_pr2.robot.get_root(), u'l_gripper_tool_frame').pose,
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

    def test_verify_collision_entries_empty(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        ces = []
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == 1

    def test_verify_collision_entries_allow_all(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        ces = [allow_all_entry()]
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == 0

    def test_verify_collision_entries_allow_all_self(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [CollisionEntry.ALL]
        ce.body_b = world_with_donbot.robot.get_name()
        ce.link_bs = [CollisionEntry.ALL]
        ces = [ce]
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == 1 + len(world_with_donbot.robot.get_self_collision_matrix()) * 2

    def test_verify_collision_entries_unknown_robot_link(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [u'muh']
        ce.min_dist = min_dist
        ces.append(ce)
        try:
            new_ces = world_with_donbot.verify_collision_entries(ces)
        except UnknownBodyException:
            assert True
        else:
            assert False, u'expected exception'

    def test_verify_collision_entries_unknown_body_b(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [CollisionEntry.ALL]
        ce.body_b = u'muh'
        ce.link_bs = [CollisionEntry.ALL]
        ce.min_dist = min_dist
        ces.append(ce)
        try:
            new_ces = world_with_donbot.verify_collision_entries(ces)
        except UnknownBodyException:
            assert True
        else:
            assert False, u'expected exception'

    def test_verify_collision_entries_unknown_link_b(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [CollisionEntry.ALL]
        ce.body_b = u'muh'
        ce.link_bs = [u'muh']
        ce.min_dist = min_dist
        ces.append(ce)
        try:
            new_ces = world_with_donbot.verify_collision_entries(ces)
        except UnknownBodyException:
            assert True
        else:
            assert False, u'expected exception'

    def test_verify_collision_entries_unknown_link_b2(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [CollisionEntry.ALL]
        ce.body_b = world_with_donbot.robot.get_name()
        ce.link_bs = [u'muh']
        ce.min_dist = min_dist
        ces.append(ce)
        try:
            new_ces = world_with_donbot.verify_collision_entries(ces)
        except UnknownBodyException:
            assert True
        else:
            assert False, u'expected exception'

    def test_verify_collision_entries1(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [CollisionEntry.ALL, u'plate']
        ce.min_dist = min_dist
        ces.append(ce)
        try:
            new_ces = world_with_donbot.verify_collision_entries(ces)
        except PhysicsWorldException:
            assert True
        else:
            assert False, u'expected exception'

    def test_verify_collision_entries2(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.link_bs = [CollisionEntry.ALL, u'muh']
        ce.min_dist = min_dist
        ces.append(ce)
        try:
            new_ces = world_with_donbot.verify_collision_entries(ces)
        except PhysicsWorldException:
            assert True
        else:
            assert False, u'expected exception'

    def test_verify_collision_entries3(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.link_bs = [CollisionEntry.ALL, u'muh']
        ce.robot_links = [CollisionEntry.ALL, u'muh']
        ce.min_dist = min_dist
        ces.append(ce)
        try:
            new_ces = world_with_donbot.verify_collision_entries(ces)
        except PhysicsWorldException:
            assert True
        else:
            assert False, u'expected exception'

    def test_verify_collision_entries3_1(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = CollisionEntry.ALL
        ce.link_bs = [u'muh']
        ce.min_dist = min_dist
        ces.append(ce)
        try:
            new_ces = world_with_donbot.verify_collision_entries(ces)
        except PhysicsWorldException:
            assert True
        else:
            assert False, u'expected exception'

    def test_verify_collision_entries_cut_off1(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = []
        ces.append(avoid_all_entry(min_dist))
        ces.append(allow_all_entry())
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == 0

    def test_verify_collision_entries_split0(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.1
        ces = [avoid_all_entry(min_dist)]
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == 1
        for ce in new_ces:
            assert ce.body_b == world_with_donbot.robot.get_name()
            assert ce.body_b != CollisionEntry.ALL
            assert world_with_donbot.all_robot_links(ce)
            assert world_with_donbot.all_link_bs(ce)
            assert ce.type == CollisionEntry. \
                AVOID_COLLISION

    def test_verify_collision_entries_split1(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.05
        ces = []
        ce1 = CollisionEntry()
        ce1.type = CollisionEntry.AVOID_COLLISION
        ce1.robot_links = [CollisionEntry.ALL]
        ce1.body_b = CollisionEntry.ALL
        ce1.link_bs = [CollisionEntry.ALL]
        ce1.min_dist = 0.1
        ces.append(ce1)
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [u'plate']
        ce.body_b = CollisionEntry.ALL
        ce.link_bs = [CollisionEntry.ALL]
        ces.append(ce)
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == 1 + \
               len(world_with_donbot.robot.get_possible_collisions(u'plate'))
        assert world_with_donbot.all_robot_links(new_ces[0])
        assert world_with_donbot.all_link_bs(new_ces[0])
        for ce in new_ces[1:]:
            assert ce.body_b == world_with_donbot.robot.get_name()
            assert ce.body_b != CollisionEntry.ALL
            assert CollisionEntry.ALL not in ce.robot_links
        i = 0
        for i in range(1):
            ce = new_ces[i]
            assert ce.type == CollisionEntry.AVOID_COLLISION
        i += 1
        for j in range(len(world_with_donbot.robot.get_possible_collisions(u'plate'))):
            ce = new_ces[i + j]
            assert ce.type == CollisionEntry.ALLOW_COLLISION

    def test_verify_collision_entries_split2(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        min_dist = 0.05
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        ces = [avoid_all_entry(min_dist)]
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [CollisionEntry.ALL]
        ce.body_b = name
        ce.link_bs = [CollisionEntry.ALL]
        ces.append(ce)
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == 1 + len(world_with_donbot.robot.get_controlled_links()) * 2
        for ce in new_ces[1:]:
            assert ce.body_b != CollisionEntry.ALL
            assert CollisionEntry.ALL not in ce.robot_links
            if ce.body_b != world_with_donbot.robot.get_name():
                assert CollisionEntry.ALL in ce.link_bs
            else:
                assert CollisionEntry.ALL not in ce.link_bs
            assert len(ce.link_bs) == 1
        i = 0
        for i in range(len(world_with_donbot.robot.get_controlled_links()) + 1):
            ce = new_ces[i]
            assert ce.type == CollisionEntry.AVOID_COLLISION
        i += 1
        for j in range(len(world_with_donbot.robot.get_controlled_links())):
            ce = new_ces[i + j]
            assert ce.type == CollisionEntry.ALLOW_COLLISION

    def test_verify_collision_entries_split3(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        min_dist = 0.05
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        ces = []
        ce1 = CollisionEntry()
        ce1.type = CollisionEntry.AVOID_COLLISION
        ce1.robot_links = [CollisionEntry.ALL]
        ce1.link_bs = [CollisionEntry.ALL]
        ce1.min_dist = min_dist
        ces.append(ce1)
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [CollisionEntry.ALL]
        ce.body_b = name
        ce.link_bs = [name]
        ces.append(ce)
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == len(world_with_donbot.robot.get_controlled_links()) * 2 + 1
        for ce in new_ces[1:]:
            assert ce.body_b != CollisionEntry.ALL
            assert CollisionEntry.ALL not in ce.robot_links
            assert CollisionEntry.ALL not in ce.link_bs
            assert len(ce.link_bs) == 1
        i = 0
        for i in range(1 +
                       len(world_with_donbot.robot.get_controlled_links())):
            ce = new_ces[i]
            assert ce.type == CollisionEntry.AVOID_COLLISION
        i += 1
        for j in range(len(world_with_donbot.robot.get_controlled_links())):
            ce = new_ces[i + j]
            assert ce.type == CollisionEntry.ALLOW_COLLISION

    def test_verify_collision_entries_split4(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        min_dist = 0.05
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)
        name2 = u'box2'
        box2 = self.cls.from_world_body(make_world_body_box(name2))
        world_with_donbot.add_object(box2)

        ces = []
        ce1 = CollisionEntry()
        ce1.type = CollisionEntry.AVOID_COLLISION
        ce1.robot_links = [CollisionEntry.ALL]
        ce1.link_bs = [CollisionEntry.ALL]
        ce1.min_dist = min_dist
        ces.append(ce1)
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [CollisionEntry.ALL]
        ce.body_b = name
        ce.link_bs = [name]
        ces.append(ce)
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == len(world_with_donbot.robot.get_controlled_links()) * 3 + 1
        for ce in new_ces[1:]:
            assert ce.body_b != CollisionEntry.ALL
            assert CollisionEntry.ALL not in ce.robot_links
            if ce.body_b == name2:
                assert CollisionEntry.ALL in ce.link_bs
            else:
                assert CollisionEntry.ALL not in ce.link_bs
            assert len(ce.link_bs) == 1
        i = -1
        for i in range(1 + len(world_with_donbot.robot.get_controlled_links()) * 2):
            ce = new_ces[i]
            assert ce.type == CollisionEntry.AVOID_COLLISION
        i += 1
        for j in range(len(world_with_donbot.robot.get_controlled_links())):
            ce = new_ces[i + j]
            assert ce.type == CollisionEntry.ALLOW_COLLISION

    def test_verify_collision_entries_split5(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        min_dist = 0.05
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        ces = [allow_all_entry()]
        ce1 = CollisionEntry()
        ce1.type = CollisionEntry.AVOID_COLLISION
        ce1.robot_links = [u'plate', u'base_link']
        ce1.body_b = name
        ce1.link_bs = [CollisionEntry.ALL]
        ce1.min_dist = min_dist
        ces.append(ce1)
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == 2

        for j in range(2):
            ce = new_ces[j]
            assert ce.type == CollisionEntry.AVOID_COLLISION

    def test_verify_collision_entries_split6(self, test_folder):
        world_with_donbot = self.make_world_with_donbot(test_folder)
        min_dist = 0.05
        ces = []
        ce1 = CollisionEntry()
        ce1.type = CollisionEntry.ALLOW_COLLISION
        ce1.robot_links = [u'plate', u'base_link']
        ce1.body_b = world_with_donbot.robot.get_name()
        ce1.link_bs = [u'gripper_finger_left_link', u'gripper_finger_right_link']
        ce1.min_dist = min_dist
        ces.append(ce1)
        new_ces = world_with_donbot.verify_collision_entries(ces)
        assert len(new_ces) == 4 + 1
        i = -1
        for i in range(1):
            ce = new_ces[i]
            assert ce.type == CollisionEntry.AVOID_COLLISION
        i += 1
        for j in range(4):
            ce = new_ces[i + j]
            assert ce.type == CollisionEntry.ALLOW_COLLISION

    def test_collision_goals_to_collision_matrix1(self, delete_test_folder):
        """
        test with no collision entries which is equal to avoid all collisions
        collision matrix should be empty, because world has no collision checker
        :param test_folder:
        :return:
        """
        world_with_donbot = self.make_world_with_donbot(delete_test_folder)
        min_dist = defaultdict(lambda: 0.05)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], min_dist)
        assert len(collision_matrix) == 0
        return world_with_donbot

    def test_collision_goals_to_collision_matrix2(self, test_folder):
        """
        avoid all with an added object should enlarge the collision matrix
        :param test_folder:
        :return:
        """
        min_dist = defaultdict(lambda: 0.05)
        world_with_donbot = self.make_world_with_donbot(test_folder)
        base_collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], min_dist)
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], min_dist)
        assert len(collision_matrix) == len(base_collision_matrix) + len(world_with_donbot.robot.get_controlled_links())
        robot_link_names = world_with_donbot.robot.get_link_names()
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist[robot_link]
            if body_b == name:
                assert body_b_link == u''
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix3(self, test_folder):
        """
        empty list should have the same effect than avoid all entry
        :param test_folder:
        :return:
        """
        min_dist = defaultdict(lambda: 0.05)
        world_with_donbot = self.make_world_with_donbot(test_folder)
        base_collision_matrix = world_with_donbot.collision_goals_to_collision_matrix([], min_dist)
        name = u'muh'
        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)
        ces = []
        ces.append(avoid_all_entry(min_dist))
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)
        assert len(collision_matrix) == len(base_collision_matrix) + len(world_with_donbot.robot.get_controlled_links())
        robot_link_names = world_with_donbot.robot.get_link_names()
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist[robot_link]
            if body_b == name:
                assert body_b_link == u''
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix4(self, test_folder):
        """
        allow all should lead to an empty collision matrix
        :param test_folder:
        :return:
        """
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'

        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        ces = []
        ces.append(allow_all_entry())
        ces.append(allow_all_entry())
        min_dist = defaultdict(lambda: 0.05)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)

        assert len(collision_matrix) == 0
        return world_with_donbot

    def test_collision_goals_to_collision_matrix5(self, test_folder):
        """

        :param test_folder:
        :return:
        """
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())

        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        ces = []
        ces.append(allow_all_entry())
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [robot_link_names[0]]
        ce.body_b = name
        ce.min_dist = 0.1
        ces.append(ce)
        min_dist = defaultdict(lambda: 0.1)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)

        assert len(collision_matrix) == 1
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == ce.min_dist
            assert body_b == name
            assert body_b_link == u''
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix6(self, test_folder):
        """
        allow collision with a specific object
        :param test_folder:
        :return:
        """
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())
        min_dist = defaultdict(lambda: 0.1)

        box = self.cls.from_world_body(make_world_body_box(name))
        world_with_donbot.add_object(box)

        allowed_link = robot_link_names[0]

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [allowed_link]
        ce.link_bs = [CollisionEntry.ALL]
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = world_with_donbot.collision_goals_to_collision_matrix(ces, min_dist)

        assert len([x for x in collision_matrix if x[0] == allowed_link]) == 0
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist[robot_link]
            if body_b == name:
                assert body_b_link == u''
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix7(self, test_folder):
        """
        allow collision with specific object
        :param test_folder:
        :return:
        """
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        name2 = u'muh2'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())
        min_dist = defaultdict(lambda: 0.05)

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
            assert dist == min_dist[robot_link]
            if body_b == name:
                assert body_b_link == u''
            assert robot_link in robot_link_names
        return world_with_donbot

    def test_collision_goals_to_collision_matrix8(self, test_folder):
        """
        allow collision between specific object and link
        :param test_folder:
        :return:
        """
        world_with_donbot = self.make_world_with_donbot(test_folder)
        name = u'muh'
        name2 = u'muh2'
        robot_link_names = list(world_with_donbot.robot.get_controlled_links())
        allowed_link = robot_link_names[0]
        min_dist = defaultdict(lambda: 0.05)

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
            assert dist == min_dist[robot_link]
            if body_b != world_with_donbot.robot.get_name():
                assert body_b_link == u''
            assert robot_link in robot_link_names
            if body_b == name2:
                assert robot_link != robot_link_names[0]
        return world_with_donbot

    def test_collision_goals_to_collision_matrix9(self, test_folder):
        """
        allow self collision
        :param test_folder:
        :return:
        """
        world_with_pr2 = self.make_world_with_pr2(test_folder)
        min_dist = defaultdict(lambda: 0.05)
        ces = []
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.robot_links = [u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link',
                                       u'l_gripper_l_finger_link', u'l_gripper_r_finger_link',
                                       u'l_gripper_r_finger_link', u'l_gripper_palm_link']
        collision_entry.body_b = world_with_pr2.robot.get_name()
        collision_entry.link_bs = [u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_forearm_roll_link',
                                   u'r_forearm_link', u'r_forearm_link']
        ces.append(collision_entry)

        collision_matrix = world_with_pr2.collision_goals_to_collision_matrix(ces, min_dist)

        # assert len(collision_matrix) == 0
        # assert len([x for x in collision_matrix if x[0] == allowed_link and x[2] == name2]) == 0
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert not (robot_link in collision_entry.robot_links and body_b_link in collision_entry.link_bs)
            assert not (body_b_link in collision_entry.robot_links and robot_link in collision_entry.link_bs)
        #     assert dist == min_dist
        #     if body_b != world_with_donbot.robot.get_name():
        #         assert body_b_link == u''
        #     assert robot_link in robot_link_names
        #     if body_b == name2:
        #         assert robot_link != robot_link_names[0]
        return world_with_pr2

    def test_collision_goals_to_collision_matrix10(self, test_folder):
        """
        avoid self collision with only specific links
        :param test_folder:
        :return:
        """
        world_with_pr2 = self.make_world_with_pr2_without_base(test_folder)
        min_dist = defaultdict(lambda: 0.05)
        ces = [allow_all_entry()]
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.robot_links = [u'base_link']
        collision_entry.body_b = world_with_pr2.robot.get_name()
        collision_entry.link_bs = [u'r_wrist_flex_link']
        ces.append(collision_entry)

        collision_matrix = world_with_pr2.collision_goals_to_collision_matrix(ces, min_dist)

        assert collision_matrix == {(u'base_link', u'pr2', u'r_wrist_flex_link'): 0.05}

        return world_with_pr2
