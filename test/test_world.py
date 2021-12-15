import shutil
from itertools import combinations

import pytest
import urdf_parser_py.urdf as up
from geometry_msgs.msg import Pose
from hypothesis import given

from giskard_msgs.msg import CollisionEntry
from giskardpy import identifier, RobotName, RobotPrefix
from giskardpy.data_types import JointStates, PrefixName
from giskardpy.exceptions import DuplicateNameException
from giskardpy.god_map import GodMap
from giskardpy.model.utils import make_world_body_box, hacky_urdf_parser_fix
from giskardpy.model.world import WorldTree
from giskardpy.utils.config_loader import ros_load_robot_config
from giskardpy.utils.utils import suppress_stderr
from utils_for_tests import pr2_urdf, donbot_urdf, compare_poses, rnd_joint_state, hsr_urdf


@pytest.fixture(scope='module')
def module_setup(request):
    pass


@pytest.fixture()
def function_setup(request, module_setup):
    pass


@pytest.fixture()
def test_folder(request):
    """
    :rtype: str
    """
    folder_name = 'tmp_data/'

    def delete_test_folder():
        try:
            shutil.rmtree(folder_name)
        except OSError:
            print('couldn\'t delete test folder')

    request.addfinalizer(delete_test_folder)
    return folder_name


@pytest.fixture()
def delete_test_folder(request):
    """
    :rtype: World
    """
    folder_name = 'tmp_data/'
    try:
        shutil.rmtree(folder_name)
    except:
        pass

    def delete_test_folder():
        try:
            shutil.rmtree(folder_name)
        except OSError:
            print('couldn\'t delete test folder')

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


def world_with_robot(urdf, prefix):
    god_map = GodMap()
    god_map.set_data(identifier.rosparam, ros_load_robot_config('package://giskardpy/config/default.yaml'))
    world = WorldTree(god_map)
    god_map.set_data(identifier.world, world)
    world.add_urdf(urdf, prefix=prefix, group_name=RobotName)
    return world


def create_world_with_pr2(prefix=None):
    """
    :rtype: WorldTree
    """
    world = world_with_robot(pr2_urdf(), prefix=prefix)
    world.god_map.set_data(identifier.controlled_joints, ['torso_lift_joint',
                                                          'r_upper_arm_roll_joint',
                                                          'r_shoulder_pan_joint',
                                                          'r_shoulder_lift_joint',
                                                          'r_forearm_roll_joint',
                                                          'r_elbow_flex_joint',
                                                          'r_wrist_flex_joint',
                                                          'r_wrist_roll_joint',
                                                          'l_upper_arm_roll_joint',
                                                          'l_shoulder_pan_joint',
                                                          'l_shoulder_lift_joint',
                                                          'l_forearm_roll_joint',
                                                          'l_elbow_flex_joint',
                                                          'l_wrist_flex_joint',
                                                          'l_wrist_roll_joint',
                                                          'head_pan_joint',
                                                          'head_tilt_joint',
                                                          'odom_x_joint',
                                                          'odom_y_joint',
                                                          'odom_z_joint'])
    return world


def create_world_with_donbot(prefix=None):
    world = world_with_robot(donbot_urdf(), prefix=prefix)
    world.god_map.set_data(identifier.controlled_joints, ['ur5_elbow_joint',
                                                          'ur5_shoulder_lift_joint',
                                                          'ur5_shoulder_pan_joint',
                                                          'ur5_wrist_1_joint',
                                                          'ur5_wrist_2_joint',
                                                          'ur5_wrist_3_joint',
                                                          'odom_x_joint',
                                                          'odom_y_joint',
                                                          'odom_z_joint'])
    return world


def create_world_with_hsr(prefix=None):
    return world_with_robot(hsr_urdf(), prefix=prefix)


def all_joint_limits(urdf):
    world = world_with_robot(urdf, None)
    return world.get_all_joint_position_limits()


pr2_joint_limits = all_joint_limits(pr2_urdf())


class TestWorldTree(object):
    def parsed_pr2_urdf(self):
        """
        :rtype: urdf_parser_py.urdf.Robot
        """
        urdf = pr2_urdf()
        with suppress_stderr():
            return up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))

    def parsed_hsr_urdf(self):
        urdf = hsr_urdf()
        with suppress_stderr():
            return up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))

    def test_link_urdf_str(self):
        world = create_world_with_pr2()
        world.links['base_footprint'].as_urdf()

    def test_load_pr2(self):
        world = create_world_with_pr2()
        parsed_urdf = self.parsed_pr2_urdf()
        assert set(world.link_names) == set(list(parsed_urdf.link_map.keys()) + [world.root_link_name.short_name])
        assert set(world.joint_names) == set(
            list(parsed_urdf.joint_map.keys()) + [PrefixName(parsed_urdf.name, world.connection_prefix)])

    def test_load_pr2_twice(self):
        world = create_world_with_pr2()
        pr22_name = 'pr22'
        try:
            world.add_urdf(pr2_urdf(), group_name=RobotName)
            assert False
        except DuplicateNameException:
            pass
        try:
            world.add_urdf(pr2_urdf(), group_name=RobotName, prefix=pr22_name)
            assert False
        except DuplicateNameException:
            pass
        world.add_urdf(pr2_urdf(), group_name='pr22', prefix=pr22_name)
        pr21 = world.groups[RobotName]
        pr22 = world.groups[pr22_name]
        for link_name in pr22.link_names:
            assert link_name.short_name in pr21.link_names
        for joint_name in pr22.joint_names:
            assert joint_name.short_name in pr21.joint_names
        assert len(world.links) == len(pr21.links) + len(pr22.links) + 1

    def test_add_box(self):
        world = create_world_with_pr2()
        box = make_world_body_box()
        box_name = box.name
        pose = Pose()
        pose.orientation.w = 1
        world.add_world_body(box, pose)
        assert box.name in world.groups
        assert box_name in world.links
        assert PrefixName(box_name, world.connection_prefix) in world.joints

    def test_attach_box(self):
        world = create_world_with_pr2()
        box = make_world_body_box()
        box_name = PrefixName(box.name, None)
        pose = Pose()
        pose.orientation.w = 1
        world.add_world_body(box, pose)
        new_parent_link_name = PrefixName('r_gripper_tool_frame', RobotPrefix)
        old_fk = world.compute_fk_pose(world.root_link_name, box_name)

        world.move_group(box.name, new_parent_link_name)

        new_fk = world.compute_fk_pose(world.root_link_name, box_name)
        assert box_name in world.groups[RobotName].link_names
        assert world.joints[world.links[box_name].parent_joint_name].parent_link_name == new_parent_link_name
        compare_poses(old_fk.pose, new_fk.pose)

        assert box_name in world.groups[RobotName].groups
        assert RobotName not in world.groups[RobotName].groups
        assert box_name not in world.minimal_group_names

    def test_load_hsr(self):
        world = create_world_with_hsr()
        parsed_urdf = self.parsed_hsr_urdf()
        assert set(world.link_names) == set(list(parsed_urdf.link_map.keys()) + [world.root_link_name.short_name])
        assert set(world.joint_names) == set(
            list(parsed_urdf.joint_map.keys()) + [PrefixName(parsed_urdf.name, world.connection_prefix)])

    def test_group_pr2_hand(self):
        world = create_world_with_pr2()
        world.register_group('r_hand', 'r_wrist_roll_link')
        assert set(world.groups['r_hand'].joint_names) == {'r_gripper_palm_joint',
                                                           'r_gripper_led_joint',
                                                           'r_gripper_motor_accelerometer_joint',
                                                           'r_gripper_tool_joint',
                                                           'r_gripper_motor_slider_joint',
                                                           'r_gripper_l_finger_joint',
                                                           'r_gripper_r_finger_joint',
                                                           'r_gripper_motor_screw_joint',
                                                           'r_gripper_l_finger_tip_joint',
                                                           'r_gripper_r_finger_tip_joint',
                                                           'r_gripper_joint'}
        assert set(world.groups['r_hand'].link_names) == {'r_wrist_roll_link',
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
                                                          'r_gripper_l_finger_tip_frame'}

    def test_get_chain(self):
        world = create_world_with_pr2()
        parsed_urdf = self.parsed_pr2_urdf()
        root_link = parsed_urdf.get_root()
        tip_link = 'r_gripper_tool_frame'
        real = world.compute_chain(root_link, tip_link, True, True, True, True)
        expected = parsed_urdf.get_chain(root_link, tip_link, True, True, True)
        assert set(real) == set(expected)

    def test_get_chain2(self):
        world = create_world_with_pr2()
        root_link = 'l_gripper_tool_frame'
        tip_link = 'r_gripper_tool_frame'
        try:
            world.compute_chain(root_link, tip_link, True, True, True, True)
            assert False
        except ValueError:
            pass

    def test_get_chain_group(self):
        root_link = 'r_wrist_roll_link'
        tip_link = 'r_gripper_r_finger_tip_link'
        world = create_world_with_pr2()
        world.register_group('r_hand', root_link)
        real = world.groups['r_hand'].compute_chain(root_link, tip_link, True, True, True, True)
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
        world = create_world_with_pr2()
        world.register_group('r_hand', 'r_wrist_roll_link')
        try:
            real = world.groups['r_hand'].compute_chain(root_link, tip_link, True, True, True, True)
            assert False
        except ValueError:
            pass

    def test_get_split_chain(self):
        world = create_world_with_pr2()
        root_link = PrefixName('l_gripper_r_finger_tip_link', None)
        tip_link = PrefixName('l_gripper_l_finger_tip_link', None)
        chain1, connection, chain2 = world.compute_split_chain(root_link, tip_link, True, True, True, True)
        chain1 = [n.short_name for n in chain1]
        connection = [n.short_name for n in connection]
        chain2 = [n.short_name for n in chain2]
        assert chain1 == ['l_gripper_r_finger_tip_link', 'l_gripper_r_finger_tip_joint', 'l_gripper_r_finger_link',
                          'l_gripper_r_finger_joint']
        assert connection == ['l_gripper_palm_link']
        assert chain2 == ['l_gripper_l_finger_joint', 'l_gripper_l_finger_link', 'l_gripper_l_finger_tip_joint',
                          'l_gripper_l_finger_tip_link']

    def test_get_split_chain_group(self):
        root_link = 'r_gripper_l_finger_tip_link'
        tip_link = 'r_gripper_r_finger_tip_link'
        world = create_world_with_pr2()
        world.register_group('r_hand', 'r_wrist_roll_link')
        chain1, connection, chain2 = world.groups['r_hand'].compute_split_chain(root_link, tip_link,
                                                                                True, True, True, True)
        assert chain1 == ['r_gripper_l_finger_tip_link',
                          'r_gripper_l_finger_tip_joint',
                          'r_gripper_l_finger_link',
                          'r_gripper_l_finger_joint']
        assert connection == ['r_gripper_palm_link']
        assert chain2 == ['r_gripper_r_finger_joint',
                          'r_gripper_r_finger_link',
                          'r_gripper_r_finger_tip_joint',
                          'r_gripper_r_finger_tip_link']

    def test_get_split_chain_hsr(self):
        world = create_world_with_hsr(prefix=RobotName)
        root_link = PrefixName('base_link', RobotName)
        tip_link = PrefixName('hand_gripper_tool_frame', RobotName)
        chain1, connection, chain2 = world.compute_split_chain(root_link, tip_link, True, True, True, True)
        chain1 = [n.short_name for n in chain1]
        connection = [n.short_name for n in connection]
        chain2 = [n.short_name for n in chain2]
        assert chain1 == []
        assert connection == ['base_link']
        assert chain2 == ['arm_lift_joint', 'arm_lift_link', 'arm_flex_joint', 'arm_flex_link', 'arm_roll_joint',
                          'arm_roll_link', 'wrist_flex_joint', 'wrist_flex_link', 'wrist_roll_joint', 'wrist_roll_link',
                          'hand_palm_joint', 'hand_palm_link', 'hand_gripper_tool_frame_joint',
                          'hand_gripper_tool_frame']

    def test_get_joint_limits2(self):
        world = create_world_with_pr2()
        lower_limit, upper_limit = world.get_joint_position_limits('l_shoulder_pan_joint')
        assert lower_limit == -0.564601836603
        assert upper_limit == 2.1353981634

    def test_search_branch(self):
        world = create_world_with_pr2()
        result = world.search_branch('odom_x_frame',
                                     stop_at_joint_when=lambda _: False,
                                     stop_at_link_when=lambda _: False)
        assert result == ([], [])
        result = world.search_branch('odom_y_frame',
                                     stop_at_joint_when=world.is_joint_controlled,
                                     stop_at_link_when=lambda _: False,
                                     collect_link_when=world.has_link_collisions)
        assert result == ([], [])
        result = world.search_branch('base_footprint',
                                     stop_at_joint_when=world.is_joint_controlled,
                                     collect_link_when=world.has_link_collisions)
        assert set(result[0]) == {'base_bellow_link',
                                  'fl_caster_l_wheel_link',
                                  'fl_caster_r_wheel_link',
                                  'fl_caster_rotation_link',
                                  'fr_caster_l_wheel_link',
                                  'fr_caster_r_wheel_link',
                                  'fr_caster_rotation_link',
                                  'bl_caster_l_wheel_link',
                                  'bl_caster_r_wheel_link',
                                  'bl_caster_rotation_link',
                                  'br_caster_l_wheel_link',
                                  'br_caster_r_wheel_link',
                                  'br_caster_rotation_link',
                                  'base_link'}
        result = world.search_branch('l_elbow_flex_link',
                                     collect_joint_when=world.is_joint_fixed)
        assert set(result[0]) == set()
        assert set(result[1]) == {'l_force_torque_adapter_joint',
                                  'l_force_torque_joint',
                                  'l_forearm_cam_frame_joint',
                                  'l_forearm_cam_optical_frame_joint',
                                  'l_forearm_joint',
                                  'l_gripper_led_joint',
                                  'l_gripper_motor_accelerometer_joint',
                                  'l_gripper_palm_joint',
                                  'l_gripper_tool_joint'}
        links, joints = world.search_branch('r_wrist_roll_link',
                                            stop_at_joint_when=world.is_joint_controlled,
                                            collect_link_when=world.has_link_collisions,
                                            collect_joint_when=lambda _: True)
        assert set(links) == {'r_gripper_l_finger_tip_link',
                              'r_gripper_l_finger_link',
                              'r_gripper_r_finger_tip_link',
                              'r_gripper_r_finger_link',
                              'r_gripper_palm_link',
                              'r_wrist_roll_link'}
        assert set(joints) == {'r_gripper_palm_joint',
                               'r_gripper_led_joint',
                               'r_gripper_motor_accelerometer_joint',
                               'r_gripper_tool_joint',
                               'r_gripper_motor_slider_joint',
                               'r_gripper_motor_screw_joint',
                               'r_gripper_l_finger_joint',
                               'r_gripper_l_finger_tip_joint',
                               'r_gripper_r_finger_joint',
                               'r_gripper_r_finger_tip_joint',
                               'r_gripper_joint'}
        links, joints = world.search_branch('br_caster_l_wheel_link',
                                            collect_link_when=lambda _: True,
                                            collect_joint_when=lambda _: True)
        assert links == ['br_caster_l_wheel_link']
        assert joints == []

    def test_get_siblings_with_collisions(self):
        world = create_world_with_pr2()
        result = world.get_siblings_with_collisions('odom_x_joint')
        assert result == []
        result = world.get_siblings_with_collisions('odom_y_joint')
        assert result == []
        result = world.get_siblings_with_collisions('odom_z_joint')
        assert result == []
        result = world.get_siblings_with_collisions('l_elbow_flex_joint')
        assert set(result) == {'l_upper_arm_roll_link', 'l_upper_arm_link'}
        result = world.get_siblings_with_collisions('r_wrist_roll_joint')
        assert result == ['r_wrist_flex_link']
        result = world.get_siblings_with_collisions('br_caster_l_wheel_joint')
        assert set(result) == {'base_bellow_link',
                               'fl_caster_l_wheel_link',
                               'fl_caster_r_wheel_link',
                               'fl_caster_rotation_link',
                               'fr_caster_l_wheel_link',
                               'fr_caster_r_wheel_link',
                               'fr_caster_rotation_link',
                               'bl_caster_l_wheel_link',
                               'bl_caster_r_wheel_link',
                               'bl_caster_rotation_link',
                               'br_caster_r_wheel_link',
                               'br_caster_rotation_link',
                               'base_link'}

    def test_get_controlled_parent_joint_of_link(self):
        world = create_world_with_pr2()
        with pytest.raises(KeyError) as e_info:
            world.get_controlled_parent_joint_of_link('odom_combined')
        assert world.get_controlled_parent_joint_of_link('odom_x_frame') == 'odom_x_joint'

    def test_get_parent_joint_of_joint(self):
        world = create_world_with_pr2()
        with pytest.raises(KeyError) as e_info:
            world.get_controlled_parent_joint_of_joint('odom_x_joint')
        with pytest.raises(KeyError) as e_info:
            world.search_for_parent_joint('r_wrist_roll_joint', stop_when=lambda x: False)
        assert world.get_controlled_parent_joint_of_joint('r_torso_lift_side_plate_joint') == 'torso_lift_joint'
        assert world.get_controlled_parent_joint_of_joint('odom_y_joint') == 'odom_x_joint'

    def test_get_all_joint_limits(self):
        world = create_world_with_pr2()
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
        world = create_world_with_pr2()
        world.register_group('r_hand', 'r_wrist_roll_link')
        assert world.groups['r_hand'].get_all_joint_position_limits() == {'r_gripper_joint': (0.0, 0.088),
                                                                          'r_gripper_l_finger_joint': (0.0, 0.548),
                                                                          'r_gripper_l_finger_tip_joint': (0.0, 0.548),
                                                                          'r_gripper_motor_screw_joint': (None, None),
                                                                          'r_gripper_motor_slider_joint': (-0.1, 0.1),
                                                                          'r_gripper_r_finger_joint': (0.0, 0.548),
                                                                          'r_gripper_r_finger_tip_joint': (0.0, 0.548)}

    def test_possible_collision_combinations(self):
        world = create_world_with_pr2()
        result = world.possible_collision_combinations('robot')
        reference = {world.sort_links(link_a, link_b) for link_a, link_b in
                     combinations(world.groups['robot'].link_names_with_collisions, 2) if
                     not world.groups['robot'].are_linked(link_a, link_b)}
        assert result == reference

    @given(rnd_joint_state(pr2_joint_limits))
    def test_pr2_fk1(self, js):
        """
        :type js:
        """
        world = create_world_with_pr2()
        root = 'odom_combined'
        tips = ['l_gripper_tool_frame', 'r_gripper_tool_frame']
        for tip in tips:
            mjs = JointStates()
            for joint_name, position in js.items():
                mjs[joint_name].position = position
            world.joint_state = mjs
            fk = world.compute_fk_pose(root, tip).pose

    def test_compute_chain_reduced_to_controlled_joints(self):
        world = create_world_with_pr2()
        link_a, link_b = world.compute_chain_reduced_to_controlled_joints('r_gripper_tool_frame',
                                                                          'l_gripper_tool_frame')
        assert link_a == 'r_wrist_roll_link'
        assert link_b == 'l_wrist_roll_link'

    def test_compute_chain_reduced_to_controlled_joints2(self):
        world = create_world_with_pr2()
        link_a, link_b = world.compute_chain_reduced_to_controlled_joints('l_upper_arm_link', 'r_upper_arm_link')
        assert link_a == 'l_upper_arm_roll_link'
        assert link_b == 'r_upper_arm_roll_link'

    def test_compute_chain_reduced_to_controlled_joints3(self):
        world = create_world_with_pr2()
        with pytest.raises(KeyError):
            world.compute_chain_reduced_to_controlled_joints('l_wrist_roll_link', 'l_gripper_r_finger_link')

