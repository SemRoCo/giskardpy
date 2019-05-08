import pytest
from geometry_msgs.msg import Pose, Point, Quaternion

from giskardpy.exceptions import DuplicateNameException, UnknownBodyException
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import make_world_body_box, make_world_body_sphere, make_world_body_cylinder, make_urdf_world_body
from utils_for_tests import pr2_urdf, donbot_urdf, boxy_urdf, base_bot_urdf


@pytest.fixture(scope=u'module')
def module_setup(request):
    pass


@pytest.fixture()
def function_setup(request, module_setup):
    """
    :rtype: WorldObject
    """
    pass


class TestUrdfObject(object):
    cls = URDFObject

    def test_urdf_from_file(self, function_setup):
        """
        :type parsed_pr2: tested_class
        """
        parsed_pr2 = self.cls(pr2_urdf())
        assert len(parsed_pr2.get_joint_names()) == 96
        assert len(parsed_pr2.get_link_names()) == 97
        assert parsed_pr2.get_name() == u'pr2'

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

    def test_from_world_body_cone(self, function_setup):
        pass

    def test_from_world_body_invalid_primitive_type(self, function_setup):
        pass

    def test_form_world_body_unsupported_type(self, function_setup):
        pass

    def test_from_world_body_urdf(self, function_setup):
        wb = make_urdf_world_body(u'pr2', pr2_urdf())
        urdf_obj = self.cls.from_world_body(wb)
        assert len(urdf_obj.get_joint_names()) == 96
        assert len(urdf_obj.get_link_names()) == 97

    def test_from_world_body_mesh(self, function_setup):
        wb = make_world_body_cylinder()
        urdf_obj = self.cls.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert len(urdf_obj.get_joint_names()) == 0

    def test_get_parent_link_name(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        assert parsed_pr2.get_parent_link_of_link(u'l_gripper_tool_frame') == u'l_gripper_palm_link'

    def test_get_link_names_from_chain(self, function_setup):
        pass

    def test_get_links_from_sub_tree1(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        urdf_obj = parsed_pr2.get_sub_tree_at_joint(u'torso_lift_joint')
        assert len(urdf_obj.get_link_names()) == 80
        assert len(urdf_obj.get_joint_names()) == 79

    def test_attach_urdf_object1(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        num_of_links_before = len(parsed_pr2.get_link_names())
        num_of_joints_before = len(parsed_pr2.get_joint_names())
        link_chain_before = len(parsed_pr2.get_links_from_sub_tree(u'torso_lift_joint'))
        box = self.cls.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
        assert box.get_name() in parsed_pr2.get_link_names()
        assert len(parsed_pr2.get_link_names()) == num_of_links_before + 1
        assert len(parsed_pr2.get_joint_names()) == num_of_joints_before + 1
        assert len(parsed_pr2.get_links_from_sub_tree(u'torso_lift_joint')) == link_chain_before + 1

    def test_attach_urdf_object2(self, function_setup):
        parsed_base_bot = self.cls(base_bot_urdf())
        links_before = set(parsed_base_bot.get_link_names())
        joints_before = set(parsed_base_bot.get_joint_names())
        donbot = make_urdf_world_body(u'mustafa', donbot_urdf())
        donbot_obj = self.cls.from_world_body(donbot)
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        parsed_base_bot.attach_urdf_object(donbot_obj, u'eef', p)
        assert donbot_obj.get_root() in parsed_base_bot.get_link_names()
        assert set(parsed_base_bot.get_link_names()).difference(
            links_before.union(set(donbot_obj.get_link_names()))) == set()
        assert set(parsed_base_bot.get_joint_names()).difference(
            joints_before.union(set(donbot_obj.get_joint_names()))) == {u'mustafa'}
        links = [l.name for l in parsed_base_bot.get_urdf_robot().links]
        assert len(links) == len(set(links))
        joints = [j.name for j in parsed_base_bot.get_urdf_robot().joints]
        assert len(joints) == len(set(joints))

    # TODO test is the tree is valid

    def test_attach_urdf_object3(self, function_setup):
        parsed_donbot = self.cls(donbot_urdf())
        pr2 = make_urdf_world_body(u'pr2', pr2_urdf())
        pr2_obj = self.cls.from_world_body(pr2)
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        try:
            parsed_donbot.attach_urdf_object(pr2_obj, u'eef', p)
            assert False, u'expected exception'
        except Exception:
            assert True

    def test_attach_twice(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        box = self.cls.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
        try:
            parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
            assert False, u'didnt get expected exception'
        except DuplicateNameException as e:
            assert True

    def test_attach_to_non_existing_link(self, function_setup):
        parsed_base_bot = self.cls(base_bot_urdf())
        box = self.cls.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        try:
            parsed_base_bot.attach_urdf_object(box, u'muh', p)
            assert False, u'didnt get expected exception'
        except UnknownBodyException:
            assert True

    def test_attach_obj_with_joint_name(self, function_setup):
        parsed_base_bot = self.cls(base_bot_urdf())
        box = self.cls.from_world_body(make_world_body_box(u'rot_z'))
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        try:
            parsed_base_bot.attach_urdf_object(box, u'eef', p)
            assert False, u'didnt get expected exception'
        except DuplicateNameException as e:
            assert True

    def test_attach_obj_with_link_name(self, function_setup):
        parsed_base_bot = self.cls(base_bot_urdf())
        box = self.cls.from_world_body(make_world_body_box(u'eef'))
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        try:
            parsed_base_bot.attach_urdf_object(box, u'eef', p)
            assert False, u'didnt get expected exception'
        except DuplicateNameException as e:
            assert True

    def test_attach5(self, function_setup):
        # TODO test that attach an object which has the same name as on of its links or joints
        pass

    def test_detach_object(self, function_setup):
        """
        :type parsed_base_bot: self.cls
        """
        parsed_base_bot = self.cls(base_bot_urdf())
        parsed_base_bot.detach_sub_tree(u'rot_z')
        assert len(parsed_base_bot.get_link_names()) == 3
        assert len(parsed_base_bot.get_joint_names()) == 2
        assert u'rot_z' not in parsed_base_bot.get_joint_names()
        assert u'eef' not in parsed_base_bot.get_link_names()

    def test_amputate_left_arm(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        parsed_pr2.detach_sub_tree(u'l_shoulder_pan_joint')
        assert len(parsed_pr2.get_link_names()) == 73
        assert len(parsed_pr2.get_joint_names()) == 72

    def test_reset1(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        links_before = set(parsed_pr2.get_link_names())
        joints_before = set(parsed_pr2.get_joint_names())
        parsed_pr2.detach_sub_tree(u'l_shoulder_pan_joint')
        parsed_pr2.reset()
        assert set(parsed_pr2.get_link_names()) == links_before
        assert set(parsed_pr2.get_joint_names()) == joints_before

    def test_reset2(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        links_before = set(parsed_pr2.get_link_names())
        joints_before = set(parsed_pr2.get_joint_names())

        box = self.cls.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)

        parsed_pr2.reset()
        assert set(parsed_pr2.get_link_names()) == links_before
        assert set(parsed_pr2.get_joint_names()) == joints_before

    def test_attach_detach(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        box = self.cls.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        original_urdf = parsed_pr2.get_urdf()
        parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
        parsed_pr2.detach_sub_tree(u'box')
        assert original_urdf == parsed_pr2.get_urdf()

    def test_detach_non_existing_object(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        try:
            parsed_pr2.detach_sub_tree(u'muh')
            assert False, u'didnt get expected exception'
        except KeyError:
            assert True

    def test_detach_at_link(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        try:
            parsed_pr2.detach_sub_tree(u'torso_lift_link')
            assert False, u'didnt get expected exception'
        except KeyError:
            assert True

    def test_get_joint_limits1(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        assert len(parsed_pr2.get_all_joint_limits()) == 45

    def test_get_joint_limits2(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        lower_limit, upper_limit = parsed_pr2.get_joint_limits(u'l_shoulder_pan_joint')
        assert lower_limit == -0.564601836603
        assert upper_limit == 2.1353981634

    def test_get_joint_limits3(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        lower_limit, upper_limit = parsed_pr2.get_joint_limits(u'l_wrist_roll_joint')
        assert lower_limit == None
        assert upper_limit == None

    def test_get_joint_limits4(self, function_setup):
        parsed_base_bot = self.cls(base_bot_urdf())
        lower_limit, upper_limit = parsed_base_bot.get_joint_limits(u'joint_x')
        assert lower_limit == -3
        assert upper_limit == 3

    def test_get_joint_names_from_chain(self, function_setup):
        pass

    def test_get_joint_names_from_chain_controllable(self, function_setup):
        pass

    def test_get_joint_names_controllable(self, function_setup):
        pass

    def test_is_joint_mimic(self, function_setup):
        pass

    def test_is_joint_continuous(self, function_setup):
        pass

    def test_get_joint_type(self, function_setup):
        pass

    def test_is_joint_type_supported(self, function_setup):
        pass

    def test_is_rotational_joint(self, function_setup):
        pass

    def test_is_translational_joint(self, function_setup):
        pass

    def test_get_sub_tree_link_names_with_collision(self, function_setup):
        pass

    def test_get_chain1(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        root = parsed_pr2.get_root()
        tip = u'l_gripper_tool_frame'
        chain = parsed_pr2.get_joint_names_from_chain(root, tip)
        assert chain == [u'base_footprint_joint',
                         u'torso_lift_joint',
                         u'l_shoulder_pan_joint',
                         u'l_shoulder_lift_joint',
                         u'l_upper_arm_roll_joint',
                         u'l_upper_arm_joint',
                         u'l_elbow_flex_joint',
                         u'l_forearm_roll_joint',
                         u'l_forearm_joint',
                         u'l_wrist_flex_joint',
                         u'l_wrist_roll_joint',
                         u'l_force_torque_adapter_joint',
                         u'l_force_torque_joint',
                         u'l_gripper_palm_joint',
                         u'l_gripper_tool_joint']

    def test_get_chain2(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        root = u'r_gripper_tool_frame'
        tip = u'l_gripper_tool_frame'
        chain = parsed_pr2.get_joint_names_from_chain(root, tip)
        assert chain == [u'r_gripper_tool_joint',
                         u'r_gripper_palm_joint',
                         u'r_wrist_roll_joint',
                         u'r_wrist_flex_joint',
                         u'r_forearm_joint',
                         u'r_forearm_roll_joint',
                         u'r_elbow_flex_joint',
                         u'r_upper_arm_joint',
                         u'r_upper_arm_roll_joint',
                         u'r_shoulder_lift_joint',
                         u'r_shoulder_pan_joint',
                         u'l_shoulder_pan_joint',
                         u'l_shoulder_lift_joint',
                         u'l_upper_arm_roll_joint',
                         u'l_upper_arm_joint',
                         u'l_elbow_flex_joint',
                         u'l_forearm_roll_joint',
                         u'l_forearm_joint',
                         u'l_wrist_flex_joint',
                         u'l_wrist_roll_joint',
                         u'l_force_torque_adapter_joint',
                         u'l_force_torque_joint',
                         u'l_gripper_palm_joint',
                         u'l_gripper_tool_joint']

    def test_get_chain3(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        tip = parsed_pr2.get_root()
        root = u'l_gripper_tool_frame'
        chain = parsed_pr2.get_joint_names_from_chain(root, tip)
        assert chain == [u'l_gripper_tool_joint',
                         u'l_gripper_palm_joint',
                         u'l_force_torque_joint',
                         u'l_force_torque_adapter_joint',
                         u'l_wrist_roll_joint',
                         u'l_wrist_flex_joint',
                         u'l_forearm_joint',
                         u'l_forearm_roll_joint',
                         u'l_elbow_flex_joint',
                         u'l_upper_arm_joint',
                         u'l_upper_arm_roll_joint',
                         u'l_shoulder_lift_joint',
                         u'l_shoulder_pan_joint',
                         u'torso_lift_joint',
                         u'base_footprint_joint']

    def test_get_chain4(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        tip = u'l_gripper_tool_frame'
        root = u'l_gripper_tool_frame'
        chain = parsed_pr2.get_joint_names_from_chain(root, tip)
        assert chain == []

    def test_get_chain5(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        tip = u'l_upper_arm_link'
        root = u'l_shoulder_lift_link'
        chain = parsed_pr2.get_joint_names_from_chain(root, tip)
        assert chain == [u'l_upper_arm_roll_joint', u'l_upper_arm_joint']

    def test_get_chain6(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        tip = u'l_gripper_palm_link'
        root = u'l_gripper_tool_frame'
        chain = parsed_pr2.get_joint_names_from_chain(root, tip)
        assert chain == [u'l_gripper_tool_joint']

    def test_get_chain_attached(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        box = self.cls.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0.1)
        p.orientation = Quaternion(1, 0, 0, 0)
        parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
        tip = u'l_gripper_tool_frame'
        root = box.get_name()
        chain = parsed_pr2.get_joint_names_from_chain(root, tip)
        assert chain == [box.get_name()]

    def test_get_chain_fixed_joints(self, function_setup):
        # TODO test fixed joints
        pass

    def test_get_chain_links(self, function_setup):
        # TODO test link=true
        pass

    def test_get_sub_tree_link_names_with_collision_boxy(self, function_setup):
        parsed_boxy = self.cls(boxy_urdf())
        expected = {u'left_arm_2_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_arm_3_link', u'left_arm_4_link',
                                          u'left_gripper_gripper_right_link'},
                    u'neck_joint_end': {u'neck_look_target'},
                    u'neck_wrist_1_joint': {u'neck_look_target', u'neck_adapter_iso50_kinect2_frame_in',
                                            u'neck_wrist_3_link', u'neck_wrist_2_link', u'neck_ee_link',
                                            u'head_mount_kinect2_rgb_optical_frame', u'neck_wrist_1_link'},
                    u'right_arm_2_joint': {u'right_gripper_finger_right_link', u'right_arm_3_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_gripper_gripper_left_link',
                                           u'right_arm_6_link', u'right_gripper_base_link', u'right_arm_4_link',
                                           u'right_arm_7_link', u'right_gripper_finger_left_link'},
                    u'right_arm_4_joint': {u'right_gripper_finger_right_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_gripper_base_link',
                                           u'right_arm_6_link', u'right_gripper_gripper_left_link', u'right_arm_7_link',
                                           u'right_gripper_finger_left_link'},
                    u'neck_wrist_3_joint': {u'neck_look_target', u'neck_adapter_iso50_kinect2_frame_in',
                                            u'neck_ee_link', u'head_mount_kinect2_rgb_optical_frame',
                                            u'neck_wrist_3_link'},
                    u'right_arm_3_joint': {u'right_gripper_finger_right_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_gripper_base_link',
                                           u'right_arm_6_link', u'right_gripper_gripper_left_link', u'right_arm_4_link',
                                           u'right_arm_7_link', u'right_gripper_finger_left_link'},
                    u'right_gripper_base_gripper_right_joint': {u'right_gripper_finger_right_link',
                                                                u'right_gripper_gripper_right_link'},
                    u'left_gripper_base_gripper_right_joint': {u'left_gripper_gripper_right_link',
                                                               u'left_gripper_finger_right_link'},
                    u'left_arm_0_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_1_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_arm_3_link', u'left_arm_4_link',
                                          u'left_arm_2_link', u'left_gripper_gripper_right_link'},
                    u'right_gripper_base_gripper_left_joint': {u'right_gripper_gripper_left_link',
                                                               u'right_gripper_finger_left_link'},
                    u'left_arm_4_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_gripper_gripper_right_link'},
                    u'left_arm_6_joint': {u'left_gripper_finger_left_link', u'left_gripper_gripper_left_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_gripper_gripper_right_link'},
                    u'right_arm_1_joint': {u'right_gripper_finger_right_link', u'right_arm_3_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_arm_2_link',
                                           u'right_gripper_gripper_left_link', u'right_arm_6_link',
                                           u'right_gripper_base_link', u'right_arm_4_link', u'right_arm_7_link',
                                           u'right_gripper_finger_left_link'},
                    u'left_arm_1_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_arm_3_link', u'left_arm_4_link',
                                          u'left_arm_2_link', u'left_gripper_gripper_right_link'},
                    u'neck_wrist_2_joint': {u'neck_look_target', u'neck_adapter_iso50_kinect2_frame_in',
                                            u'neck_wrist_3_link', u'neck_wrist_2_link', u'neck_ee_link',
                                            u'head_mount_kinect2_rgb_optical_frame'},
                    u'triangle_base_joint': {u'left_arm_3_link', u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                             u'left_gripper_base_link', u'left_gripper_finger_right_link',
                                             u'left_arm_2_link', u'right_gripper_finger_right_link',
                                             u'left_gripper_finger_left_link', u'right_arm_3_link',
                                             u'calib_right_arm_base_link', u'triangle_base_link', u'right_arm_4_link',
                                             u'right_gripper_finger_left_link', u'left_arm_6_link',
                                             u'calib_left_arm_base_link', u'right_gripper_base_link',
                                             u'right_gripper_gripper_right_link', u'left_arm_1_link',
                                             u'left_arm_7_link', u'right_gripper_gripper_left_link',
                                             u'right_arm_1_link', u'left_arm_4_link', u'right_arm_5_link',
                                             u'right_arm_2_link', u'right_arm_6_link', u'right_arm_7_link',
                                             u'left_gripper_gripper_right_link'},
                    u'neck_elbow_joint': {u'neck_look_target', u'neck_adapter_iso50_kinect2_frame_in',
                                          u'neck_forearm_link', u'neck_wrist_3_link', u'neck_wrist_2_link',
                                          u'neck_ee_link', u'head_mount_kinect2_rgb_optical_frame',
                                          u'neck_wrist_1_link'},
                    u'right_arm_5_joint': {u'right_gripper_finger_right_link', u'right_gripper_gripper_right_link',
                                           u'right_gripper_base_link', u'right_arm_6_link',
                                           u'right_gripper_gripper_left_link', u'right_arm_7_link',
                                           u'right_gripper_finger_left_link'},
                    u'left_arm_3_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_arm_4_link',
                                          u'left_gripper_gripper_right_link'},
                    u'neck_shoulder_pan_joint': {u'neck_upper_arm_link', u'neck_look_target',
                                                 u'neck_adapter_iso50_kinect2_frame_in', u'neck_forearm_link',
                                                 u'neck_wrist_3_link', u'neck_wrist_2_link', u'neck_shoulder_link',
                                                 u'head_mount_kinect2_rgb_optical_frame', u'neck_wrist_1_link',
                                                 u'neck_ee_link'},
                    u'right_arm_0_joint': {u'right_gripper_finger_right_link', u'right_arm_3_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_arm_2_link',
                                           u'right_gripper_gripper_left_link', u'right_arm_6_link',
                                           u'right_gripper_base_link', u'right_arm_1_link', u'right_arm_4_link',
                                           u'right_arm_7_link', u'right_gripper_finger_left_link'},
                    u'neck_shoulder_lift_joint': {u'neck_upper_arm_link', u'neck_look_target',
                                                  u'neck_adapter_iso50_kinect2_frame_in', u'neck_forearm_link',
                                                  u'neck_wrist_3_link', u'neck_wrist_2_link', u'neck_ee_link',
                                                  u'head_mount_kinect2_rgb_optical_frame', u'neck_wrist_1_link'},
                    u'left_arm_5_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_gripper_base_link',
                                          u'left_arm_7_link', u'left_gripper_finger_right_link',
                                          u'left_gripper_gripper_right_link'},
                    u'left_gripper_base_gripper_left_joint': {u'left_gripper_finger_left_link',
                                                              u'left_gripper_gripper_left_link'},
                    u'right_arm_6_joint': {u'right_gripper_finger_right_link', u'right_gripper_gripper_right_link',
                                           u'right_gripper_base_link', u'right_gripper_gripper_left_link',
                                           u'right_arm_7_link', u'right_gripper_finger_left_link'}}
        for joint in parsed_boxy.get_joint_names_controllable():
            assert set(parsed_boxy.get_sub_tree_link_names_with_collision(joint)).difference(expected[joint]) == set()

    def test_get_sub_tree_link_names_with_collision_pr2(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        expected = {u'l_shoulder_pan_joint': {u'l_shoulder_pan_link', u'l_shoulder_lift_link', u'l_upper_arm_roll_link',
                                              u'l_upper_arm_link', u'l_elbow_flex_link', u'l_forearm_roll_link',
                                              u'l_forearm_link', u'l_wrist_flex_link', u'l_wrist_roll_link',
                                              u'l_gripper_palm_link', u'l_gripper_l_finger_link',
                                              u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link',
                                              u'l_gripper_r_finger_tip_link'},
                    u'br_caster_l_wheel_joint': {u'br_caster_l_wheel_link'},
                    u'r_gripper_l_finger_tip_joint': {u'r_gripper_l_finger_tip_link'},
                    u'r_elbow_flex_joint': {u'r_elbow_flex_link', u'r_forearm_roll_link', u'r_forearm_link',
                                            u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_gripper_palm_link',
                                            u'r_gripper_l_finger_link', u'r_gripper_r_finger_link',
                                            u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'torso_lift_joint': {u'torso_lift_link', u'head_pan_link', u'laser_tilt_mount_link',
                                          u'r_shoulder_pan_link', u'l_shoulder_pan_link', u'head_tilt_link',
                                          u'r_shoulder_lift_link', u'l_shoulder_lift_link', u'head_plate_frame',
                                          u'r_upper_arm_roll_link', u'l_upper_arm_roll_link', u'r_upper_arm_link',
                                          u'l_upper_arm_link', u'r_elbow_flex_link', u'l_elbow_flex_link',
                                          u'r_forearm_roll_link', u'l_forearm_roll_link', u'r_forearm_link',
                                          u'l_forearm_link', u'r_wrist_flex_link', u'l_wrist_flex_link',
                                          u'r_wrist_roll_link', u'l_wrist_roll_link', u'r_gripper_palm_link',
                                          u'r_gripper_l_finger_link', u'r_gripper_r_finger_link',
                                          u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link',
                                          u'l_gripper_palm_link', u'l_gripper_l_finger_link',
                                          u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link',
                                          u'l_gripper_r_finger_tip_link'},
                    u'r_gripper_l_finger_joint': {u'r_gripper_l_finger_link', u'r_gripper_l_finger_tip_link'},
                    u'r_forearm_roll_joint': {u'r_forearm_roll_link', u'r_forearm_link', u'r_wrist_flex_link',
                                              u'r_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link',
                                              u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link',
                                              u'r_gripper_r_finger_tip_link'},
                    u'l_gripper_r_finger_tip_joint': {u'l_gripper_r_finger_tip_link'},
                    u'r_shoulder_lift_joint': {u'r_shoulder_lift_link', u'r_upper_arm_roll_link', u'r_upper_arm_link',
                                               u'r_elbow_flex_link', u'r_forearm_roll_link', u'r_forearm_link',
                                               u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_gripper_palm_link',
                                               u'r_gripper_l_finger_link', u'r_gripper_r_finger_link',
                                               u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'fl_caster_rotation_joint': {u'fl_caster_rotation_link', u'fl_caster_l_wheel_link',
                                                  u'fl_caster_r_wheel_link'},
                    u'l_gripper_motor_screw_joint': set(),
                    u'r_wrist_roll_joint': {u'r_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link',
                                            u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link',
                                            u'r_gripper_r_finger_tip_link'},
                    u'r_gripper_motor_slider_joint': set(),
                    u'l_forearm_roll_joint': {u'l_forearm_roll_link', u'l_forearm_link', u'l_wrist_flex_link',
                                              u'l_wrist_roll_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link',
                                              u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link',
                                              u'l_gripper_r_finger_tip_link'},
                    u'r_gripper_joint': set(),
                    u'bl_caster_rotation_joint': {u'bl_caster_rotation_link', u'bl_caster_l_wheel_link',
                                                  u'bl_caster_r_wheel_link'},
                    u'fl_caster_r_wheel_joint': {u'fl_caster_r_wheel_link'},
                    u'l_shoulder_lift_joint': {u'l_shoulder_lift_link', u'l_upper_arm_roll_link', u'l_upper_arm_link',
                                               u'l_elbow_flex_link', u'l_forearm_roll_link', u'l_forearm_link',
                                               u'l_wrist_flex_link', u'l_wrist_roll_link', u'l_gripper_palm_link',
                                               u'l_gripper_l_finger_link', u'l_gripper_r_finger_link',
                                               u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'head_pan_joint': {u'head_pan_link', u'head_tilt_link', u'head_plate_frame'},
                    u'head_tilt_joint': {u'head_tilt_link', u'head_plate_frame'},
                    u'fr_caster_l_wheel_joint': {u'fr_caster_l_wheel_link'},
                    u'fl_caster_l_wheel_joint': {u'fl_caster_l_wheel_link'},
                    u'l_gripper_motor_slider_joint': set(),
                    u'br_caster_r_wheel_joint': {u'br_caster_r_wheel_link'},
                    u'r_gripper_motor_screw_joint': set(),
                    u'r_upper_arm_roll_joint': {u'r_upper_arm_roll_link', u'r_upper_arm_link', u'r_elbow_flex_link',
                                                u'r_forearm_roll_link', u'r_forearm_link', u'r_wrist_flex_link',
                                                u'r_wrist_roll_link', u'r_gripper_palm_link',
                                                u'r_gripper_l_finger_link', u'r_gripper_r_finger_link',
                                                u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'fr_caster_rotation_joint': {u'fr_caster_rotation_link', u'fr_caster_l_wheel_link',
                                                  u'fr_caster_r_wheel_link'},
                    u'torso_lift_motor_screw_joint': set(),
                    u'bl_caster_l_wheel_joint': {u'bl_caster_l_wheel_link'},
                    u'r_wrist_flex_joint': {u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_gripper_palm_link',
                                            u'r_gripper_l_finger_link', u'r_gripper_r_finger_link',
                                            u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'r_gripper_r_finger_tip_joint': {u'r_gripper_r_finger_tip_link'},
                    u'l_elbow_flex_joint': {u'l_elbow_flex_link', u'l_forearm_roll_link', u'l_forearm_link',
                                            u'l_wrist_flex_link', u'l_wrist_roll_link', u'l_gripper_palm_link',
                                            u'l_gripper_l_finger_link', u'l_gripper_r_finger_link',
                                            u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'laser_tilt_mount_joint': {u'laser_tilt_mount_link'},
                    u'r_shoulder_pan_joint': {u'r_shoulder_pan_link', u'r_shoulder_lift_link', u'r_upper_arm_roll_link',
                                              u'r_upper_arm_link', u'r_elbow_flex_link', u'r_forearm_roll_link',
                                              u'r_forearm_link', u'r_wrist_flex_link', u'r_wrist_roll_link',
                                              u'r_gripper_palm_link', u'r_gripper_l_finger_link',
                                              u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link',
                                              u'r_gripper_r_finger_tip_link'},
                    u'fr_caster_r_wheel_joint': {u'fr_caster_r_wheel_link'},
                    u'l_wrist_roll_joint': {u'l_wrist_roll_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link',
                                            u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link',
                                            u'l_gripper_r_finger_tip_link'},
                    u'r_gripper_r_finger_joint': {u'r_gripper_r_finger_link', u'r_gripper_r_finger_tip_link'},
                    u'bl_caster_r_wheel_joint': {u'bl_caster_r_wheel_link'},
                    u'l_gripper_joint': set(),
                    u'l_gripper_l_finger_tip_joint': {u'l_gripper_l_finger_tip_link'},
                    u'br_caster_rotation_joint': {u'br_caster_rotation_link', u'br_caster_l_wheel_link',
                                                  u'br_caster_r_wheel_link'},
                    u'l_gripper_l_finger_joint': {u'l_gripper_l_finger_link', u'l_gripper_l_finger_tip_link'},
                    u'l_wrist_flex_joint': {u'l_wrist_flex_link', u'l_wrist_roll_link', u'l_gripper_palm_link',
                                            u'l_gripper_l_finger_link', u'l_gripper_r_finger_link',
                                            u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'l_upper_arm_roll_joint': {u'l_upper_arm_roll_link', u'l_upper_arm_link', u'l_elbow_flex_link',
                                                u'l_forearm_roll_link', u'l_forearm_link', u'l_wrist_flex_link',
                                                u'l_wrist_roll_link', u'l_gripper_palm_link',
                                                u'l_gripper_l_finger_link', u'l_gripper_r_finger_link',
                                                u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'l_gripper_r_finger_joint': {u'l_gripper_r_finger_link', u'l_gripper_r_finger_tip_link'}}
        for joint in parsed_pr2.get_joint_names_controllable():
            assert set(parsed_pr2.get_sub_tree_link_names_with_collision(joint)).difference(expected[joint]) == set()

    def test_get_sub_tree_link_names_with_collision_donbot(self, function_setup):
        parsed_donbot = self.cls(donbot_urdf())
        expected = {u'ur5_wrist_3_joint': {u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                           u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                           u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_elbow_joint': {u'ur5_forearm_link', u'ur5_wrist_1_link', u'ur5_wrist_2_link',
                                         u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                         u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                         u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_wrist_1_joint': {u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link',
                                           u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link',
                                           u'gripper_finger_left_link', u'gripper_gripper_right_link',
                                           u'gripper_finger_right_link'},
                    u'odom_z_joint': {u'base_link', u'plate', u'ur5_base_link', u'ur5_shoulder_link',
                                      u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link',
                                      u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                      u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                      u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_shoulder_lift_joint': {u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link',
                                                 u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link',
                                                 u'gripper_base_link', u'gripper_gripper_left_link',
                                                 u'gripper_finger_left_link', u'gripper_gripper_right_link',
                                                 u'gripper_finger_right_link'},
                    u'odom_y_joint': {u'base_link', u'plate', u'ur5_base_link', u'ur5_shoulder_link',
                                      u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link',
                                      u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                      u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                      u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_wrist_2_joint': {u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link',
                                           u'gripper_base_link', u'gripper_gripper_left_link',
                                           u'gripper_finger_left_link', u'gripper_gripper_right_link',
                                           u'gripper_finger_right_link'},
                    u'odom_x_joint': {u'base_link', u'plate', u'ur5_base_link', u'ur5_shoulder_link',
                                      u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link',
                                      u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                      u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                      u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_shoulder_pan_joint': {u'ur5_shoulder_link', u'ur5_upper_arm_link', u'ur5_forearm_link',
                                                u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link',
                                                u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link',
                                                u'gripper_finger_left_link', u'gripper_gripper_right_link',
                                                u'gripper_finger_right_link'},
                    u'gripper_joint': {u'gripper_gripper_right_link', u'gripper_finger_right_link'}}
        for joint in parsed_donbot.get_joint_names_controllable():
            assert set(parsed_donbot.get_sub_tree_link_names_with_collision(joint)).difference(expected[joint]) == set()

    def test_has_link_collision(self, function_setup):
        pass
