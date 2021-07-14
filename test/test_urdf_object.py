import pytest
from geometry_msgs.msg import Pose, Point, Quaternion

from giskardpy.exceptions import DuplicateNameException, UnknownBodyException
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import make_world_body_box, make_world_body_sphere, make_world_body_cylinder, make_urdf_world_body
from utils_for_tests import pr2_urdf, donbot_urdf, boxy_urdf, base_bot_urdf

set_of_pr2_joints = {'r_gripper_motor_accelerometer_joint',
                     'high_def_optical_frame_joint',
                     'r_gripper_palm_joint',
                     'head_plate_frame_joint',
                     'l_shoulder_pan_joint',
                     'br_caster_l_wheel_joint',
                     'wide_stereo_r_stereo_camera_optical_frame_joint',
                     'odom_z_joint',
                     'r_gripper_l_finger_tip_joint',
                     'laser_tilt_joint',
                     'head_mount_kinect2_rgb_optical_frame_joint',
                     'r_elbow_flex_joint',
                     'torso_lift_joint',
                     'l_force_torque_joint',
                     'high_def_frame_joint',
                     'r_gripper_tool_joint',
                     'r_gripper_l_finger_joint',
                     'r_forearm_roll_joint',
                     'odom_y_joint',
                     'r_forearm_cam_optical_frame_joint',
                     'narrow_stereo_r_stereo_camera_optical_frame_joint',
                     'narrow_stereo_l_stereo_camera_frame_joint',
                     'l_gripper_r_finger_tip_joint',
                     'head_mount_prosilica_optical_frame_joint',
                     'odom_x_joint',
                     'imu_joint',
                     'l_gripper_motor_accelerometer_joint',
                     'r_shoulder_lift_joint',
                     'head_mount_kinect_rgb_optical_frame_joint',
                     'fl_caster_rotation_joint',
                     'l_gripper_motor_screw_joint',
                     'r_gripper_led_joint',
                     'narrow_stereo_l_stereo_camera_optical_frame_joint',
                     'r_wrist_roll_joint',
                     'double_stereo_frame_joint',
                     'narrow_stereo_optical_frame_joint',
                     'r_gripper_motor_slider_joint',
                     'head_mount_kinect_ir_joint',
                     'l_force_torque_adapter_joint',
                     'l_forearm_roll_joint',
                     'r_gripper_joint',
                     'bl_caster_rotation_joint',
                     'fl_caster_r_wheel_joint',
                     'l_shoulder_lift_joint',
                     'narrow_stereo_r_stereo_camera_frame_joint',
                     'head_pan_joint',
                     'head_mount_joint',
                     'projector_wg6802418_frame_joint',
                     'l_upper_arm_joint',
                     'l_forearm_cam_optical_frame_joint',
                     'head_tilt_joint',
                     'fr_caster_l_wheel_joint',
                     'head_mount_prosilica_joint',
                     'head_mount_kinect_ir_optical_frame_joint',
                     'narrow_stereo_frame_joint',
                     'wide_stereo_l_stereo_camera_frame_joint',
                     'wide_stereo_r_stereo_camera_frame_joint',
                     'r_forearm_cam_frame_joint',
                     'fl_caster_l_wheel_joint',
                     'l_forearm_joint',
                     'l_gripper_l_finger_tip_joint',
                     'l_gripper_motor_slider_joint',
                     'l_gripper_led_joint',
                     'br_caster_r_wheel_joint',
                     'r_gripper_motor_screw_joint',
                     'wide_stereo_optical_frame_joint',
                     'l_gripper_palm_joint',
                     'r_upper_arm_roll_joint',
                     'sensor_mount_frame_joint',
                     'fr_caster_rotation_joint',
                     'torso_lift_motor_screw_joint',
                     'head_mount_kinect_rgb_joint',
                     'head_mount_kinect2_ir_optical_frame_joint',
                     'l_gripper_l_finger_joint',
                     'bl_caster_l_wheel_joint',
                     'r_wrist_flex_joint',
                     'l_forearm_cam_frame_joint',
                     'base_bellow_joint',
                     'r_gripper_r_finger_tip_joint',
                     'l_elbow_flex_joint',
                     'base_laser_joint',
                     'laser_tilt_mount_joint',
                     'wide_stereo_l_stereo_camera_optical_frame_joint',
                     'r_shoulder_pan_joint',
                     'fr_caster_r_wheel_joint',
                     'l_wrist_roll_joint',
                     'r_gripper_r_finger_joint',
                     'bl_caster_r_wheel_joint',
                     'l_torso_lift_side_plate_joint',
                     'l_gripper_joint',
                     'r_upper_arm_joint',
                     'r_forearm_joint',
                     'l_gripper_tool_joint',
                     'br_caster_rotation_joint',
                     'wide_stereo_frame_joint',
                     'base_footprint_joint',
                     'projector_wg6802418_child_frame_joint',
                     'r_torso_lift_side_plate_joint',
                     'l_upper_arm_roll_joint',
                     'l_gripper_r_finger_joint',
                     'l_wrist_flex_joint'}
set_of_pr2_links = {'l_upper_arm_link',
                    'laser_tilt_link',
                    'l_shoulder_lift_link',
                    'r_gripper_l_finger_tip_link',
                    'projector_wg6802418_frame',
                    'narrow_stereo_r_stereo_camera_frame',
                    'high_def_frame',
                    'fl_caster_l_wheel_link',
                    'br_caster_rotation_link',
                    'base_bellow_link',
                    'head_mount_link',
                    'head_tilt_link',
                    'r_torso_lift_side_plate_link',
                    'l_gripper_palm_link',
                    'l_wrist_flex_link',
                    'head_mount_kinect2_rgb_optical_frame',
                    'r_forearm_link',
                    'r_gripper_palm_link',
                    'head_mount_kinect_ir_link',
                    'l_torso_lift_side_plate_link',
                    'r_upper_arm_link',
                    'r_gripper_r_finger_tip_link',
                    'l_forearm_cam_frame',
                    'projector_wg6802418_child_frame',
                    'l_forearm_cam_optical_frame',
                    'l_gripper_motor_screw_link',
                    'head_mount_kinect_rgb_optical_frame',
                    'base_laser_link',
                    'wide_stereo_l_stereo_camera_frame',
                    'r_forearm_cam_optical_frame',
                    'fl_caster_r_wheel_link',
                    'head_mount_kinect_ir_optical_frame',
                    'torso_lift_motor_screw_link',
                    'wide_stereo_r_stereo_camera_optical_frame',
                    'r_gripper_l_finger_link',
                    'l_shoulder_pan_link',
                    'l_gripper_l_finger_tip_link',
                    'l_force_torque_adapter_link',
                    'l_gripper_l_finger_tip_frame',
                    'base_footprint',
                    'l_elbow_flex_link',
                    'l_gripper_led_frame',
                    'l_gripper_r_finger_link',
                    'l_force_torque_link',
                    'odom_y_frame',
                    'narrow_stereo_optical_frame',
                    'br_caster_r_wheel_link',
                    'fr_caster_l_wheel_link',
                    'br_caster_l_wheel_link',
                    'l_gripper_r_finger_tip_link',
                    'r_shoulder_lift_link',
                    'narrow_stereo_l_stereo_camera_frame',
                    'l_wrist_roll_link',
                    'l_forearm_link',
                    'l_forearm_roll_link',
                    'sensor_mount_link',
                    'r_wrist_roll_link',
                    'base_link',
                    'l_gripper_motor_accelerometer_link',
                    'wide_stereo_optical_frame',
                    'head_mount_kinect2_ir_optical_frame',
                    'l_gripper_tool_frame',
                    'odom_combined',
                    'r_forearm_roll_link',
                    'head_pan_link',
                    'imu_link',
                    'head_mount_prosilica_link',
                    'bl_caster_r_wheel_link',
                    'head_mount_prosilica_optical_frame',
                    'fl_caster_rotation_link',
                    'wide_stereo_l_stereo_camera_optical_frame',
                    'r_gripper_led_frame',
                    'r_gripper_l_finger_tip_frame',
                    'odom_x_frame',
                    'l_gripper_l_finger_link',
                    'fr_caster_rotation_link',
                    'torso_lift_link',
                    'fr_caster_r_wheel_link',
                    'narrow_stereo_l_stereo_camera_optical_frame',
                    'r_shoulder_pan_link',
                    'wide_stereo_link',
                    'l_upper_arm_roll_link',
                    'l_gripper_motor_slider_link',
                    'double_stereo_link',
                    'r_elbow_flex_link',
                    'r_gripper_motor_accelerometer_link',
                    'bl_caster_l_wheel_link',
                    'laser_tilt_mount_link',
                    'narrow_stereo_link',
                    'r_gripper_motor_slider_link',
                    'r_forearm_cam_frame',
                    'bl_caster_rotation_link',
                    'head_plate_frame',
                    'narrow_stereo_r_stereo_camera_optical_frame',
                    'high_def_optical_frame',
                    'head_mount_kinect_rgb_link',
                    'r_upper_arm_roll_link',
                    'r_gripper_r_finger_link',
                    'wide_stereo_r_stereo_camera_frame',
                    'r_gripper_motor_screw_link',
                    'r_gripper_tool_frame',
                    'r_wrist_flex_link'
                    }


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
        assert set(parsed_pr2.get_joint_names()) == set_of_pr2_joints

        assert set(parsed_pr2.get_link_names()) == set_of_pr2_links
        assert parsed_pr2.get_name() == u'pr2'

    def test_urdf_from_file2(self, function_setup):
        """
        :type parsed_pr2: tested_class
        """
        with open(u'urdfs/tiny_ball.urdf', u'r') as f:
            urdf_string = f.read()
        parsed_pr2 = self.cls(urdf_string)
        assert len(parsed_pr2.get_joint_names()) == 0
        assert len(parsed_pr2.get_link_names()) == 1
        assert parsed_pr2.get_name() == u'ball'

    def test_from_world_body_box(self, function_setup):
        wb = make_world_body_box()
        urdf_obj = self.cls.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert urdf_obj.get_urdf_link('box').collision
        assert urdf_obj.get_urdf_link('box').visual
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
        assert set(urdf_obj.get_joint_names()) == set_of_pr2_joints
        assert set(urdf_obj.get_link_names()) == set_of_pr2_links

    def test_from_world_body_mesh(self, function_setup):
        wb = make_world_body_cylinder()
        urdf_obj = self.cls.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert len(urdf_obj.get_joint_names()) == 0

    def test_get_parent_link_name(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        assert parsed_pr2.get_parent_link_of_link(u'l_gripper_tool_frame') == u'l_gripper_palm_link'

    def test_get_parent_joint_of_joint(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        assert parsed_pr2.get_parent_joint_of_joint(u'l_gripper_tool_joint') == u'l_gripper_palm_joint'

    def test_get_link_names_from_chain(self, function_setup):
        pass

    def test_get_links_from_sub_tree1(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        urdf_obj = parsed_pr2.get_sub_tree_at_joint(u'torso_lift_joint')
        assert set(urdf_obj.get_link_names()) == {'double_stereo_link',
                                                  'head_mount_kinect2_ir_optical_frame',
                                                  'head_mount_kinect2_rgb_optical_frame',
                                                  'head_mount_kinect_ir_link',
                                                  'head_mount_kinect_ir_optical_frame',
                                                  'head_mount_kinect_rgb_link',
                                                  'head_mount_kinect_rgb_optical_frame',
                                                  'head_mount_link',
                                                  'head_mount_prosilica_link',
                                                  'head_mount_prosilica_optical_frame',
                                                  'head_pan_link',
                                                  'head_plate_frame',
                                                  'head_tilt_link',
                                                  'high_def_frame',
                                                  'high_def_optical_frame',
                                                  'imu_link',
                                                  'l_elbow_flex_link',
                                                  'l_force_torque_adapter_link',
                                                  'l_force_torque_link',
                                                  'l_forearm_cam_frame',
                                                  'l_forearm_cam_optical_frame',
                                                  'l_forearm_link',
                                                  'l_forearm_roll_link',
                                                  'l_gripper_l_finger_link',
                                                  'l_gripper_l_finger_tip_frame',
                                                  'l_gripper_l_finger_tip_link',
                                                  'l_gripper_led_frame',
                                                  'l_gripper_motor_accelerometer_link',
                                                  'l_gripper_motor_screw_link',
                                                  'l_gripper_motor_slider_link',
                                                  'l_gripper_palm_link',
                                                  'l_gripper_r_finger_link',
                                                  'l_gripper_r_finger_tip_link',
                                                  'l_gripper_tool_frame',
                                                  'l_shoulder_lift_link',
                                                  'l_shoulder_pan_link',
                                                  'l_torso_lift_side_plate_link',
                                                  'l_upper_arm_link',
                                                  'l_upper_arm_roll_link',
                                                  'l_wrist_flex_link',
                                                  'l_wrist_roll_link',
                                                  'laser_tilt_link',
                                                  'laser_tilt_mount_link',
                                                  'narrow_stereo_l_stereo_camera_frame',
                                                  'narrow_stereo_l_stereo_camera_optical_frame',
                                                  'narrow_stereo_link',
                                                  'narrow_stereo_optical_frame',
                                                  'narrow_stereo_r_stereo_camera_frame',
                                                  'narrow_stereo_r_stereo_camera_optical_frame',
                                                  'projector_wg6802418_child_frame',
                                                  'projector_wg6802418_frame',
                                                  'r_elbow_flex_link',
                                                  'r_forearm_cam_frame',
                                                  'r_forearm_cam_optical_frame',
                                                  'r_forearm_link',
                                                  'r_forearm_roll_link',
                                                  'r_gripper_l_finger_link',
                                                  'r_gripper_l_finger_tip_frame',
                                                  'r_gripper_l_finger_tip_link',
                                                  'r_gripper_led_frame',
                                                  'r_gripper_motor_accelerometer_link',
                                                  'r_gripper_motor_screw_link',
                                                  'r_gripper_motor_slider_link',
                                                  'r_gripper_palm_link',
                                                  'r_gripper_r_finger_link',
                                                  'r_gripper_r_finger_tip_link',
                                                  'r_gripper_tool_frame',
                                                  'r_shoulder_lift_link',
                                                  'r_shoulder_pan_link',
                                                  'r_torso_lift_side_plate_link',
                                                  'r_upper_arm_link',
                                                  'r_upper_arm_roll_link',
                                                  'r_wrist_flex_link',
                                                  'r_wrist_roll_link',
                                                  'sensor_mount_link',
                                                  'torso_lift_link',
                                                  'wide_stereo_l_stereo_camera_frame',
                                                  'wide_stereo_l_stereo_camera_optical_frame',
                                                  'wide_stereo_link',
                                                  'wide_stereo_optical_frame',
                                                  'wide_stereo_r_stereo_camera_frame',
                                                  'wide_stereo_r_stereo_camera_optical_frame'}
        assert set(urdf_obj.get_joint_names()) == {'double_stereo_frame_joint',
                                                   'head_mount_joint',
                                                   'head_mount_kinect2_ir_optical_frame_joint',
                                                   'head_mount_kinect2_rgb_optical_frame_joint',
                                                   'head_mount_kinect_ir_joint',
                                                   'head_mount_kinect_ir_optical_frame_joint',
                                                   'head_mount_kinect_rgb_joint',
                                                   'head_mount_kinect_rgb_optical_frame_joint',
                                                   'head_mount_prosilica_joint',
                                                   'head_mount_prosilica_optical_frame_joint',
                                                   'head_pan_joint',
                                                   'head_plate_frame_joint',
                                                   'head_tilt_joint',
                                                   'high_def_frame_joint',
                                                   'high_def_optical_frame_joint',
                                                   'imu_joint',
                                                   'l_elbow_flex_joint',
                                                   'l_force_torque_adapter_joint',
                                                   'l_force_torque_joint',
                                                   'l_forearm_cam_frame_joint',
                                                   'l_forearm_cam_optical_frame_joint',
                                                   'l_forearm_joint',
                                                   'l_forearm_roll_joint',
                                                   'l_gripper_joint',
                                                   'l_gripper_l_finger_joint',
                                                   'l_gripper_l_finger_tip_joint',
                                                   'l_gripper_led_joint',
                                                   'l_gripper_motor_accelerometer_joint',
                                                   'l_gripper_motor_screw_joint',
                                                   'l_gripper_motor_slider_joint',
                                                   'l_gripper_palm_joint',
                                                   'l_gripper_r_finger_joint',
                                                   'l_gripper_r_finger_tip_joint',
                                                   'l_gripper_tool_joint',
                                                   'l_shoulder_lift_joint',
                                                   'l_shoulder_pan_joint',
                                                   'l_torso_lift_side_plate_joint',
                                                   'l_upper_arm_joint',
                                                   'l_upper_arm_roll_joint',
                                                   'l_wrist_flex_joint',
                                                   'l_wrist_roll_joint',
                                                   'laser_tilt_joint',
                                                   'laser_tilt_mount_joint',
                                                   'narrow_stereo_frame_joint',
                                                   'narrow_stereo_l_stereo_camera_frame_joint',
                                                   'narrow_stereo_l_stereo_camera_optical_frame_joint',
                                                   'narrow_stereo_optical_frame_joint',
                                                   'narrow_stereo_r_stereo_camera_frame_joint',
                                                   'narrow_stereo_r_stereo_camera_optical_frame_joint',
                                                   'projector_wg6802418_child_frame_joint',
                                                   'projector_wg6802418_frame_joint',
                                                   'r_elbow_flex_joint',
                                                   'r_forearm_cam_frame_joint',
                                                   'r_forearm_cam_optical_frame_joint',
                                                   'r_forearm_joint',
                                                   'r_forearm_roll_joint',
                                                   'r_gripper_joint',
                                                   'r_gripper_l_finger_joint',
                                                   'r_gripper_l_finger_tip_joint',
                                                   'r_gripper_led_joint',
                                                   'r_gripper_motor_accelerometer_joint',
                                                   'r_gripper_motor_screw_joint',
                                                   'r_gripper_motor_slider_joint',
                                                   'r_gripper_palm_joint',
                                                   'r_gripper_r_finger_joint',
                                                   'r_gripper_r_finger_tip_joint',
                                                   'r_gripper_tool_joint',
                                                   'r_shoulder_lift_joint',
                                                   'r_shoulder_pan_joint',
                                                   'r_torso_lift_side_plate_joint',
                                                   'r_upper_arm_joint',
                                                   'r_upper_arm_roll_joint',
                                                   'r_wrist_flex_joint',
                                                   'r_wrist_roll_joint',
                                                   'sensor_mount_frame_joint',
                                                   'wide_stereo_frame_joint',
                                                   'wide_stereo_l_stereo_camera_frame_joint',
                                                   'wide_stereo_l_stereo_camera_optical_frame_joint',
                                                   'wide_stereo_optical_frame_joint',
                                                   'wide_stereo_r_stereo_camera_frame_joint',
                                                   'wide_stereo_r_stereo_camera_optical_frame_joint'}

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
        assert set(parsed_pr2.get_link_names()) == {'base_bellow_link',
                                                    'base_footprint',
                                                    'base_laser_link',
                                                    'base_link',
                                                    'bl_caster_l_wheel_link',
                                                    'bl_caster_r_wheel_link',
                                                    'bl_caster_rotation_link',
                                                    'br_caster_l_wheel_link',
                                                    'br_caster_r_wheel_link',
                                                    'br_caster_rotation_link',
                                                    'double_stereo_link',
                                                    'fl_caster_l_wheel_link',
                                                    'fl_caster_r_wheel_link',
                                                    'fl_caster_rotation_link',
                                                    'fr_caster_l_wheel_link',
                                                    'fr_caster_r_wheel_link',
                                                    'fr_caster_rotation_link',
                                                    'head_mount_kinect2_ir_optical_frame',
                                                    'head_mount_kinect2_rgb_optical_frame',
                                                    'head_mount_kinect_ir_link',
                                                    'head_mount_kinect_ir_optical_frame',
                                                    'head_mount_kinect_rgb_link',
                                                    'head_mount_kinect_rgb_optical_frame',
                                                    'head_mount_link',
                                                    'head_mount_prosilica_link',
                                                    'head_mount_prosilica_optical_frame',
                                                    'head_pan_link',
                                                    'head_plate_frame',
                                                    'head_tilt_link',
                                                    'high_def_frame',
                                                    'high_def_optical_frame',
                                                    'imu_link',
                                                    'l_torso_lift_side_plate_link',
                                                    'laser_tilt_link',
                                                    'laser_tilt_mount_link',
                                                    'narrow_stereo_l_stereo_camera_frame',
                                                    'narrow_stereo_l_stereo_camera_optical_frame',
                                                    'narrow_stereo_link',
                                                    'narrow_stereo_optical_frame',
                                                    'narrow_stereo_r_stereo_camera_frame',
                                                    'narrow_stereo_r_stereo_camera_optical_frame',
                                                    'odom_combined',
                                                    'odom_x_frame',
                                                    'odom_y_frame',
                                                    'projector_wg6802418_child_frame',
                                                    'projector_wg6802418_frame',
                                                    'r_elbow_flex_link',
                                                    'r_forearm_cam_frame',
                                                    'r_forearm_cam_optical_frame',
                                                    'r_forearm_link',
                                                    'r_forearm_roll_link',
                                                    'r_gripper_l_finger_link',
                                                    'r_gripper_l_finger_tip_frame',
                                                    'r_gripper_l_finger_tip_link',
                                                    'r_gripper_led_frame',
                                                    'r_gripper_motor_accelerometer_link',
                                                    'r_gripper_motor_screw_link',
                                                    'r_gripper_motor_slider_link',
                                                    'r_gripper_palm_link',
                                                    'r_gripper_r_finger_link',
                                                    'r_gripper_r_finger_tip_link',
                                                    'r_gripper_tool_frame',
                                                    'r_shoulder_lift_link',
                                                    'r_shoulder_pan_link',
                                                    'r_torso_lift_side_plate_link',
                                                    'r_upper_arm_link',
                                                    'r_upper_arm_roll_link',
                                                    'r_wrist_flex_link',
                                                    'r_wrist_roll_link',
                                                    'sensor_mount_link',
                                                    'torso_lift_link',
                                                    'torso_lift_motor_screw_link',
                                                    'wide_stereo_l_stereo_camera_frame',
                                                    'wide_stereo_l_stereo_camera_optical_frame',
                                                    'wide_stereo_link',
                                                    'wide_stereo_optical_frame',
                                                    'wide_stereo_r_stereo_camera_frame',
                                                    'wide_stereo_r_stereo_camera_optical_frame'}
        assert set(parsed_pr2.get_joint_names()) == {'base_bellow_joint',
                                                     'base_footprint_joint',
                                                     'base_laser_joint',
                                                     'bl_caster_l_wheel_joint',
                                                     'bl_caster_r_wheel_joint',
                                                     'bl_caster_rotation_joint',
                                                     'br_caster_l_wheel_joint',
                                                     'br_caster_r_wheel_joint',
                                                     'br_caster_rotation_joint',
                                                     'double_stereo_frame_joint',
                                                     'fl_caster_l_wheel_joint',
                                                     'fl_caster_r_wheel_joint',
                                                     'fl_caster_rotation_joint',
                                                     'fr_caster_l_wheel_joint',
                                                     'fr_caster_r_wheel_joint',
                                                     'fr_caster_rotation_joint',
                                                     'head_mount_joint',
                                                     'head_mount_kinect2_ir_optical_frame_joint',
                                                     'head_mount_kinect2_rgb_optical_frame_joint',
                                                     'head_mount_kinect_ir_joint',
                                                     'head_mount_kinect_ir_optical_frame_joint',
                                                     'head_mount_kinect_rgb_joint',
                                                     'head_mount_kinect_rgb_optical_frame_joint',
                                                     'head_mount_prosilica_joint',
                                                     'head_mount_prosilica_optical_frame_joint',
                                                     'head_pan_joint',
                                                     'head_plate_frame_joint',
                                                     'head_tilt_joint',
                                                     'high_def_frame_joint',
                                                     'high_def_optical_frame_joint',
                                                     'imu_joint',
                                                     'l_torso_lift_side_plate_joint',
                                                     'laser_tilt_joint',
                                                     'laser_tilt_mount_joint',
                                                     'narrow_stereo_frame_joint',
                                                     'narrow_stereo_l_stereo_camera_frame_joint',
                                                     'narrow_stereo_l_stereo_camera_optical_frame_joint',
                                                     'narrow_stereo_optical_frame_joint',
                                                     'narrow_stereo_r_stereo_camera_frame_joint',
                                                     'narrow_stereo_r_stereo_camera_optical_frame_joint',
                                                     'odom_x_joint',
                                                     'odom_y_joint',
                                                     'odom_z_joint',
                                                     'projector_wg6802418_child_frame_joint',
                                                     'projector_wg6802418_frame_joint',
                                                     'r_elbow_flex_joint',
                                                     'r_forearm_cam_frame_joint',
                                                     'r_forearm_cam_optical_frame_joint',
                                                     'r_forearm_joint',
                                                     'r_forearm_roll_joint',
                                                     'r_gripper_joint',
                                                     'r_gripper_l_finger_joint',
                                                     'r_gripper_l_finger_tip_joint',
                                                     'r_gripper_led_joint',
                                                     'r_gripper_motor_accelerometer_joint',
                                                     'r_gripper_motor_screw_joint',
                                                     'r_gripper_motor_slider_joint',
                                                     'r_gripper_palm_joint',
                                                     'r_gripper_r_finger_joint',
                                                     'r_gripper_r_finger_tip_joint',
                                                     'r_gripper_tool_joint',
                                                     'r_shoulder_lift_joint',
                                                     'r_shoulder_pan_joint',
                                                     'r_torso_lift_side_plate_joint',
                                                     'r_upper_arm_joint',
                                                     'r_upper_arm_roll_joint',
                                                     'r_wrist_flex_joint',
                                                     'r_wrist_roll_joint',
                                                     'sensor_mount_frame_joint',
                                                     'torso_lift_joint',
                                                     'torso_lift_motor_screw_joint',
                                                     'wide_stereo_frame_joint',
                                                     'wide_stereo_l_stereo_camera_frame_joint',
                                                     'wide_stereo_l_stereo_camera_optical_frame_joint',
                                                     'wide_stereo_optical_frame_joint',
                                                     'wide_stereo_r_stereo_camera_frame_joint',
                                                     'wide_stereo_r_stereo_camera_optical_frame_joint'}

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
        p.position = Point(0.0, 0.0, 0.0)
        p.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        original_links = parsed_pr2.get_link_names()
        original_joints = parsed_pr2.get_joint_names()
        # original_urdf = parsed_pr2.get_urdf_str()
        parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
        parsed_pr2.detach_sub_tree(u'box')
        assert set(original_links) == set(parsed_pr2.get_link_names())
        assert set(original_joints) == set(parsed_pr2.get_joint_names())
        # assert original_urdf == parsed_pr2.get_urdf_str()

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

    def test_get_all_joint_limits(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        assert parsed_pr2.get_all_joint_limits() == {'bl_caster_l_wheel_joint': (None, None),
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

    def test_get_joint_limits2(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        lower_limit, upper_limit = parsed_pr2.get_joint_position_limits(u'l_shoulder_pan_joint')
        assert lower_limit == -0.564601836603
        assert upper_limit == 2.1353981634

    def test_get_joint_limits3(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        lower_limit, upper_limit = parsed_pr2.get_joint_position_limits(u'l_wrist_roll_joint')
        assert lower_limit == None
        assert upper_limit == None

    def test_get_joint_limits4(self, function_setup):
        parsed_base_bot = self.cls(base_bot_urdf())
        lower_limit, upper_limit = parsed_base_bot.get_joint_position_limits(u'joint_x')
        assert lower_limit == -3
        assert upper_limit == 3

    def get_movable_parent(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())

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
        assert chain == [u'odom_x_joint',
                         u'odom_y_joint',
                         u'odom_z_joint',
                         u'base_footprint_joint',
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
                         u'base_footprint_joint',
                         u'odom_z_joint',
                         u'odom_y_joint',
                         u'odom_x_joint', ]

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
        parsed_donbot = self.cls(donbot_urdf())
        chain = parsed_donbot.get_chain('odom', 'odom_x_frame', joints=False, fixed=False)
        assert chain == parsed_donbot._urdf_robot.get_chain('odom', 'odom_x_frame', joints=False, fixed=False)

    def test_get_chain_fixed_joints2(self, function_setup):
        parsed_donbot = self.cls(donbot_urdf())
        chain = parsed_donbot.get_chain('odom', 'gripper_gripper_right_link', joints=True, fixed=False)
        assert chain == parsed_donbot._urdf_robot.get_chain('odom', 'gripper_gripper_right_link', joints=True,
                                                            fixed=False)

    def test_get_chain_joints_false1(self, function_setup):
        parsed_donbot = self.cls(donbot_urdf())
        chain = parsed_donbot.get_chain('odom', 'odom_x_frame', joints=False)
        assert chain == parsed_donbot._urdf_robot.get_chain('odom', 'odom_x_frame', joints=False)

    def test_get_chain_joints_false2(self, function_setup):
        parsed_donbot = self.cls(donbot_urdf())
        chain = parsed_donbot.get_chain('base_link', 'plate', joints=False)
        assert chain == ['base_link', 'plate']

    def test_get_chain_1(self, function_setup):
        parsed_donbot = self.cls(donbot_urdf())
        chain = parsed_donbot.get_chain('odom', 'gripper_tool_frame')
        assert chain == parsed_donbot._urdf_robot.get_chain('odom', 'gripper_tool_frame')

    def test_get_chain_2(self, function_setup):
        parsed_donbot = self.cls(donbot_urdf())
        chain = parsed_donbot.get_chain('base_link', 'plate')
        assert chain == ['base_link', 'plate_joint', 'plate']

    def test_get_leaves(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        leaves = parsed_pr2.get_leaves()
        assert len(leaves) == 40

    def test_get_controllable_parent_joint(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        joint = parsed_pr2.get_movable_parent_joint(u'r_gripper_tool_frame')
        assert joint == u'r_wrist_roll_joint'

    def test_get_link_names_from_joint_chain(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        result = parsed_pr2.get_link_names_from_joint_chain(u'odom_x_joint', u'odom_y_joint')
        assert result == [u'odom_x_frame']

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
                    u'l_gripper_r_finger_joint': {u'l_gripper_r_finger_link', u'l_gripper_r_finger_tip_link'},
                    u'odom_x_joint': {u'l_upper_arm_link', u'l_shoulder_lift_link', u'r_gripper_l_finger_tip_link',
                                      u'fl_caster_l_wheel_link', u'br_caster_rotation_link', u'base_bellow_link',
                                      u'head_tilt_link', u'l_gripper_palm_link', u'l_wrist_flex_link',
                                      u'r_forearm_link', u'r_gripper_palm_link', u'r_upper_arm_link',
                                      u'r_gripper_r_finger_tip_link', u'fl_caster_r_wheel_link',
                                      u'r_gripper_l_finger_link', u'l_shoulder_pan_link',
                                      u'l_gripper_l_finger_tip_link', u'l_elbow_flex_link', u'l_gripper_r_finger_link',
                                      u'br_caster_r_wheel_link', u'fr_caster_l_wheel_link', u'br_caster_l_wheel_link',
                                      u'l_gripper_r_finger_tip_link', u'r_shoulder_lift_link', u'l_wrist_roll_link',
                                      u'l_forearm_link', u'l_forearm_roll_link', u'r_wrist_roll_link', u'base_link',
                                      u'r_forearm_roll_link', u'head_pan_link', u'bl_caster_r_wheel_link',
                                      u'fl_caster_rotation_link', u'l_gripper_l_finger_link',
                                      u'fr_caster_rotation_link', u'torso_lift_link', u'fr_caster_r_wheel_link',
                                      u'r_shoulder_pan_link', u'l_upper_arm_roll_link', u'r_elbow_flex_link',
                                      u'bl_caster_l_wheel_link', u'laser_tilt_mount_link', u'bl_caster_rotation_link',
                                      u'head_plate_frame', u'r_upper_arm_roll_link', u'r_gripper_r_finger_link',
                                      u'r_wrist_flex_link'
                                      },
                    u'odom_y_joint': {u'l_upper_arm_link', u'l_shoulder_lift_link', u'r_gripper_l_finger_tip_link',
                                      u'fl_caster_l_wheel_link', u'br_caster_rotation_link', u'base_bellow_link',
                                      u'head_tilt_link', u'l_gripper_palm_link', u'l_wrist_flex_link',
                                      u'r_forearm_link', u'r_gripper_palm_link', u'r_upper_arm_link',
                                      u'r_gripper_r_finger_tip_link', u'fl_caster_r_wheel_link',
                                      u'r_gripper_l_finger_link', u'l_shoulder_pan_link',
                                      u'l_gripper_l_finger_tip_link', u'l_elbow_flex_link', u'l_gripper_r_finger_link',
                                      u'br_caster_r_wheel_link', u'fr_caster_l_wheel_link', u'br_caster_l_wheel_link',
                                      u'l_gripper_r_finger_tip_link', u'r_shoulder_lift_link', u'l_wrist_roll_link',
                                      u'l_forearm_link', u'l_forearm_roll_link', u'r_wrist_roll_link', u'base_link',
                                      u'r_forearm_roll_link', u'head_pan_link', u'bl_caster_r_wheel_link',
                                      u'fl_caster_rotation_link', u'l_gripper_l_finger_link',
                                      u'fr_caster_rotation_link', u'torso_lift_link', u'fr_caster_r_wheel_link',
                                      u'r_shoulder_pan_link', u'l_upper_arm_roll_link', u'r_elbow_flex_link',
                                      u'bl_caster_l_wheel_link', u'laser_tilt_mount_link', u'bl_caster_rotation_link',
                                      u'head_plate_frame', u'r_upper_arm_roll_link', u'r_gripper_r_finger_link',
                                      u'r_wrist_flex_link'
                                      },
                    u'odom_z_joint': {u'l_upper_arm_link', u'l_shoulder_lift_link', u'r_gripper_l_finger_tip_link',
                                      u'fl_caster_l_wheel_link', u'br_caster_rotation_link', u'base_bellow_link',
                                      u'head_tilt_link', u'l_gripper_palm_link', u'l_wrist_flex_link',
                                      u'r_forearm_link', u'r_gripper_palm_link', u'r_upper_arm_link',
                                      u'r_gripper_r_finger_tip_link', u'fl_caster_r_wheel_link',
                                      u'r_gripper_l_finger_link', u'l_shoulder_pan_link',
                                      u'l_gripper_l_finger_tip_link', u'l_elbow_flex_link', u'l_gripper_r_finger_link',
                                      u'br_caster_r_wheel_link', u'fr_caster_l_wheel_link', u'br_caster_l_wheel_link',
                                      u'l_gripper_r_finger_tip_link', u'r_shoulder_lift_link', u'l_wrist_roll_link',
                                      u'l_forearm_link', u'l_forearm_roll_link', u'r_wrist_roll_link', u'base_link',
                                      u'r_forearm_roll_link', u'head_pan_link', u'bl_caster_r_wheel_link',
                                      u'fl_caster_rotation_link', u'l_gripper_l_finger_link',
                                      u'fr_caster_rotation_link', u'torso_lift_link', u'fr_caster_r_wheel_link',
                                      u'r_shoulder_pan_link', u'l_upper_arm_roll_link', u'r_elbow_flex_link',
                                      u'bl_caster_l_wheel_link', u'laser_tilt_mount_link', u'bl_caster_rotation_link',
                                      u'r_gripper_r_finger_link', u'r_upper_arm_roll_link', u'r_wrist_flex_link',
                                      u'head_plate_frame', }}
        for joint in parsed_pr2.get_joint_names_controllable():
            diff = set(parsed_pr2.get_sub_tree_link_names_with_collision(joint)).difference(expected[joint])
            assert diff == set(), u'diff for joint {} is not empty: {}'.format(joint, diff)

    def test_get_sub_tree_link_names_with_collision_donbot(self, function_setup):
        parsed_donbot = self.cls(donbot_urdf())
        expected = {u'ur5_wrist_3_joint': {u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                           u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                           u'gripper_gripper_right_link', u'gripper_finger_right_link',
                                           u'camera_holder_link', u'wrist_collision'},
                    u'ur5_elbow_joint': {u'ur5_forearm_link', u'ur5_wrist_1_link', u'ur5_wrist_2_link',
                                         u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                         u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                         u'gripper_gripper_right_link', u'gripper_finger_right_link',
                                         u'camera_holder_link', u'wrist_collision'},
                    u'ur5_wrist_1_joint': {u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link',
                                           u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link',
                                           u'gripper_finger_left_link', u'gripper_gripper_right_link',
                                           u'gripper_finger_right_link',
                                           u'camera_holder_link', u'wrist_collision'},
                    u'odom_z_joint': {u'base_link', u'plate', u'ur5_base_link', u'ur5_shoulder_link',
                                      u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link',
                                      u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                      u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                      u'gripper_gripper_right_link', u'gripper_finger_right_link',
                                      u'camera_holder_link', u'wrist_collision', u'switches', u'charger', u'wlan',
                                      u'ur5_touchpad', u'e_stop'},
                    u'ur5_shoulder_lift_joint': {u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link',
                                                 u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link',
                                                 u'gripper_base_link', u'gripper_gripper_left_link',
                                                 u'gripper_finger_left_link', u'gripper_gripper_right_link',
                                                 u'gripper_finger_right_link',
                                                 u'camera_holder_link', u'wrist_collision'},
                    u'odom_y_joint': {u'base_link', u'plate', u'ur5_base_link', u'ur5_shoulder_link',
                                      u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link',
                                      u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                      u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                      u'gripper_gripper_right_link', u'gripper_finger_right_link',
                                      u'camera_holder_link', u'wrist_collision', u'switches', u'charger', u'wlan',
                                      u'ur5_touchpad', u'e_stop'},
                    u'ur5_wrist_2_joint': {u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link',
                                           u'gripper_base_link', u'gripper_gripper_left_link',
                                           u'gripper_finger_left_link', u'gripper_gripper_right_link',
                                           u'gripper_finger_right_link',
                                           u'camera_holder_link', u'wrist_collision'},
                    u'odom_x_joint': {u'base_link', u'plate', u'ur5_base_link', u'ur5_shoulder_link',
                                      u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link',
                                      u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link',
                                      u'gripper_gripper_left_link', u'gripper_finger_left_link',
                                      u'gripper_gripper_right_link', u'gripper_finger_right_link',
                                      u'camera_holder_link', u'wrist_collision', u'switches', u'charger', u'wlan',
                                      u'ur5_touchpad', u'e_stop'},
                    u'ur5_shoulder_pan_joint': {u'ur5_shoulder_link', u'ur5_upper_arm_link', u'ur5_forearm_link',
                                                u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link',
                                                u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link',
                                                u'gripper_finger_left_link', u'gripper_gripper_right_link',
                                                u'gripper_finger_right_link',
                                                u'camera_holder_link', u'wrist_collision'},
                    u'gripper_joint': {u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'refills_finger_joint': {}}
        for joint in parsed_donbot.get_joint_names_controllable():
            assert set(parsed_donbot.get_sub_tree_link_names_with_collision(joint)).difference(expected[joint]) == set()

    def test_has_link_collision(self, function_setup):
        pass

    def test_get_first_link_with_collision(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        assert parsed_pr2.get_first_child_links_with_collision(u'odom_combined') == u'base_link'

    def test_get_non_base_movement_root(self, function_setup):
        parsed_donbot = self.cls(donbot_urdf())
        assert parsed_donbot.get_non_base_movement_root() == u'base_footprint'

    def test_get_non_base_movement_root2(self, function_setup):
        parsed_pr2 = self.cls(pr2_urdf())
        assert parsed_pr2.get_non_base_movement_root() == u'base_footprint'
