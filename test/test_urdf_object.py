import pytest
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from giskard_msgs.msg import WorldBody

from giskardpy.exceptions import DuplicateNameException
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import make_world_body_box, make_world_body_sphere, make_world_body_cylinder, make_urdf_world_body


@pytest.fixture()
def parsed_pr2():
    """
    :rtype: URDFObject
    """
    return URDFObject.from_urdf_file(u'urdfs/pr2.urdf')


@pytest.fixture()
def parsed_base_bot():
    """
    :rtype: URDFObject
    """
    return URDFObject.from_urdf_file(u'urdfs/2d_base_bot.urdf')

def pr2_urdf():
    with open(u'urdfs/pr2.urdf', u'r') as f:
        urdf_string = f.read()
    return urdf_string

class TestUrdfObject(object):
    def test_urdf_from_str(self, parsed_pr2):
        pr2 = URDFObject(pr2_urdf())
        assert pr2 == parsed_pr2

    def test_urdf_from_file(self, parsed_pr2):
        """
        :type parsed_pr2: URDFObject
        """
        assert len(parsed_pr2.get_joint_names()) == 96
        assert len(parsed_pr2.get_link_names()) == 97
        assert parsed_pr2.get_name() == u'pr2'

    def test_from_world_body_box(self):
        wb = make_world_body_box()
        urdf_obj = URDFObject.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert len(urdf_obj.get_joint_names()) == 0

    def test_from_world_body_sphere(self):
        wb = make_world_body_sphere()
        urdf_obj = URDFObject.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert len(urdf_obj.get_joint_names()) == 0

    def test_from_world_body_cylinder(self):
        wb = make_world_body_cylinder()
        urdf_obj = URDFObject.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert len(urdf_obj.get_joint_names()) == 0

    def test_from_world_body_cone(self):
        pass

    def test_from_world_body_invalid_primitive_type(self):
        pass

    def test_form_world_body_unsupported_type(self):
        pass

    def test_from_world_body_urdf(self):
        wb = make_urdf_world_body(u'pr2', pr2_urdf())
        urdf_obj = URDFObject.from_world_body(wb)
        assert len(urdf_obj.get_joint_names()) == 96
        assert len(urdf_obj.get_link_names()) == 97

    def test_from_world_body_mesh(self):
        wb = make_world_body_cylinder()
        urdf_obj = URDFObject.from_world_body(wb)
        assert len(urdf_obj.get_link_names()) == 1
        assert len(urdf_obj.get_joint_names()) == 0

    def test_get_parent_link_name(self, parsed_pr2):
        assert parsed_pr2.get_parent_link_name(u'l_gripper_tool_frame') == u'l_gripper_palm_link'

    def test_get_link_names_from_chain(self, parsed_pr2):
        pass

    def test_get_links_from_sub_tree1(self, parsed_pr2):
        urdf_obj = parsed_pr2.get_sub_tree_at_joint(u'torso_lift_joint')
        assert len(urdf_obj.get_link_names()) == 80
        assert len(urdf_obj.get_joint_names()) == 79

    def test_attach_urdf_object1(self, parsed_pr2):
        num_of_links_before = len(parsed_pr2.get_link_names())
        num_of_joints_before = len(parsed_pr2.get_joint_names())
        link_chain_before = len(parsed_pr2.get_links_from_sub_tree(u'torso_lift_joint'))
        box = URDFObject.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
        assert box.get_name() in parsed_pr2.get_link_names()
        assert len(parsed_pr2.get_link_names()) == num_of_links_before + 1
        assert len(parsed_pr2.get_joint_names()) == num_of_joints_before + 1
        assert len(parsed_pr2.get_links_from_sub_tree(u'torso_lift_joint')) == link_chain_before + 1

    def test_attach_urdf_object2(self, parsed_base_bot):
        links_before = set(parsed_base_bot.get_link_names())
        joints_before = set(parsed_base_bot.get_joint_names())
        pr2 = make_urdf_world_body(u'pr2', pr2_urdf())
        pr2_obj = URDFObject.from_world_body(pr2)
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        parsed_base_bot.attach_urdf_object(pr2_obj, u'eef', p)
        assert pr2_obj.get_root() in parsed_base_bot.get_link_names()
        assert set(parsed_base_bot.get_link_names()).difference(links_before.union(set(pr2_obj.get_link_names()))) == set()
        assert set(parsed_base_bot.get_joint_names()).difference(joints_before.union(set(pr2_obj.get_joint_names()))) == {u'pr2'}

    def test_attach_twice(self, parsed_pr2):
        box = URDFObject.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
        try:
            parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
            assert False, u'didnt get expected exception'
        except DuplicateNameException as e:
            assert True

    def test_attach_to_non_existing_link(self, parsed_base_bot):
        box = URDFObject.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        try:
            parsed_base_bot.attach_urdf_object(box, u'muh', p)
            assert False, u'didnt get expected exception'
        except KeyError:
            assert True

    def test_attach_obj_with_joint_name(self, parsed_base_bot):
        box = URDFObject.from_world_body(make_world_body_box(u'rot_z'))
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        try:
            parsed_base_bot.attach_urdf_object(box, u'eef', p)
            assert False, u'didnt get expected exception'
        except DuplicateNameException as e:
            assert True

    def test_attach_obj_with_link_name(self, parsed_base_bot):
        box = URDFObject.from_world_body(make_world_body_box(u'eef'))
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        try:
            parsed_base_bot.attach_urdf_object(box, u'eef', p)
            assert False, u'didnt get expected exception'
        except DuplicateNameException as e:
            assert True

    def test_detach_object(self, parsed_base_bot):
        """
        :type parsed_base_bot: URDFObject
        """
        parsed_base_bot.detach_sub_tree(u'rot_z')
        assert len(parsed_base_bot.get_link_names()) == 3
        assert len(parsed_base_bot.get_joint_names()) == 2
        assert u'rot_z' not in parsed_base_bot.get_joint_names()
        assert u'eef' not in parsed_base_bot.get_link_names()

    def test_amputate_left_arm(self, parsed_pr2):
        parsed_pr2.detach_sub_tree(u'l_shoulder_pan_joint')
        assert len(parsed_pr2.get_link_names()) == 73
        assert len(parsed_pr2.get_joint_names()) == 72

    def test_attach_detach(self, parsed_pr2):
        box = URDFObject.from_world_body(make_world_body_box())
        p = Pose()
        p.position = Point(0, 0, 0)
        p.orientation = Quaternion(0, 0, 0, 1)
        original_urdf = parsed_pr2.get_urdf()
        parsed_pr2.attach_urdf_object(box, u'l_gripper_tool_frame', p)
        parsed_pr2.detach_sub_tree(u'box')
        assert original_urdf == parsed_pr2.get_urdf()

    def test_detach_non_existing_object(self, parsed_pr2):
        try:
            parsed_pr2.detach_sub_tree(u'muh')
            assert False, u'didnt get expected exception'
        except KeyError:
            assert True

    def test_detach_at_link(self, parsed_pr2):
        try:
            parsed_pr2.detach_sub_tree(u'torso_lift_link')
            assert False, u'didnt get expected exception'
        except KeyError:
            assert True

    def test_get_joint_limits1(self, parsed_pr2):
        assert len(parsed_pr2.get_all_joint_limits()) == 45

    def test_get_joint_limits2(self, parsed_pr2):
        lower_limit, upper_limit = parsed_pr2.get_joint_limits(u'l_shoulder_pan_joint')
        assert lower_limit == -0.564601836603
        assert upper_limit == 2.1353981634

    def test_get_joint_limits3(self, parsed_pr2):
        lower_limit, upper_limit = parsed_pr2.get_joint_limits(u'l_wrist_roll_joint')
        assert lower_limit == None
        assert upper_limit == None

    def test_get_joint_limits4(self, parsed_base_bot):
        lower_limit, upper_limit = parsed_base_bot.get_joint_limits(u'joint_x')
        assert lower_limit == -3
        assert upper_limit == 3

    def test_get_joint_names_from_chain(self):
        pass

    def test_get_joint_names_from_chain_controllable(self):
        pass

    def test_get_joint_names_controllable(self):
        pass

    def test_is_joint_mimic(self):
        pass

    def test_is_joint_continuous(self):
        pass

    def test_get_joint_type(self):
        pass

    def test_is_joint_type_supported(self):
        pass

    def test_is_rotational_joint(self):
        pass

    def test_is_translational_joint(self):
        pass

    def test_get_sub_tree_link_names_with_collision(self):
        pass

    def test_has_link_collision(self):
        pass
