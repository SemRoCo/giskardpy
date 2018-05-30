import unittest
from collections import namedtuple, OrderedDict

from giskardpy.god_map import GodMap
from giskardpy.pybullet_world import PyBulletWorld
import pybullet as p

from giskardpy.trajectory import MultiJointState, SingleJointState

PKG = 'giskardpy'


class TestPyBulletWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = PyBulletWorld(False)
        cls.w.activate_viewer()
        cls.w.clear_world()
        super(TestPyBulletWorld, cls).setUpClass()

    def get_pr2_collision_js(self):
        js = [('head_pan_joint', -0.016552842705291115),
              ('head_tilt_joint', 0.7287556667322448),
              ('torso_lift_joint', 0.2901924421705799),
              ('l_elbow_flex_joint', -1.355750932685288),
              ('l_forearm_roll_joint', 0.9745881904684524),
              ('l_shoulder_lift_joint', 0.23908648260374907),
              ('l_shoulder_pan_joint', 0.2981063106986737),
              ('l_upper_arm_roll_joint', 0.8475079507641174),
              ('l_wrist_flex_joint', -1.020876123413078),
              ('l_wrist_roll_joint', 6.866114367042748),
              ('r_elbow_flex_joint', -1.3797725450953955),
              ('r_forearm_roll_joint', 5.461882478288552),
              ('r_shoulder_lift_joint', 0.24897300589444363),
              ('r_shoulder_pan_joint', -0.54341218738967),
              ('r_upper_arm_roll_joint', -0.8518325699325913),
              ('r_wrist_flex_joint', -1.1127555669432887),
              ('r_wrist_roll_joint', 2.3104656448272642),
              ('laser_tilt_mount_joint', 0.0),
              ('r_gripper_l_finger_joint', 0.0),
              ('r_gripper_r_finger_joint', 0.0),
              ('l_gripper_l_finger_joint', 0.0),
              ('l_gripper_r_finger_joint', 0.0),
              ('l_gripper_l_finger_tip_joint', 0.0),
              ('l_gripper_r_finger_tip_joint', 0.0),
              ('r_gripper_l_finger_tip_joint', 0.0),
              ('r_gripper_r_finger_tip_joint', 0.0),
              ('fl_caster_l_wheel_joint', 0.0),
              ('fl_caster_r_wheel_joint', 0.0),
              ('fr_caster_l_wheel_joint', 0.0),
              ('fr_caster_r_wheel_joint', 0.0),
              ('bl_caster_l_wheel_joint', 0.0),
              ('bl_caster_r_wheel_joint', 0.0),
              ('br_caster_l_wheel_joint', 0.0),
              ('br_caster_r_wheel_joint', 0.0),
              ('fl_caster_rotation_joint', 0.0),
              ('fr_caster_rotation_joint', 0.0),
              ('bl_caster_rotation_joint', 0.0),
              ('br_caster_rotation_joint', 0.0)]
        mjs = OrderedDict()
        for joint_name, joint_position in js:
            sjs = SingleJointState()
            sjs.name = joint_name
            sjs.position = joint_position
            mjs[joint_name] = sjs
        return mjs

    def tearDown(self):
        super(TestPyBulletWorld, self).tearDown()
        self.w.clear_world()

    def test_spawn_object1(self):
        self.w.add_ground_plane()
        self.assertEqual(p.getNumBodies(), 1)

    def test_delete1(self):
        self.w.add_ground_plane()
        self.w.delete_object('plane')
        self.assertEqual(p.getNumBodies(), 0)

    def test_clear_world1(self):
        self.w.clear_world()
        self.assertEqual(p.getNumBodies(), 0)

    def test_spawn_robot1(self):
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        self.assertEqual(p.getNumBodies(), 1)

    def test_delete_robot1(self):
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        self.assertEqual(p.getNumBodies(), 1)
        self.w.delete_robot('pr2')
        self.assertEqual(p.getNumBodies(), 0)

    def test_collision_detection1(self):
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        r = self.w.get_robot('pr2')
        self.w.set_joint_state('pr2', r.get_zero_joint_state())
        self.assertEqual(0, len(self.w.check_collisions()))

    def test_collision_detection2(self):
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        mjs = self.get_pr2_collision_js()
        self.w.set_joint_state('pr2', mjs)
        collisions = self.w.check_collisions()
        collisions = [x for x in collisions.values() if x.contact_distance < 0.05]
        self.assertEqual(8, len(collisions))

    def test_list_objects1(self):
        self.w.add_ground_plane()
        self.assertEqual(self.w.get_object_list(), ['plane'])

    def test_joint_state(self):
        mjs = self.get_pr2_collision_js()
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        self.w.set_joint_state('pr2', mjs)
        mjs2 = self.w.get_joint_state('pr2')
        for joint_name, sjs in mjs.items():
            self.assertEqual(sjs.position, mjs2.get(joint_name).position)


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestPyBulletWorld',
                    test=TestPyBulletWorld)
