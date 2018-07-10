import unittest
from bson.json_util import default
from collections import namedtuple, OrderedDict, defaultdict
from time import sleep

from giskardpy.god_map import GodMap
from giskardpy.pybullet_world import PyBulletWorld
import pybullet as p

from giskardpy.data_types import MultiJointState, SingleJointState

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
              ('r_wrist_roll_joint', 2.3104656448272642),]
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
        self.w.add_ground_plane()

    def test_spawn_object1(self):
        self.assertEqual(p.getNumBodies(), 1)

    def test_delete1(self):
        self.w.delete_object('plane')
        self.assertEqual(0, p.getNumBodies())

    def test_clear_world1(self):
        self.w.clear_world()
        self.assertEqual(1, p.getNumBodies())

    def test_spawn_robot1(self):
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        self.assertEqual(2, p.getNumBodies())

    def test_delete_robot1(self):
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        self.assertEqual(2, p.getNumBodies())
        self.w.delete_robot()
        self.assertEqual(1, p.getNumBodies())

    def test_collision_detection1(self):
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        r = self.w.get_robot()
        self.w.set_robot_joint_state(r.get_zero_joint_state())
        cut_off_distances = defaultdict(lambda : 0.05)
        collisions = self.w.check_collisions(cut_off_distances, self_collision_d=0.05)
        self.assertEqual(0, len(collisions), str(dict(collisions).keys()))

    def test_collision_detection2(self):
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        mjs = self.get_pr2_collision_js()
        self.w.set_robot_joint_state(mjs)
        cut_off_distances = defaultdict(lambda: 0.05)
        collisions = self.w.check_collisions(cut_off_distances, self_collision_d=0.05)
        self.assertEqual(2, len(collisions), str(dict(collisions).keys()))

    def test_list_objects1(self):
        self.w.add_ground_plane()
        self.assertEqual(self.w.get_object_list(), ['plane'])

    def test_joint_state(self):
        mjs = self.get_pr2_collision_js()
        self.w.spawn_robot_from_urdf_file('pr2', 'pr2.urdf')
        self.w.set_robot_joint_state(mjs)
        mjs2 = self.w.get_robot_joint_state()
        for joint_name, sjs in mjs.items():
            self.assertEqual(sjs.position, mjs2.get(joint_name).position)

    def test_set_object_joint_state(self):
        object_name = 'pr2'
        mjs = self.get_pr2_collision_js()
        self.w.spawn_object_from_urdf_file(object_name, 'pr2.urdf')
        self.w.set_object_joint_state(object_name, mjs)
        mjs2 = self.w.get_object_joint_state(object_name)
        for joint_name, sjs in mjs.items():
            self.assertEqual(sjs.position, mjs2.get(joint_name).position)




if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestPyBulletWorld',
                    test=TestPyBulletWorld)
