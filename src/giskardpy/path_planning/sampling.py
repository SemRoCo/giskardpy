import json
from copy import deepcopy

import numpy as np
from random import uniform

import rospy
import tf.transformations
import yaml
from py_trees import Status

from geometry_msgs.msg import Pose, PoseStamped
from giskard_msgs.msg import Constraint
from giskard_msgs.srv import GetAttachedObjectsRequest, GetAttachedObjects
from giskardpy import identifier
from giskardpy.path_planning.motion_validator import ObjectRayMotionValidator, SimpleRayMotionValidator
from giskardpy.path_planning.state_validator import GiskardRobotBulletCollisionChecker
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import pose_to_list, transform_pose, pose_diff
from giskardpy.utils.utils import convert_dictionary_to_ros_message, convert_ros_message_to_dictionary


class GoalRegionSampler:
    def __init__(self, is_3D, goal, validation_fun, heuristic_fun, goal_orientation=None, precision=5):
        self.goal = goal
        self.is_3D = is_3D
        self.precision = precision
        self.validation_fun = validation_fun
        self.h = heuristic_fun
        self.valid_samples = list()
        self.samples = list()
        self.goal_orientation = goal_orientation
        self.goal_list = pose_to_list(goal)[0]

    def _get_pitch(self, pos_b):
        pos_a = np.zeros(3)
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        dz = pos_b[2] - pos_a[2]
        pos_c = pos_a + np.array([dx, dy, 0])
        a = np.sqrt(np.sum((np.array(pos_c) - np.array(pos_a)) ** 2))
        return np.arctan2(dz, a)

    def get_pitch(self, pos_b):
        """ :returns: value in [-np.pi, np.pi] """
        l_a = self._get_pitch(pos_b)
        return -l_a

    def _get_yaw(self, pos_b):
        pos_a = np.zeros(3)
        dx = pos_b[0] - pos_a[0]
        dz = pos_b[2] - pos_a[2]
        pos_c = pos_a + np.array([0, 0, dz])
        pos_d = pos_c + np.array([dx, 0, 0])
        g = np.sqrt(np.sum((np.array(pos_b) - np.array(pos_d)) ** 2))
        a = np.sqrt(np.sum((np.array(pos_c) - np.array(pos_d)) ** 2))
        return np.arctan2(g, a)

    def get_yaw(self, pos_b):
        """ :returns: value in [-2.*np.pi, 2.*np.pi] """
        l_a = self._get_yaw(pos_b)
        if pos_b[0] >= 0 and pos_b[1] >= 0:
            return l_a
        elif pos_b[0] >= 0 and pos_b[1] <= 0:
            return -l_a
        elif pos_b[0] <= 0 and pos_b[1] >= 0:
            return np.pi - l_a
        else:
            return -np.pi + l_a  # todo: -np.pi + l_a

    def valid_sample(self, d=0.5, max_samples=100):
        if type(d) == list:
            x, y, z = d
        elif type(d) == float:
            x, y, z = [d] * 3
        else:
            raise Exception('ffffffffffffff')
        i = 0
        while i < max_samples:
            valid = False
            while not valid and i < max_samples:
                s = self._sample(i * x / max_samples, i * y / max_samples, i * z / max_samples)
                valid = self.validation_fun(s)
                i += 1
            self.valid_samples.append([self.h(s), s])
        self.valid_samples = sorted(self.valid_samples, key=lambda e: e[0])
        return self.valid_samples[0][1]

    def sample(self, d, samples=100):
        i = 0
        if type(d) == list and len(d) == 3:
            x, y, z = d
        elif type(d) == float:
            x, y, z = [d] * 3
        else:
            raise Exception(f'The given parameter d must either be a list of 3 floats or one float, however it is {d}.')
        while i < samples:
            s = self._sample(i * x / samples, i * y / samples, i * z / samples)
            self.samples.append([self.h(s), s])
            i += 1
        self.samples = sorted(self.samples, key=lambda e: e[0])
        return self.samples[0][1]

    def get_orientation(self, p):
        if self.goal_orientation is not None:
            return [self.goal_orientation.x,
                    self.goal_orientation.y,
                    self.goal_orientation.z,
                    self.goal_orientation.w]
        if self.is_3D:
            diff = [self.goal.position.x - p.position.x,
                    self.goal.position.y - p.position.y,
                    self.goal.position.z - p.position.z]
            pitch = self.get_pitch(diff)
            yaw = self.get_yaw(diff)
            return tf.transformations.quaternion_from_euler(0, pitch, yaw)
        else:
            diff = [self.goal.position.x - p.position.x,
                    self.goal.position.y - p.position.y,
                    0]
            yaw = self.get_yaw(diff)
            return tf.transformations.quaternion_from_euler(0, 0, yaw)

    def _sample(self, x, y, z):
        s = Pose()
        s.orientation.w = 1

        x = round(uniform(self.goal.position.x - x, self.goal.position.x + x), self.precision)
        y = round(uniform(self.goal.position.y - y, self.goal.position.y + y), self.precision)
        s.position.x = x
        s.position.y = y

        if self.is_3D:
            z = round(uniform(self.goal.position.z - z, self.goal.position.z + z), self.precision)
            s.position.z = z
            q = self.get_orientation(s)
            s.orientation.x = q[0]
            s.orientation.y = q[1]
            s.orientation.z = q[2]
            s.orientation.w = q[3]
        else:
            q = self.get_orientation(s)
            s.orientation.x = q[0]
            s.orientation.y = q[1]
            s.orientation.z = q[2]
            s.orientation.w = q[3]

        return s


class PreGraspSampler(GiskardBehavior):
    def __init__(self, name='PreGraspSampler'):
        super().__init__(name=name)
        self.is_3D = True

    def get_pregrasp_goal(self, cmd):
        try:
            return next(c for c in cmd.constraints if c.type in ['CartesianPreGrasp'])
        except StopIteration:
            return None

    def get_cart_goal(self, cart_c):

        pregrasp_pos = None
        self.__goal_dict = yaml.load(cart_c.parameter_value_pair)
        if 'goal_position' in self.__goal_dict:
            pregrasp_pos = convert_dictionary_to_ros_message(self.__goal_dict[u'goal_position'])
        dist = 0.0
        if 'dist' in self.__goal_dict:
            dist = self.__goal_dict['dist']
        ros_pose = convert_dictionary_to_ros_message(self.__goal_dict[u'grasping_goal'])
        self.grasping_object_name = None
        if 'grasping_object' in self.__goal_dict:
            self.grasping_object_name = self.__goal_dict['grasping_object']
        goal = transform_pose('map', ros_pose)  # type: PoseStamped
        self.goal_orientation = None
        if 'grasping_orientation' in self.__goal_dict:
            self.goal_orientation = convert_dictionary_to_ros_message(self.__goal_dict[u'grasping_orientation'])

        self.root_link = self.__goal_dict[u'root_link']
        tip_link = self.__goal_dict[u'tip_link']
        get_attached_objects = rospy.ServiceProxy('~get_attached_objects', GetAttachedObjects)
        if tip_link in get_attached_objects(GetAttachedObjectsRequest()).object_names:
            tip_link = self.get_robot().get_parent_link_of_link(tip_link)
        self.tip_link = tip_link
        link_names = self.robot.link_names

        if self.root_link not in link_names:
            raise Exception(u'Root_link {} is no known link of the robot.'.format(self.root_link))
        if self.tip_link not in link_names:
            raise Exception(u'Tip_link {} is no known link of the robot.'.format(self.tip_link))
        # if not self.robot.are_linked(self.root_link, self.tip_link):
        #    raise Exception(u'Did not found link chain of the robot from'
        #                    u' root_link {} to tip_link {}.'.format(self.root_link, self.tip_link))

        return goal, pregrasp_pos, dist

    def get_cartesian_pose_constraints(self, goal):

        d = dict()
        d[u'parameter_value_pair'] = deepcopy(self.__goal_dict)

        goal.header.stamp = rospy.Time(0)

        c_d = deepcopy(d)
        c_d[u'parameter_value_pair'][u'goal'] = convert_ros_message_to_dictionary(goal)

        c = Constraint()
        c.type = u'CartesianPreGrasp'
        c.parameter_value_pair = json.dumps(c_d[u'parameter_value_pair'])

        return c

    def update(self):

        # Check if move_cmd exists
        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.SUCCESS

        # Check if move_cmd contains a Cartesian Goal
        cart_c = self.get_pregrasp_goal(move_cmd)
        if not cart_c:
            return Status.SUCCESS

        goal, pregrasp_pos, dist = self.get_cart_goal(cart_c)
        collision_matrix = self.collision_scene.update_collision_environment()
        if not collision_matrix:
            raise Exception('PreGrasp Sampling is not possible, since the collision matrix is empty')
        if pregrasp_pos is None:
            pregrasp_goal = self.get_pregrasp_cb(goal, self.root_link, self.tip_link, dist=dist)
        else:
            # TODO: check if given position is collision free
            pregrasp_goal = self.get_pregrasp_orientation_cb(pregrasp_pos, goal)

        move_cmd.constraints.remove(cart_c)
        move_cmd.constraints.append(self.get_cartesian_pose_constraints(pregrasp_goal))

        return Status.SUCCESS

    def get_in_map(self, ps):
        return transform_pose('map', ps).pose

    def get_pregrasp_cb(self, goal, root_link, tip_link, dist=0.00):
        collision_scene = self.god_map.unsafe_get_data(identifier.collision_scene)
        robot = collision_scene.robot
        js = self.god_map.unsafe_get_data(identifier.joint_states)
        self.state_validator = GiskardRobotBulletCollisionChecker(self.is_3D, root_link, tip_link,
                                                                  collision_scene, self.god_map, dist=dist)
        self.motion_validator = ObjectRayMotionValidator(collision_scene, tip_link, robot, self.state_validator,
                                                         self.god_map, js=js)

        p, _ = self.sample(js, tip_link, self.get_in_map(goal))
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose = p

        self.state_validator.clear()
        self.motion_validator.clear()

        return ps

    def get_pregrasp_orientation_cb(self, start, goal):
        q_arr = self.get_orientation(self.get_in_map(start), self.get_in_map(goal))
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose = self.get_in_map(start)
        ps.pose.orientation.x = q_arr[0]
        ps.pose.orientation.y = q_arr[1]
        ps.pose.orientation.z = q_arr[2]
        ps.pose.orientation.w = q_arr[3]
        return ps

    def get_orientation(self, curr, goal):
        return GoalRegionSampler(self.is_3D, goal, lambda _: True, lambda _: 0,
                                 goal_orientation=self.goal_orientation).get_orientation(curr)

    def sample(self, js, tip_link, goal, tries=1000, d=0.5):

        def compute_diff(a, b):
            return pose_diff(a, b)

        valid_fun = self.state_validator.is_collision_free
        try_i = 0
        next_goals = list()
        if self.grasping_object_name is not None:
            c = self.collision_scene.get_aabb_info(self.grasping_object_name)
            xyz = np.abs(np.array(c.d) - np.array(c.u)).tolist()
        else:
            xyz = d
        while try_i < tries and len(next_goals) == 0:
            goal_grs = GoalRegionSampler(self.is_3D, goal, valid_fun, lambda _: 0,
                                         goal_orientation=self.goal_orientation)
            s_m = SimpleRayMotionValidator(self.collision_scene, tip_link, self.god_map, ignore_state_validator=True,
                                           js=js)
            # Sample goal points which are e.g. in the aabb of the object to pick up
            goal_grs.sample(xyz)
            goals = list()
            next_goals = list()
            for s in goal_grs.samples:
                o_g = s[1]
                if s_m.checkMotion(o_g, goal):
                    goals.append(o_g)
            # Find valid goal poses which allow motion towards the sampled goal points above
            tip_link_grs = GoalRegionSampler(self.is_3D, goal, valid_fun, lambda _: 0,
                                             goal_orientation=self.goal_orientation)
            for sampled_goal in goals:
                if len(next_goals) > 0:
                    break
                tip_link_grs.valid_sample(d=.5, max_samples=10)
                for s in tip_link_grs.valid_samples:
                    n_g = s[1]
                    if self.motion_validator.checkMotion(n_g, sampled_goal):
                        next_goals.append([compute_diff(n_g, sampled_goal), n_g, sampled_goal])
                tip_link_grs.valid_samples = list()
            s_m.clear()
            try_i += 1
        if len(next_goals) != 0:
            return sorted(next_goals, key=lambda e: e[0])[0][1], sorted(next_goals, key=lambda e: e[0])[0][2]
        else:
            raise Exception('Could not find PreGrasp samples.')