import rospy
import sys
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf.transformations import quaternion_about_axis
from PyKDL import *
import tf
from urdf_parser_py.urdf import URDF
import math
import yaml
from giskardpy import symbolic_wrapper as w
import copy
from giskardpy.python_interface import GiskardWrapper
from iai_naive_kinematics_sim.srv import SetJointState, SetJointStateRequest, SetJointStateResponse  # comment this line
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_about_axis
import tf2_ros
import tf2_geometry_msgs
from giskardpy.qp_problem_builder import SoftConstraint
from collections import OrderedDict
from geometry_msgs.msg import Vector3Stamped, Vector3
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message
from giskardpy import symengine_wrappers as sw
from giskardpy import god_map as gm
from giskardpy import tfwrapper as tf_wrapper
from giskardpy.tfwrapper import lookup_pose, pose_to_kdl, np_to_kdl, kdl_to_pose
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from pr2_controllers_msgs.msg import Pr2GripperCommandAction, Pr2GripperCommandActionGoal


class Utils:

    def __init__(self):
        # initialize utils
        print("Utils is ready for all calculs")
        self.config_file = None

    def estimated_positions_fro_circle(self, point, point2, axis, angle):
        self.center = point
        print("Center point")
        print(self.center)
        self.projection = point2
        print("radius Pointt")
        print(self.projection)
        self.angle = angle
        print("angle of new point")
        print(self.angle)

        self.vector_center_to_projection = [self.projection[0] - self.center[0],
                                            self.projection[1] - self.center[1],
                                            self.projection[2] - self.center[2]]
        self.estimated_vector = []
        for a in range(3):
            if axis[a] == 1 and a == 0:
                self.estimated_vector = self.vector_rotation_on_roll(self.vector_center_to_projection, self.angle)
            elif axis[a] == 1 and a == 1:
                self.estimated_vector = self.vector_rotation_on_pitch(self.vector_center_to_projection, self.angle)
            elif axis[a] == 1 and a == 2:
                self.estimated_vector = self.vector_rotation_on_yaw(self.vector_center_to_projection, self.angle)

        self.estimated_position = [self.estimated_vector[0] + self.center[0],
                                   self.estimated_vector[1] + self.center[1],
                                   self.estimated_vector[2] + self.center[2]]
        return self.estimated_position

    def get_distance(self, point1, point2):
        return np.linalg.norm(np.array(point2) - np.array(point1))

    def get_angle_casadi(self, point1, point2, point3):
        v12 = w.vector3(*point1) - w.vector3(*point2)
        v13 = w.vector3(*point1) - w.vector3(*point3)
        v23 = w.vector3(*point2) - w.vector3(*point3)
        d12 = w.norm(v12)
        d13 = w.norm(v13)
        d23 = w.norm(v23)
        # return w.acos(w.dot(v12, v13.T)[0] / (d12 * d13))
        return w.acos((d12 ** 2 + d13 ** 2 - d23 ** 2) / (2 * d12 * d13))

    def get_angle_with_atan(self, point1, point2, point3):
        v12 = w.vector3(*point1) - w.vector3(*point2)
        v13 = w.vector3(*point1) - w.vector3(*point3)
        # v23 = w.vector3(*point2) - w.vector3(*point3)
        # return (v12[0] * v13[2] - v12[1] * v13[0]) / abs(float(v12[0] * v13[2] - v12[1] * v13[0]))
        return math.atan2(v12[1], v12[0]) - math.atan2(v13[1], v13[0])

    def get_angle(self, point1, point2, point3):
        v12 = np.subtract(point1, point2)
        v13 = np.subtract(point1, point3)
        d12 = self.get_distance(point1, point2)
        d13 = self.get_distance(point1, point3)
        return np.arccos(np.dot(v12, v13) / (d12 * d13))

    def vector_rotation_on_yaw(self, vector, angle):
        return [np.cos(angle) * vector[0] + (-np.sin(angle) * vector[1]) + 0 * vector[2],
                np.sin(angle) * vector[0] + (np.cos(angle) * vector[1]) + 0 * vector[2],
                0 * vector[0] + 0 * vector[1] + 1 * vector[2]]

    def vector_rotation_on_roll(self, vector, angle):
        return [1 * vector[0] + 0 * vector[1] + 0 * vector[2],
                0 * vector[0] + np.cos(angle) * vector[1] + (-np.sin(angle) * vector[2]),
                0 * vector[0] + np.sin(angle) * vector[1] + (np.cos(angle) * vector[2])]

    def vector_rotation_on_pitch(self, vector, angle):
        return [np.cos(angle) * vector[0] + 0 * vector[1] + (np.sin(angle) * vector[2]),
                0 * vector[0] + 1 * vector[1] + 0 * vector[2],
                (-np.sin(angle) * vector[0]) + 0 * vector[1] + (np.cos(angle) * vector[2])]

    def rotation2d(self, vector, angle):
        return [np.cos(angle) * vector[0] - np.sin(angle) * vector[1],
                np.sin(angle) * vector[0] + np.cos(angle) * vector[1]]

    def rotation_gripper_to_object(self, axis):
        if str(axis) == "y":
            h_g = w.Matrix([[-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
            return h_g
        elif axis == "z":
            h_g = w.Matrix([[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            return h_g
        elif axis == "y_2":
            h_g = w.Matrix([[-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]])
            return h_g

    def joints_without_rotation(self):
        return ["oven_area_oven_knob_stove_1_joint",
                "oven_area_oven_knob_stove_2_joint",
                "oven_area_oven_knob_stove_3_joint",
                "oven_area_oven_knob_stove_4_joint"]

    def rotate_oven_knob_stove(self):
        h_g = w.Matrix([[0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        return h_g

    def fake_parallel_vectors(self, fake_point_on_object, fake_point_on_gripper, axis):
        # return [x + 2 for x in fake_point_on_object], [y - 2 for y in fake_point_on_gripper]
        pass

    # pose1 is the start and pose2 the direction
    def get_vector(self, pose1, pose2):
        return [pose2[0] - pose1[0], pose2[1] - pose1[1], pose2[2] - pose1[2]]

    def cross_product(self, u, v):
        """
        this method performs the cross product of two vector,
        check if the vectors are orthogonal
        :param u: first vector
        :type: array float
        :param v: second vector
        :type: array float
        :return: 0 if the vectors are orthogonal
        """
        # return ((u[1] * v[2]) - (u[2] * v[1])) + ((u[2] * v[0]) - (u[0] * v[2])) + ((u[0] * v[1]) - (u[1] * v[0]))
        # return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        return round(u[0] * v[0] + u[1] * v[1] + u[2] * v[2], 2)

    def cross_productaxis(self, u, v, axis):
        grasp_axis = self.axis_converter(axis)
        u1 = []
        v1 = []
        for x in range(len(grasp_axis)):
            if grasp_axis[x] == 0:
                u1.append(u[x])
                v1.append(v[x])
        return v1[0] * u1[0] + v1[1] * u1[1]

    def axis_converter(self, axis):
        if isinstance(axis, str):
            if axis == "x":
                return [1, 0, 0]
            elif axis == "y":
                return [0, 1, 0]
            elif axis == "z":
                return [0, 0, 1]

    def create_point_on_line(self, first_point, second_point, factor):
        """
        this method create a new point on the line first_point - second_point
        :param first_point: the first point of line, or this is the basis point
        :type: array of float
        :param second_point: any second point on the line
        :type: array of float
        :param factor: any factor, for forward (+) and backward (-) point
        :return: new point on the line , array of float
        """
        return [first_point[0] + factor * (first_point[0] - second_point[0]),
                first_point[1] + factor * (first_point[1] - second_point[1]),
                first_point[2] + factor * (first_point[2] - second_point[2])]

    def translate_point_on_axis(self, axis, point):
        point_duplicate = copy.deepcopy(point)
        return [point_duplicate[0] + axis[0],
                point_duplicate[1] + axis[1],
                point_duplicate[2] + axis[2]]

    def get_symbol(self, value):
        if value >= 0:
            return 1
        return -1


class ConfigFileManager:
    _urdf = None
    _parsed_urdf = None
    _config_file = None
    _pr2_palm_link = {'l_gripper_tool_frame': "l_gripper_palm_link", 'r_gripper_tool_frame': "r_gripper_palm_link"}

    def __init__(self):
        # initialize utils
        print("Config File manager is ready for all setup")
        self._config_file = []

    def load_urdf(self, description_name):
        self._urdf = rospy.get_param(description_name)
        self._parsed_urdf = URDF.from_xml_string(self._urdf)

    def set_yaml_config_file(self, path="."):
        for j in self._parsed_urdf.joints:
            self._config_file.append({'name': j.name, 'params': {  # 'axis': j.axis,  # 'limits': joint_limits,
                'constraint_type': "",
                'controllable_link': "",
                # 'joint_type': j.joint_type,
                'grasp_axis': ''
            }
                                      })
        with open(path, 'w') as file:
            conf_f = yaml.dump(self._config_file, file)
            print conf_f

    def load_yaml_config_file(self, path="."):
        with open(path) as file:
            self._config_file = yaml.full_load(file)
            print self._config_file

    def update_joint_of_config_file(self, path=".", joint_name=None, constraint_type=None,
                                    controllable_link=None, grasp_axis=None):
        for j in self._config_file:
            if joint_name == j["name"]:

                if constraint_type is not None:
                    j["params"]["constraint_type"] = constraint_type
                if controllable_link is not None:
                    j["params"]["controllable_link "] = controllable_link
                if grasp_axis is not None:
                    j["params"]['grasp_axis'] = grasp_axis

        with open(path, 'w') as file:
            conf_f = yaml.dump(self._config_file, file)
            print conf_f

    def get_params_joint(self, joint_name=None):
        for jn in self._config_file:
            if jn['name'] == joint_name:
                return jn['params']

        return None

    def get_palm_link(self, robot_name, gripper_frame):
        if robot_name == "pr2":
            return self._pr2_palm_link[gripper_frame]

        return -1
