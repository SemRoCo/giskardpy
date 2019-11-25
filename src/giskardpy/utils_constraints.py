import rospy
import sys
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf.transformations import quaternion_about_axis
from PyKDL import *
import tf
from urdf_parser_py.urdf import URDF
import math
import yaml
from giskardpy import symbolic_wrapper as w


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



class ConfigFileManager:
    _urdf = None
    _parsed_urdf = None
    _config_file = None

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
