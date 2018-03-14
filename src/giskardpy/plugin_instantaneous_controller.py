from collections import OrderedDict
import pylab as plt
from copy import copy

import rospy
from control_msgs.msg import FollowJointTrajectoryGoal
import numpy as np

from trajectory_msgs.msg import JointTrajectoryPoint

from giskardpy.input_system import JointStatesInput, FrameInput
from giskardpy.plugin import Plugin
from giskardpy.symengine_controller import JointController, CartesianController


class ControllerPlugin(Plugin):
    def __init__(self):
        self._joint_states_identifier = 'js'
        self._goal_identifier = 'goal'
        self._next_cmd_identifier = 'motor'
        self._controller = None
        super(ControllerPlugin, self).__init__()

    def get_readings(self):
        if len(self.next_cmd) > 0:
            updates = {self._next_cmd_identifier: self.next_cmd}
            return updates
        return {}

    def update(self):
        if self.god_map.get_data(self._goal_identifier) is not None:
            next_cmd = self._controller.get_cmd(self.god_map.get_expr_values())
            self.next_cmd.update(next_cmd)

    def start(self, god_map):
        self.next_cmd = {}
        super(ControllerPlugin, self).start(god_map)

    def stop(self):
        pass

    def copy(self):
        return copy(self)

    def __copy__(self):
        cp = self.__class__()
        cp._controller = self._controller
        return cp


class JointControllerPlugin(ControllerPlugin):
    def __init__(self):
        super(JointControllerPlugin, self).__init__()
        self._goal_identifier = 'joint_goal'
        self._solution_identifier = 'joint_solution'

    def start(self, god_map):
        super(JointControllerPlugin, self).start(god_map)
        if self._controller is None:
            urdf = rospy.get_param('robot_description')
            self._controller = JointController(urdf)
            current_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                                 self._controller.robot.get_joint_names(),
                                                                 self._joint_states_identifier,
                                                                 'position')
            goal_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                              self._controller.robot.get_joint_names(),
                                                              self._goal_identifier,
                                                              'position')
            self._controller.init(current_joints, goal_joints)


class CartesianControllerPlugin(ControllerPlugin):
    def __init__(self, root, tip):
        super(CartesianControllerPlugin, self).__init__()
        self.root = root
        self.tip = tip
        self._goal_identifier = 'cartesian_goal'
        self._solution_identifier = 'cartesian_solution'

    def start(self, god_map):
        super(CartesianControllerPlugin, self).start(god_map)
        if self._controller is None:
            urdf = rospy.get_param('robot_description')
            self._controller = CartesianController(urdf)
            current_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                                 self._controller.robot.get_chain_joints(self.root, self.tip),
                                                                 self._joint_states_identifier,
                                                                 'position')
            trans_prefix = '{}/translation'.format(self._goal_identifier)
            rot_prefix = '{}/rotation'.format(self._goal_identifier)
            goal_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)

            trans_prefix = 'fk_{}/pose/position'.format(self.tip)
            rot_prefix = 'fk_{}/pose/orientation'.format(self.tip)
            current_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)

            self._controller.init(self.root, self.tip, goal_pose=goal_input, current_evaluated=current_input,
                                  current_joints=current_joints)

    def __copy__(self):
        cp = self.__class__(self.root, self.tip)
        cp._controller = self._controller
        return cp