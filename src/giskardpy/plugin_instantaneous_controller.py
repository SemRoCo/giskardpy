import inspect
import itertools
import json
import traceback
from copy import copy
from multiprocessing import Process
from time import time, sleep

from giskard_msgs.msg import MoveGoal, MoveCmd
from py_trees import Status
from rospy_message_converter.message_converter import convert_ros_message_to_dictionary

import giskardpy.constraints
from giskardpy.constraints import ExternalCollisionAvoidance, JointPosition
from giskardpy.exceptions import InsolvableException, ImplementationException, GiskardException
import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.plugin_action_server import GetGoal
from giskardpy.symengine_controller import InstantaneousController
from giskardpy.tfwrapper import transform_pose
from giskardpy import logging
from collections import OrderedDict, namedtuple
from giskardpy.qp_problem_builder import JointConstraint


class ControllerPlugin(GiskardBehavior):
    def __init__(self, name):
        super(ControllerPlugin, self).__init__(name)
        self.path_to_functions = self.get_god_map().safe_get_data(identifier.data_folder)
        self.nWSR = self.get_god_map().safe_get_data(identifier.nWSR)
        self.soft_constraints = None
        self.qp_data = {}
        self.get_god_map().safe_set_data(identifier.qp_data, self.qp_data) # safe dict on godmap and work on ref
        self.rc_prismatic_velocity = self.get_god_map().safe_get_data(identifier.rc_prismatic_velocity)
        self.rc_continuous_velocity = self.get_god_map().safe_get_data(identifier.rc_continuous_velocity)
        self.rc_revolute_velocity = self.get_god_map().safe_get_data(identifier.rc_revolute_velocity)
        self.rc_other_velocity = self.get_god_map().safe_get_data(identifier.rc_other_velocity)

    def initialise(self):
        super(ControllerPlugin, self).initialise()
        self.init_controller()

    def setup(self, timeout=0.0):
        return super(ControllerPlugin, self).setup(5.0)

    def init_controller(self):
        new_soft_constraints = self.get_god_map().safe_get_data(identifier.soft_constraint_identifier)
        if self.soft_constraints is None or set(self.soft_constraints.keys()) != set(new_soft_constraints.keys()):
            self.soft_constraints = copy(new_soft_constraints)
            self.controller = InstantaneousController(self.get_robot(),
                                                  u'{}/{}/'.format(self.path_to_functions, self.get_robot().get_name()))

            controlled_joints = self.get_robot().controlled_joints
            joint_to_symbols_str = OrderedDict(
                (x, self.robot.get_joint_position_symbol(x)) for x in controlled_joints)

            #make_filter_masks(self.get_god_map().safe_get_data(identifier.H), )

            if(self.get_god_map().safe_get_data(identifier.check_reachability) and False):
                joint_constraints = OrderedDict()
                for k in controlled_joints:
                    joint_type = self.get_robot().get_joint_type(k)
                    weight = self.robot._joint_constraints[k].weight
                    if joint_type == 'prismatic':
                        joint_constraints[(self.robot.get_name(), k)] = JointConstraint(-self.rc_prismatic_velocity, self.rc_prismatic_velocity, weight)
                    elif joint_type == 'continuous':
                        joint_constraints[(self.robot.get_name(), k)] = JointConstraint(-self.rc_continuous_velocity, self.rc_continuous_velocity, weight)
                    elif joint_type == 'revolute':
                        joint_constraints[(self.robot.get_name(), k)] = JointConstraint(-self.rc_revolute_velocity, self.rc_revolute_velocity, weight)
                    else:
                        joint_constraints[(self.robot.get_name(), k)] = JointConstraint(-self.rc_other_velocity, self.rc_other_velocity, weight)
            else:
                joint_constraints = OrderedDict(((self.robot.get_name(), k), self.robot._joint_constraints[k]) for k in
                                                     controlled_joints)
            hard_constraints = OrderedDict(((self.robot.get_name(), k), self.robot._hard_constraints[k]) for k in
                                                controlled_joints if k in self.robot._hard_constraints)

            self.controller.set_controlled_joints(controlled_joints, joint_to_symbols_str, joint_constraints, hard_constraints)


            #self.controller.set_controlled_joints(self.get_robot().controlled_joints)
            self.controller.update_soft_constraints(self.soft_constraints)
            # p = Process(target=self.controller.compile)
            # p.start()
            # while p.is_alive():
            #     sleep(0.05)
            # p.join()
            self.controller.compile()

            self.qp_data[identifier.weight_keys[-1]], \
            self.qp_data[identifier.b_keys[-1]], \
            self.qp_data[identifier.bA_keys[-1]], \
            self.qp_data[identifier.xdot_keys[-1]] = self.controller.get_qpdata_key_map()

    def update(self):
        last_cmd = self.get_god_map().safe_get_data(identifier.cmd)
        self.get_god_map().safe_set_data(identifier.last_cmd, last_cmd)

        expr = self.controller.get_expr()
        expr = self.god_map.get_values(expr)

        next_cmd, \
        self.qp_data[identifier.H[-1]], \
        self.qp_data[identifier.A[-1]], \
        self.qp_data[identifier.lb[-1]], \
        self.qp_data[identifier.ub[-1]], \
        self.qp_data[identifier.lbA[-1]], \
        self.qp_data[identifier.ubA[-1]], \
        self.qp_data[identifier.xdot_full[-1]] = self.controller.get_cmd(expr, self.nWSR)
        self.get_god_map().safe_set_data(identifier.cmd, next_cmd)

        return Status.RUNNING
