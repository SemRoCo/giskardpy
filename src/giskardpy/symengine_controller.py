import hashlib
import warnings
from collections import OrderedDict
from itertools import chain

import symengine_wrappers as sw
from giskardpy.qp_problem_builder import QProblemBuilder, SoftConstraint
from giskardpy.symengine_robot import Robot


class SymEngineController(object):
    """
    This class handles constraints and computes joint commands using symengine and qpOases.
    """

    # TODO should anybody who uses this class know about constraints?

    def __init__(self, robot, path_to_functions):
        """
        :type robot: Robot
        :param path_to_functions: location where compiled functions are stored
        :type: str
        """
        self.path_to_functions = path_to_functions
        self.robot = robot
        self.controlled_joints = []
        self.hard_constraints = {}
        self.joint_constraints = {}
        self.soft_constraints = {}
        self.free_symbols = None
        self.qp_problem_builder = None

    def set_controlled_joints(self, joint_names):
        """
        :type joint_names: set
        """
        self.controlled_joints = joint_names
        self.joint_to_symbols_str = OrderedDict((x, self.robot.get_joint_symbol(x)) for x in self.controlled_joints)
        self.joint_constraints = OrderedDict(((self.robot.get_name(), k), self.robot._joint_constraints[k]) for k in
                                             self.controlled_joints)
        self.hard_constraints = OrderedDict(((self.robot.get_name(), k), self.robot._hard_constraints[k]) for k in
                                            self.controlled_joints if k in self.robot._hard_constraints)

    def update_soft_constraints(self, soft_constraints, free_symbols=None):
        """
        Triggers a recompile if the number of soft constraints has changed.
        :type soft_constraints: dict
        :type free_symbols: set
        """
        if free_symbols is not None:
            warnings.warn(u'use of free_symbols deprecated', DeprecationWarning)
        # TODO bug if soft constraints get replaced, actual amount does not change.
        last_number_of_constraints = len(self.soft_constraints)
        if free_symbols is not None:
            if self.free_symbols is None:
                self.free_symbols = set()
            self.free_symbols.update(free_symbols)
        self.soft_constraints.update(soft_constraints)
        if last_number_of_constraints != len(self.soft_constraints):
            self.qp_problem_builder = None

    def compile(self):
        a = ''.join(str(x) for x in sorted(chain(self.soft_constraints.keys(),
                                                 self.hard_constraints.keys(),
                                                 self.joint_constraints.keys())))
        function_hash = hashlib.md5(a + self.robot.get_urdf_str()).hexdigest()
        path_to_functions = self.path_to_functions + function_hash
        self.qp_problem_builder = QProblemBuilder(self.joint_constraints,
                                                  self.hard_constraints,
                                                  self.soft_constraints,
                                                  self.joint_to_symbols_str.values(),
                                                  self.free_symbols,
                                                  path_to_functions)

    def get_cmd(self, substitutions, nWSR=None):
        """
        Computes joint commands that satisfy constrains given substitutions.
        :param substitutions: maps symbol names as str to floats.
        :type substitutions: dict
        :param nWSR: magic number, if None throws errors, increase this until it stops.
        :type nWSR: int
        :return: maps joint names to command
        :rtype: dict
        """
        next_cmd = self.qp_problem_builder.get_cmd(substitutions, nWSR)
        if next_cmd is None:
            pass
        return {name: next_cmd[symbol] for name, symbol in self.joint_to_symbols_str.items()}

    def get_expr(self):
        return self.qp_problem_builder.get_expr()

