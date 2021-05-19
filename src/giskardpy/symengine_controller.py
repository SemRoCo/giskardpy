import hashlib
import warnings
from collections import OrderedDict
from itertools import chain
from giskardpy.qp_problem_builder import QProblemBuilder
from giskardpy.robot import Robot


class InstantaneousController(object):
    """
    This class handles constraints and computes joint commands using symengine and qpOases.
    """

    # TODO should anybody who uses this class know about constraints?


    def __init__(self, robot, sample_period, prediciton_horizon, control_horizon, path_to_functions):
        """
        :type robot: Robot
        :param path_to_functions: location where compiled functions are stored
        :type: str
        """
        self.path_to_functions = path_to_functions
        self.prediciton_horizon = prediciton_horizon
        self.control_horizon = control_horizon
        self.robot = robot
        self.controlled_joints = []
        self.hard_constraints = {}
        self.joint_constraints = {}
        self.soft_constraints = {}
        self.debug_expressions = {}
        self.free_symbols = None
        self.sample_period = sample_period
        self.qp_problem_builder = None # type: QProblemBuilder


    def get_qpdata_key_map(self):
        return self.qp_problem_builder.b_names(), self.qp_problem_builder.bA_names()

    def update_constraints(self, joint_to_symbols_str, soft_constraints, joint_constraints, hard_constraints,
                           debug_expressions):
        """
        Triggers a recompile if the number of soft constraints has changed.
        :type soft_constraints: dict
        :type free_symbols: set
        """
        # TODO bug if soft constraints get replaced, actual amount does not change.
        last_number_of_constraints = len(self.soft_constraints)
        self.soft_constraints.update(soft_constraints)
        if last_number_of_constraints != len(self.soft_constraints):
            self.qp_problem_builder = None

        self.joint_to_symbols_str = joint_to_symbols_str
        self.joint_constraints = joint_constraints
        self.hard_constraints = hard_constraints
        self.debug_expressions = debug_expressions


    def compile(self):
        a = ''.join(str(x) for x in sorted(chain([(x,) for x in self.soft_constraints.keys()],
                                                 self.hard_constraints.keys(),
                                                 self.joint_constraints.keys())))
        function_hash = hashlib.md5((a + self.robot.get_urdf_str()).encode('utf-8')).hexdigest()
        path_to_functions = self.path_to_functions + function_hash
        self.qp_problem_builder = QProblemBuilder(self.joint_constraints,
                                                  self.hard_constraints,
                                                  self.soft_constraints,
                                                  self.sample_period,
                                                  self.prediciton_horizon,
                                                  self.control_horizon,
                                                  self.debug_expressions,
                                                  path_to_functions)

    @profile
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
        next_velocity, next_acceleration, next_jerk, \
        H, A, lb, ub, lbA, ubA, xdot_full = self.qp_problem_builder.get_cmd(substitutions, nWSR)
        return {name: next_velocity[symbol] for name, (symbol, _) in self.joint_to_symbols_str.items()}, \
               {name: next_acceleration[symbol] for name, (symbol, _) in self.joint_to_symbols_str.items()}, \
               H, A, lb, ub, lbA, ubA, xdot_full

    def get_expr(self):
        return self.qp_problem_builder.get_expr()

