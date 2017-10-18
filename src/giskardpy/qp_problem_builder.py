from collections import OrderedDict, namedtuple

import numpy as np
import sympy as sp
from sympy.utilities.autowrap import ufuncify, autowrap

from giskardpy.qp_solver import QPSolver

SoftConstraint = namedtuple('SoftConstraint', ['lower', 'upper', 'weight', 'expression'])
HardConstraint = namedtuple('HardConstraint', ['lower', 'upper', 'expression'])
JointConstraint = namedtuple('JointConstraint', ['lower', 'upper', 'weight'])


class QProblemBuilder(object):
    BACKEND = 'Cython'

    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict,
                 controller_observables, robot_observables):
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.controller_observables = controller_observables
        self.robot_observables = robot_observables
        self.make_sympy_matrices()

        self.qp_solver = QPSolver(self.H.shape[0], len(self.lbA))

    def make_sympy_matrices(self):
        weights = []
        lb = []
        ub = []
        lbA = []
        ubA = []
        soft_expressions = []
        hard_expressions = []
        for c in self.joint_constraints_dict.values():
            weights.append(c.weight)
            lb.append(c.lower)
            ub.append(c.upper)
        for c in self.hard_constraints_dict.values():
            lbA.append(c.lower)
            ubA.append(c.upper)
            hard_expressions.append(c.expression)
        for c in self.soft_constraints_dict.values():
            weights.append(c.weight)
            lbA.append(c.lower)
            ubA.append(c.upper)
            lb.append(-1e9)
            ub.append(1e9)
            soft_expressions.append(c.expression)

        self.H = sp.diag(*weights)

        self.np_g = np.zeros(len(weights))

        self.lb = sp.Matrix(lb)
        self.ub = sp.Matrix(ub)

        # make A
        # hard part
        A_hard = sp.Matrix(hard_expressions)
        A_hard = A_hard.jacobian(self.robot_observables)
        zerosHxS = sp.zeros(A_hard.shape[0], len(soft_expressions))
        A_hard = zerosHxS.col_insert(0, A_hard)

        # soft part
        A_soft = sp.Matrix(soft_expressions)
        A_soft = A_soft.jacobian(self.robot_observables)
        identity3x3 = sp.eye(A_soft.shape[0])
        A_soft = identity3x3.col_insert(0, A_soft)

        # final A
        self.A = A_soft.row_insert(0, A_hard)

        self.lbA = sp.Matrix(lbA)
        self.ubA = sp.Matrix(ubA)

        self.aH = autowrap(self.H, args=list(self.H.free_symbols), backend=self.BACKEND)
        self.aA = autowrap(self.A, args=list(self.A.free_symbols), backend=self.BACKEND)
        self.alb = autowrap(self.lb, args=list(self.lb.free_symbols), backend=self.BACKEND)
        self.aub = autowrap(self.ub, args=list(self.ub.free_symbols), backend=self.BACKEND)
        self.albA = autowrap(self.lbA, args=list(self.lbA.free_symbols), backend=self.BACKEND)
        self.aubA = autowrap(self.ubA, args=list(self.ubA.free_symbols), backend=self.BACKEND)
        pass

    def transfrom_observable_matrix(self, argument_names, observables):
        asdf = [str(x) for x in argument_names]
        return {str(k): observables[k] for k in asdf}

    # @profile
    def update_observables(self, observables_update):
        self.np_H = self.update_expression_matrix(self.H, observables_update)
        self.np_A = self.update_expression_matrix(self.A, observables_update)
        self.np_lb = self.update_expression_vector(self.lb, observables_update)
        self.np_ub = self.update_expression_vector(self.ub, observables_update)
        self.np_lbA = self.update_expression_vector(self.lbA, observables_update)
        self.np_ubA = self.update_expression_vector(self.ubA, observables_update)

        xdot_full = self.qp_solver.solve(self.np_H, self.np_g, self.np_A,
                                         self.np_lb, self.np_ub, self.np_lbA, self.np_ubA)
        if xdot_full is None:
            return None
        cmd_dict = OrderedDict()
        for j, joint_name in enumerate(self.robot_observables):
            cmd_dict[joint_name] = xdot_full[j]
        return cmd_dict

    @profile
    def update_observables_cython(self, observables_update):
        self.np_H = self.update_cython_expression_matrix(self.aH, self.H.free_symbols, observables_update)
        self.np_A = self.update_cython_expression_matrix(self.aA, self.A.free_symbols, observables_update)
        self.np_lb = self.update_cython_expression_vector(self.alb, self.lb.free_symbols, observables_update)
        self.np_ub = self.update_cython_expression_vector(self.aub, self.ub.free_symbols, observables_update)
        self.np_lbA = self.update_cython_expression_vector(self.albA, self.lbA.free_symbols, observables_update)
        self.np_ubA = self.update_cython_expression_vector(self.aubA, self.ubA.free_symbols, observables_update)

        xdot_full = self.qp_solver.solve(self.np_H, self.np_g, self.np_A,
                                         self.np_lb, self.np_ub, self.np_lbA, self.np_ubA)
        if xdot_full is None:
            return None
        cmd_dict = OrderedDict()
        for j, joint_name in enumerate(self.robot_observables):
            cmd_dict[joint_name] = xdot_full[j]
        return cmd_dict

    def update_expression_matrix(self, matrix, updates_dict):
        m_sub = matrix.subs(updates_dict)
        return np.array(m_sub.tolist(), dtype=float)

    def update_cython_expression_matrix(self, matrix, argument_names, updates_dict):
        m_sub = matrix(**self.transfrom_observable_matrix(argument_names, updates_dict))
        return m_sub

    def update_expression_vector(self, vector, updates_dict):
        np_v = self.update_expression_matrix(vector, updates_dict)
        return np_v.reshape(len(np_v))

    def update_cython_expression_vector(self, vector, argument_names, updates_dict):
        np_v = self.update_cython_expression_matrix(vector, argument_names, updates_dict)
        return np_v.reshape(len(np_v))
