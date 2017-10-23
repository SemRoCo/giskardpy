from collections import OrderedDict, namedtuple

import numpy as np
import sympy as sp
from sympy.utilities.autowrap import autowrap

from giskardpy.qp_solver import QPSolver

SoftConstraint = namedtuple('SoftConstraint', ['lower', 'upper', 'weight', 'expression'])
HardConstraint = namedtuple('HardConstraint', ['lower', 'upper', 'expression'])
JointConstraint = namedtuple('JointConstraint', ['lower', 'upper', 'weight'])


class QProblemBuilder(object):
    BACKEND = 'Cython'

    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict):
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.controlled_joints_strs = list(self.joint_constraints_dict.keys())
        self.controlled_joints = sp.sympify(self.controlled_joints_strs)
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
        for jn in self.controlled_joints:
            c = self.joint_constraints_dict[str(jn)]
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
        A_hard = A_hard.jacobian(self.controlled_joints)
        zerosHxS = sp.zeros(A_hard.shape[0], len(soft_expressions))
        A_hard = zerosHxS.col_insert(0, A_hard)

        # soft part
        A_soft = sp.Matrix(soft_expressions)
        A_soft = A_soft.jacobian(self.controlled_joints)
        identity3x3 = sp.eye(A_soft.shape[0])
        A_soft = identity3x3.col_insert(0, A_soft)

        # final A
        self.A = A_soft.row_insert(0, A_hard)

        self.lbA = sp.Matrix(lbA)
        self.ubA = sp.Matrix(ubA)

        self.cython_H = autowrap(self.H, args=list(self.H.free_symbols), backend=self.BACKEND)
        self.H_symbols = [str(x) for x in self.H.free_symbols]

        self.cython_A = autowrap(self.A, args=list(self.A.free_symbols), backend=self.BACKEND)
        self.A_symbols = [str(x) for x in self.A.free_symbols]

        self.cython_lb = autowrap(self.lb, args=list(self.lb.free_symbols), backend=self.BACKEND)
        self.lb_symbols = [str(x) for x in self.lb.free_symbols]

        self.cython_ub = autowrap(self.ub, args=list(self.ub.free_symbols), backend=self.BACKEND)
        self.ub_symbols = [str(x) for x in self.ub.free_symbols]

        self.cython_lbA = autowrap(self.lbA, args=list(self.lbA.free_symbols), backend=self.BACKEND)
        self.lbA_symbols = [str(x) for x in self.lbA.free_symbols]

        self.cython_ubA = autowrap(self.ubA, args=list(self.ubA.free_symbols), backend=self.BACKEND)
        self.ubA_symbols = [str(x) for x in self.ubA.free_symbols]


    def filter_observables(self, argument_names, observables_update):
        return {str(k): observables_update[k] for k in argument_names}

    def update_observables(self, observables_update):
        self.np_H = self.update_expression_matrix(self.cython_H, self.H_symbols, observables_update)
        self.np_A = self.update_expression_matrix(self.cython_A, self.A_symbols, observables_update)
        self.np_lb = self.update_expression_vector(self.cython_lb, self.lb_symbols, observables_update)
        self.np_ub = self.update_expression_vector(self.cython_ub, self.ub_symbols, observables_update)
        self.np_lbA = self.update_expression_vector(self.cython_lbA, self.lbA_symbols, observables_update)
        self.np_ubA = self.update_expression_vector(self.cython_ubA, self.ubA_symbols, observables_update)

        xdot_full = self.qp_solver.solve(self.np_H, self.np_g, self.np_A,
                                         self.np_lb, self.np_ub, self.np_lbA, self.np_ubA)
        if xdot_full is None:
            return None
        return OrderedDict((observable, xdot_full[i]) for i, observable in enumerate(self.controlled_joints_strs))

    def update_expression_matrix(self, matrix, argument_names, updates_dict):
        args = self.filter_observables(argument_names, updates_dict)
        if len(args) == 1:
            return matrix(args.values()[0])
        return matrix(**args)

    def update_expression_vector(self, vector, argument_names, updates_dict):
        np_v = self.update_expression_matrix(vector, argument_names, updates_dict)
        return np_v.reshape(len(np_v))
