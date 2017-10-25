from collections import OrderedDict, namedtuple

import numpy as np
# import sympy as sp
# from sympy.utilities.autowrap import autowrap
import symengine as sp2

from giskardpy.qp_solver import QPSolver

SoftConstraint = namedtuple('SoftConstraint', ['lower', 'upper', 'weight', 'expression'])
HardConstraint = namedtuple('HardConstraint', ['lower', 'upper', 'expression'])
JointConstraint = namedtuple('JointConstraint', ['lower', 'upper', 'weight'])


class QProblemBuilder(object):
    # BACKEND = 'Cython'

    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict):
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.controlled_joints_strs = list(self.joint_constraints_dict.keys())
        self.controlled_joints = [sp2.Symbol(n) for n in self.controlled_joints_strs]
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

        self.H = sp2.diag(*weights)

        self.np_g = np.zeros(len(weights))

        self.lb = sp2.Matrix(lb)
        self.ub = sp2.Matrix(ub)

        # make A
        # hard part
        M_controlled_joints = sp2.Matrix(self.controlled_joints)
        A_hard = sp2.Matrix(hard_expressions)
        A_hard = A_hard.jacobian(M_controlled_joints)
        zerosHxS = sp2.zeros(A_hard.shape[0], len(soft_expressions))
        A_hard = A_hard.row_join(zerosHxS)

        # soft part
        A_soft = sp2.Matrix(soft_expressions)
        A_soft = A_soft.jacobian(M_controlled_joints)
        identity3x3 = sp2.eye(A_soft.shape[0])
        A_soft = A_soft.row_join(identity3x3)

        # final A
        self.A = A_hard.col_join(A_soft)

        self.lbA = sp2.Matrix(lbA)
        self.ubA = sp2.Matrix(ubA)

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
        return OrderedDict((observable, xdot_full[i]) for i, observable in enumerate(self.controlled_joints_strs))

    def update_expression_matrix(self, matrix, updates_dict):
        return np.array(matrix.subs(updates_dict).tolist(), dtype=float).reshape(matrix.shape)

    def update_expression_vector(self, vector, updates_dict):
        np_v = self.update_expression_matrix(vector, updates_dict)
        return np_v.reshape(len(np_v))
