from collections import OrderedDict, namedtuple

import numpy as np
#from sympy.utilities.autowrap import autowrap
import symengine as sp
from time import clock

from giskardpy.qp_solver import QPSolver

SoftConstraint = namedtuple('SoftConstraint', ['lower', 'upper', 'weight', 'expression'])
HardConstraint = namedtuple('HardConstraint', ['lower', 'upper', 'expression'])
JointConstraint = namedtuple('JointConstraint', ['lower', 'upper', 'weight'])


def pretty_matrix_format_str(col_names, row_names, min_col_width=10):
    w_first_col = max(*[len(n) for n in row_names])
    widths = [max(min_col_width, len(c)) for c in col_names]

    out = ''.join([(' ' * w_first_col)] + ['  {:>{:d}}'.format(n, w) for n, w in zip(col_names, widths)])
    for y in range(len(row_names)):
        out += '\n{:>{:d}}'.format(row_names[y], w_first_col)
        out += ''.join([', {}:>{:d}.5{}'.format('{', w, '}') for w in widths])

    return out

def format_matrix(matrix, mat_str):
    return mat_str.format(*matrix.reshape(1, matrix.shape[0] * matrix.shape[1]).tolist()[0])


class QProblemBuilder(object):

    BACKEND = 'Cython'

    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict):
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.controlled_joints_strs = list(self.joint_constraints_dict.keys())
        self.controlled_joints = [sp.Symbol(n) for n in self.controlled_joints_strs]
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
        M_controlled_joints = sp.Matrix(self.controlled_joints)
        A_hard = sp.Matrix(hard_expressions)
        A_hard = A_hard.jacobian(M_controlled_joints)
        zerosHxS = sp.zeros(A_hard.shape[0], len(soft_expressions))
        A_hard = A_hard.row_join(zerosHxS)

        # soft part
        A_soft = sp.Matrix(soft_expressions)
        A_soft = A_soft.jacobian(M_controlled_joints)
        identity3x3 = sp.eye(A_soft.shape[0])
        A_soft = A_soft.row_join(identity3x3)

        # final A
        self.A = A_hard.col_join(A_soft)

        self.lbA = sp.Matrix(lbA)
        self.ubA = sp.Matrix(ubA)

        self.cython_H = self.H #autowrap(self.H, args=list(self.H.free_symbols), backend=self.BACKEND)
        self.H_symbols = [str(x) for x in self.H.free_symbols]

        self.cython_A = self.A #autowrap(self.A, args=list(self.A.free_symbols), backend=self.BACKEND)
        self.A_symbols = [str(x) for x in self.A.free_symbols]

        self.cython_lb = self.lb #autowrap(self.lb, args=list(self.lb.free_symbols), backend=self.BACKEND)
        self.lb_symbols = [str(x) for x in self.lb.free_symbols]

        self.cython_ub = self.ub #autowrap(self.ub, args=list(self.ub.free_symbols), backend=self.BACKEND)
        self.ub_symbols = [str(x) for x in self.ub.free_symbols]

        self.cython_lbA = self.lbA #autowrap(self.lbA, args=list(self.lbA.free_symbols), backend=self.BACKEND)
        self.lbA_symbols = [str(x) for x in self.lbA.free_symbols]

        self.cython_ubA = self.ubA #autowrap(self.ubA, args=list(self.ubA.free_symbols), backend=self.BACKEND)
        self.ubA_symbols = [str(x) for x in self.ubA.free_symbols]

        # Strings for printing
        col_names = self.controlled_joints_strs + ['slack'] * len(soft_expressions)
        row_names = self.hard_constraints_dict.keys() + self.soft_constraints_dict.keys()

        self.str_A = pretty_matrix_format_str(col_names, row_names)

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
        try:
            return np.array(matrix.subs(args).tolist(), dtype=float).reshape(matrix.shape)
        except Exception as e:
            print(matrix.subs(args))
            raise e


    def update_expression_vector(self, vector, argument_names, updates_dict):
        np_v = self.update_expression_matrix(vector, argument_names, updates_dict)
        return np_v.reshape(len(np_v))

    def print_jacobian(self):
        print('Matrix A: \n{}'.format(format_matrix(self.np_A, self.str_A)))
