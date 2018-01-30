from collections import OrderedDict, namedtuple

import numpy as np
from time import time

from giskardpy import USE_SYMENGINE
from giskardpy import print_wrapper

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw
from giskardpy.qp_solver import QPSolver

SoftConstraint = namedtuple('SoftConstraint', ['lower', 'upper', 'weight', 'expression'])
HardConstraint = namedtuple('HardConstraint', ['lower', 'upper', 'expression'])
JointConstraint = namedtuple('JointConstraint', ['lower', 'upper', 'weight'])

BIG_NUMBER = 1e9


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
    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict, backend=None, logging=print_wrapper):
        self.backend = backend
        self.logging = logging
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.controlled_joints_strs = list(self.joint_constraints_dict.keys())
        self.controlled_joints = [spw.Symbol(n) for n in self.controlled_joints_strs]
        self.soft_constraint_indices = {}
        self.make_sympy_matrices()

        self.qp_solver = QPSolver(self.H.shape[0], len(self.lbA))

    # @profile
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
        for scname, c in self.soft_constraints_dict.items():
            self.soft_constraint_indices[scname] = len(lbA)
            weights.append(c.weight)
            lbA.append(c.lower)
            ubA.append(c.upper)
            lb.append(-BIG_NUMBER)
            ub.append(BIG_NUMBER)
            soft_expressions.append(c.expression)

        self.H = spw.diag(*weights)

        self.np_g = np.zeros(len(weights))

        self.lb = spw.Matrix(lb)
        self.ub = spw.Matrix(ub)

        # make A
        # hard part
        M_controlled_joints = spw.Matrix(self.controlled_joints)
        A_hard = spw.Matrix(hard_expressions)
        A_hard = A_hard.jacobian(M_controlled_joints)
        zerosHxS = spw.zeros(A_hard.shape[0], len(soft_expressions))
        A_hard = A_hard.row_join(zerosHxS)

        # soft part
        A_soft = spw.Matrix(soft_expressions)
        t = time()
        A_soft = A_soft.jacobian(M_controlled_joints)
        self.logging('jacobian took {}'.format(time() - t))
        identity = spw.eye(A_soft.shape[0])
        A_soft = A_soft.row_join(identity)

        # final A
        self.A = A_hard.col_join(A_soft)

        self.lbA = spw.Matrix(lbA)
        self.ubA = spw.Matrix(ubA)

        t = time()
        try:
            self.cython_H = spw.speed_up(self.H, self.H.free_symbols)
        except Exception as e:
            raise Exception('Error while wrapping weight matrix! Error: {}\n'.format(e))

        try:
            self.cython_A = spw.speed_up(self.A, self.A.free_symbols)
        except Exception as e:
            raise Exception('Error while wrapping jacobian! Error: {}\n'.format(e))

        try:
            self.cython_lb = spw.speed_up(self.lb, self.lb.free_symbols)
        except Exception as e:
            raise Exception('Error while wrapping lower bounds! Error: {}\n'.format(e))

        try:
            self.cython_ub = spw.speed_up(self.ub, self.ub.free_symbols)
        except Exception as e:
            raise Exception('Error while wrapping upper bounds! Error: {}\n'.format(e))

        try:
            self.cython_lbA = spw.speed_up(self.lbA, self.lbA.free_symbols)
        except Exception as e:
            raise Exception('Error while wrapping jacobian lower bounds! Error: {}\n'.format(e))

        try:
            self.cython_ubA = spw.speed_up(self.ubA, self.ubA.free_symbols)
        except Exception as e:
            raise Exception('Error while wrapping jacobian upper bounds! Error: {}\n'.format(e))

        self.logging('autowrap took {}'.format(time() - t))

        # Strings for printing
        col_names = self.controlled_joints_strs + ['slack'] * len(soft_expressions)
        row_names = self.hard_constraints_dict.keys() + self.soft_constraints_dict.keys()

        self.str_A = pretty_matrix_format_str(col_names, row_names)

    # @profile
    def update_observables(self, observables_update):
#        print('Evaluating H')
        self.np_H = self.cython_H(**observables_update)
#        print('Evaluating A')
        self.np_A = self.cython_A(**observables_update)
#        print('Evaluating ctrl lb')
        self.np_lb = self.cython_lb(**observables_update).reshape(self.lb.shape[0])
        #print('Evaluating ctrl ub')
        self.np_ub = self.cython_ub(**observables_update).reshape(self.ub.shape[0])
        #print('Evaluating A lb')
        self.np_lbA = self.cython_lbA(**observables_update).reshape(self.lbA.shape[0])
        #print('Evaluating A ub')
        self.np_ubA = self.cython_ubA(**observables_update).reshape(self.ubA.shape[0])

        xdot_full = self.qp_solver.solve(self.np_H, self.np_g, self.np_A,
                                         self.np_lb, self.np_ub, self.np_lbA, self.np_ubA)
        if xdot_full is None:
            return None
        return OrderedDict((observable, xdot_full[i]) for i, observable in enumerate(self.controlled_joints_strs))

    def constraints_met(self, lbThreshold=0.01, ubThreshold=-0.01, names=None):
        if names == None:
            for x in range(len(self.np_lbA)):
                if self.np_lbA[x] > lbThreshold or self.np_ubA[x] < ubThreshold:
                    return False
        else:
            for name in names:
                x = self.soft_constraint_indices[name]
                if self.np_lbA[x] > lbThreshold or self.np_ubA[x] < ubThreshold:
                    return False
        return True

    def str_jacobian(self):
        return format_matrix(self.np_A, self.str_A)

    def log_jacobian(self):
        self.logging('Matrix A: \n{}'.format(self.str_jacobian()))
