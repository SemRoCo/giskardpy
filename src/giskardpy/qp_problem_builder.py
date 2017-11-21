from collections import OrderedDict, namedtuple

import numpy as np
from time import time

from giskardpy import USE_SYMENGINE, BACKEND

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw
from giskardpy.qp_solver import QPSolver

SoftConstraint = namedtuple('SoftConstraint', ['lower', 'upper', 'weight', 'expression'])
HardConstraint = namedtuple('HardConstraint', ['lower', 'upper', 'expression'])
JointConstraint = namedtuple('JointConstraint', ['lower', 'upper', 'weight'])

BIG_NUMBER = 1e9

class QProblemBuilder(object):
    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict, backend=None):
        self.backend = backend
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.controlled_joints_strs = list(self.joint_constraints_dict.keys())
        self.controlled_joints = [spw.Symbol(n) for n in self.controlled_joints_strs]
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
        for c in self.soft_constraints_dict.values():
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
        print('jacobian took {}'.format(time() - t))
        identity = spw.eye(A_soft.shape[0])
        A_soft = A_soft.row_join(identity)

        # final A
        self.A = A_hard.col_join(A_soft)

        self.lbA = spw.Matrix(lbA)
        self.ubA = spw.Matrix(ubA)

        t = time()
        self.cython_H = spw.speed_up(self.H, self.H.free_symbols, backend=BACKEND)

        self.cython_A = spw.speed_up(self.A, self.A.free_symbols, backend=BACKEND)

        self.cython_lb = spw.speed_up(self.lb, self.lb.free_symbols, backend=BACKEND)

        self.cython_ub = spw.speed_up(self.ub, self.ub.free_symbols, backend=BACKEND)

        self.cython_lbA = spw.speed_up(self.lbA, self.lbA.free_symbols, backend=BACKEND)

        self.cython_ubA = spw.speed_up(self.ubA, self.ubA.free_symbols, backend=BACKEND)
        print('autowrap took {}'.format(time() - t))

    # @profile
    def update_observables(self, observables_update):
        self.np_H = self.cython_H(**observables_update)
        self.np_A = self.cython_A(**observables_update)
        self.np_lb = self.cython_lb(**observables_update).reshape(self.lb.shape[0])
        self.np_ub = self.cython_ub(**observables_update).reshape(self.ub.shape[0])
        self.np_lbA = self.cython_lbA(**observables_update).reshape(self.lbA.shape[0])
        self.np_ubA = self.cython_ubA(**observables_update).reshape(self.ubA.shape[0])

        xdot_full = self.qp_solver.solve(self.np_H, self.np_g, self.np_A,
                                         self.np_lb, self.np_ub, self.np_lbA, self.np_ubA)
        if xdot_full is None:
            return None
        return OrderedDict((observable, xdot_full[i]) for i, observable in enumerate(self.controlled_joints_strs))
