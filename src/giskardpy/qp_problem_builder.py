from collections import OrderedDict, namedtuple

import numpy as np
from time import time

from giskardpy import USE_SYMENGINE, BACKEND
from giskardpy.cvxopt_qp_solver import CVXQPSolver
from giskardpy.osqp_solver import OSQPSolver
from giskardpy.quadprog_qp_solver import QuadProgQPSolver

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
        # self.qp_solver = OSQPSolver(self.H.shape[0], len(self.lbA))
        # self.qp_solver = CVXQPSolver(self.H.shape[0], len(self.lbA))
        # self.qp_solver = QuadProgQPSolver(self.H.shape[0], len(self.lbA))

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
        for k, c in self.soft_constraints_dict.items():
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

        big_ass_M_A = self.A.row_join(self.lbA).row_join(self.ubA)
        big_ass_M_H = self.H.row_join(self.lb).row_join(self.ub)
        self.big_ass_M = big_ass_M_A.col_join(big_ass_M_H)

        t = time()
        self.cython_big_ass_M = spw.speed_up(self.big_ass_M, self.big_ass_M.free_symbols, backend=BACKEND)
        # self.big_ass_list = self.H.reshape(len(self.H),1)\
        #     .col_join(self.A.reshape(len(self.A),1))\
        #     .col_join(self.lb).col_join(self.ub)\
        #     .col_join(self.lbA).col_join(self.ubA)
        # self.cython_list = spw.speed_up(self.big_ass_list.T, self.big_ass_list.free_symbols, backend=BACKEND)

        print('autowrap took {}'.format(time() - t))
        # raise Exception()

    # @profile
    def update_observables(self, observables_update):
        # evaluated_updates = OrderedDict()
        # for k, v in observables_update.items():
        #     if not isinstance(v, int) and not isinstance(v, float):
        #         evaluated_updates[k] = v(observables_update)
        #     else:
        #         evaluated_updates[k] = v
        evaluated_updates = observables_update

        # self.np_big_ass_list = self.cython_list(**evaluated_updates)
        # i1 = len(self.H)
        # self.np_H = self.np_big_ass_list[0, 0:i1].reshape(self.H.shape)
        # i2 = i1 + len(self.A)
        # self.np_A = self.np_big_ass_list[0, i1:i2].reshape(self.A.shape)
        # i1 = i2
        # i2 += len(self.lb)
        # self.np_lb = self.np_big_ass_list[0, i1:i2]
        # i1 = i2
        # i2 += len(self.ub)
        # self.np_ub = self.np_big_ass_list[0, i1:i2]
        # i1 = i2
        # i2 += len(self.lbA)
        # self.np_lbA = self.np_big_ass_list[0, i1:i2]
        # i1 = i2
        # self.np_ubA = self.np_big_ass_list[0, i1:]

        self.np_big_ass_M = self.cython_big_ass_M(**evaluated_updates)
        self.np_H = np.array(self.np_big_ass_M[self.A.shape[0]:,:-2])
        self.np_A = np.array(self.np_big_ass_M[:self.A.shape[0],:self.A.shape[1]])
        self.np_lb = np.array(self.np_big_ass_M[self.A.shape[0]:,-2])
        self.np_ub = np.array(self.np_big_ass_M[self.A.shape[0]:,-1])
        self.np_lbA = np.array(self.np_big_ass_M[:self.A.shape[0],-2])
        self.np_ubA = np.array(self.np_big_ass_M[:self.A.shape[0],-1])

        xdot_full = self.qp_solver.solve(self.np_H, self.np_g, self.np_A,
                                         self.np_lb, self.np_ub, self.np_lbA, self.np_ubA)
        # print(xdot_full[-7:])
        if xdot_full is None:
            return None
        return OrderedDict((observable, xdot_full[i]) for i, observable in enumerate(self.controlled_joints_strs))
