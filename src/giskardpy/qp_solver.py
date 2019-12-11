import numpy as np

import qpoases
from qpoases import PyReturnValue
import casadi as ca
import osqp
from scipy import sparse

from giskardpy.exceptions import MAX_NWSR_REACHEDException, QPSolverException
from giskardpy import logging


class QPSolver(object):
    # RETURN_VALUE_DICT = {value: name for name, value in vars(PyReturnValue).items()}

    def __init__(self, h, j, s):
        """
        :param dim_a: number of joint constraints + number of soft constraints
        :type int
        :param dim_b: number of hard constraints + number of soft constraints
        :type int
        """
        # self.init(dim_a, dim_b)
        self.h = h
        self.j = j
        self.s = s
        self.started = False
        self.shape = (0, 0)
        pass

    #@profile
    def init_problem(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
        I = np.identity(A.shape[1])

        AI = np.concatenate((A, I))

        Hs=sparse.csc_matrix(H)
        AIs = sparse.csc_matrix(AI)

        self.m = osqp.OSQP()
        self.m.setup(P=Hs, q=g, A=AIs, l=lb, u=ub, rho=0.5,
                                           sigma=5,
                                           max_iter=100,
                                           eps_abs=10.0,
                                           eps_rel=0.01,
                                           eps_prim_inf=100.5,
                                           eps_dual_inf=100.5,
                                           alpha=1.0,
                                           delta=0.1,
                                           polish=0,
                                           polish_refine_iter=2,
                                           verbose=False)

        self.started = True
        self.old_Hs = Hs
        self.old_AIs = AIs
        r = self.m.solve()
        return r.x.T


    #@profile
    def hot_start(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
        I = np.identity(A.shape[1])

        AI = np.concatenate((A, I))
        Hs = sparse.csc_matrix(H)
        AIs = sparse.csc_matrix(AI)

        if(not np.array_equal(AIs.nonzero(), self.old_AIs.nonzero()) or not np.array_equal(Hs.nonzero(), self.old_Hs.nonzero())):
            return self.init_problem(H, g, A, lb, ub, lbA, ubA)


        self.m.update(Px=Hs.data, q=g, Ax=AIs.data, l=lb, u=ub)

        r = self.m.solve()
        return r.x.T

    #@profile
    def solve(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
        """
        x^T*H*x + x^T*g
        s.t.: lbA < A*x < ubA
        and    lb <  x  < ub
        :param H: 2d diagonal weight matrix, shape = (jc (joint constraints) + sc (soft constraints)) * (jc + sc)
        :type np.array
        :param g: 1d zero vector of len joint constraints + soft constraints
        :type np.array
        :param A: 2d jacobi matrix of hc (hard constraints) and sc, shape = (hc + sc) * (number of joints)
        :type np.array
        :param lb: 1d vector containing lower bound of x, len = jc + sc
        :type np.array
        :param ub: 1d vector containing upper bound of x, len = js + sc
        :type np.array
        :param lbA: 1d vector containing lower bounds for the change of hc and sc, len = hc+sc
        :type np.array
        :param ubA: 1d vector containing upper bounds for the change of hc and sc, len = hc+sc
        :type np.array
        :param nWSR:
        :type np.array
        :return: x according to the equations above, len = joint constraints + soft constraints
        :type np.array
        """
        j_mask = H.sum(axis=1) != 0
        s_mask = j_mask[self.j:]
        h_mask = np.concatenate((np.array([True] * self.h), s_mask))
        A = A[h_mask][:, j_mask].copy()
        #lbA = lbA[h_mask]
        #ubA = ubA[h_mask]
        jh_mask = np.concatenate((h_mask, j_mask))
        lb = lb[jh_mask]
        ub = ub[jh_mask]
        H = H[j_mask][:, j_mask]
        g = np.zeros(H.shape[0])
        if A.shape != self.shape:
            self.started = False
            self.shape = A.shape


        try:
            if(not np.array_equal(self.j_mask, j_mask) or not np.array_equal(self.s_mask, s_mask) or not np.array_equal(self.h_mask, h_mask)):
                self.j_mask = j_mask
                self.s_mask = s_mask
                self.h_mask = h_mask
                self.started = False
        except AttributeError:
            self.j_mask = j_mask
            self.s_mask = s_mask
            self.h_mask = h_mask




        if not self.started:
            return self.init_problem(H, g, A, lb, ub, lbA, ubA)
        else:
            return self.hot_start(H, g, A, lb, ub, lbA, ubA)