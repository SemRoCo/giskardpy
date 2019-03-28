import numpy as np
import pickle
import warnings
from collections import OrderedDict, namedtuple
from time import time

import giskardpy.symengine_wrappers as spw
from giskardpy import BACKEND
from giskardpy.qp_solver import QPSolver
from giskardpy.symengine_wrappers import load_compiled_function, safe_compiled_function

SoftConstraint = namedtuple(u'SoftConstraint', [u'lower', u'upper', u'weight', u'expression'])
HardConstraint = namedtuple(u'HardConstraint', [u'lower', u'upper', u'expression'])
JointConstraint = namedtuple(u'JointConstraint', [u'lower', u'upper', u'weight'])

BIG_NUMBER = 1e9


class QProblemBuilder(object):
    """
    Wraps around QPOases. Builds the required matrices from constraints.
    """

    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict, controlled_joint_symbols,
                 free_symbols=None, path_to_functions=''):
        """
        :type joint_constraints_dict: dict
        :type hard_constraints_dict: dict
        :type soft_constraints_dict: dict
        :type controlled_joint_symbols: set
        :type free_symbols: set
        :param path_to_functions: location where the compiled functions can be safed.
        :type path_to_functions: str
        """
        if free_symbols is not None:
            warnings.warn(u'use of free_symbols deprecated', DeprecationWarning)
        assert (not len(controlled_joint_symbols) > len(joint_constraints_dict))
        assert (not len(controlled_joint_symbols) < len(joint_constraints_dict))
        assert (len(hard_constraints_dict) <= len(controlled_joint_symbols))
        self.path_to_functions = path_to_functions
        self.free_symbols = free_symbols
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.controlled_joints = controlled_joint_symbols
        self.make_matrices()

        self.shape1 = len(self.hard_constraints_dict) + len(self.soft_constraints_dict)
        self.shape2 = len(self.joint_constraints_dict) + len(self.soft_constraints_dict)

        self.qp_solver = QPSolver(len(self.joint_constraints_dict) + len(self.soft_constraints_dict),
                                  len(self.hard_constraints_dict) + len(self.soft_constraints_dict))
        self.lbAs = None  # for debugging purposes

    def get_expr(self):
        return self.cython_big_ass_M.str_params

    # @profile
    def make_matrices(self):
        """
        Turns constrains into a function that computes the matrices needed for QPOases.
        """
        # TODO split this into smaller functions to increase readability
        t_total = time()
        # TODO cpu intensive
        weights = []
        lb = []
        ub = []
        lbA = []
        ubA = []
        soft_expressions = []
        hard_expressions = []
        for k, c in self.joint_constraints_dict.items():
            weights.append(c.weight)
            lb.append(c.lower)
            ub.append(c.upper)
        for k, c in self.hard_constraints_dict.items():
            lbA.append(c.lower)
            ubA.append(c.upper)
            hard_expressions.append(c.expression)
        for k, c in self.soft_constraints_dict.items():
            weights.append(c.weight)
            lbA.append(c.lower)
            ubA.append(c.upper)
            lb.append(-BIG_NUMBER)
            ub.append(BIG_NUMBER)
            assert not isinstance(c.expression, spw.Matrix), u'Matrices are not allowed as soft constraint expression'
            soft_expressions.append(c.expression)

        self.cython_big_ass_M = load_compiled_function(self.path_to_functions)
        self.np_g = np.zeros(len(weights))

        if self.cython_big_ass_M is None:
            print(u'new controller requested; compiling')
            self.H = spw.diag(*weights)

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
            print(u'jacobian took {}'.format(time() - t))
            identity = spw.eye(A_soft.shape[0])
            A_soft = A_soft.row_join(identity)

            # final A
            self.A = A_hard.col_join(A_soft)

            self.lbA = spw.Matrix(lbA)
            self.ubA = spw.Matrix(ubA)

            big_ass_M_A = self.A.row_join(self.lbA).row_join(self.ubA)
            big_ass_M_H = self.H.row_join(self.lb).row_join(self.ub)
            # putting everything into one big matrix to take full advantage of cse in speed_up()
            self.big_ass_M = big_ass_M_A.col_join(big_ass_M_H)

            t = time()
            if self.free_symbols is None:
                self.free_symbols = self.big_ass_M.free_symbols
            self.cython_big_ass_M = spw.speed_up(self.big_ass_M, self.free_symbols, backend=BACKEND)
            if self.path_to_functions is not None:
                safe_compiled_function(self.cython_big_ass_M, self.path_to_functions)
            print(u'autowrap took {}'.format(time() - t))
        else:
            print(u'controller loaded {}'.format(self.path_to_functions))
        print(u'controller ready {}s'.format(time() - t_total))

    def save_pickle(self, hash, f):
        with open(u'/tmp/{}'.format(hash), u'w') as file:
            pickle.dump(f, file)

    def load_pickle(self, hash):
        return pickle.load(hash)

    def debug_print(self, np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, xdot_full=None):
        import pandas as pd
        lb = []
        ub = []
        lbA = []
        ubA = []
        weights = []
        xdot = []
        for iJ, k in enumerate(self.joint_constraints_dict.keys()):
            key = 'j -- ' + str(k)
            lb.append(key)
            ub.append(key)
            weights.append(key)
            xdot.append(key)

        for iH, k in enumerate(self.hard_constraints_dict.keys()):
            key = 'h -- ' + str(k)
            lbA.append(key)
            ubA.append(key)

        for iS, k in enumerate(self.soft_constraints_dict.keys()):
            key = 's -- ' + str(k)
            lbA.append(key)
            ubA.append(key)
            weights.append(key)
            xdot.append(key)
        p_lb = pd.DataFrame(np_lb[:-len(self.soft_constraints_dict)], lb).sort_index()
        p_ub = pd.DataFrame(np_ub[:-len(self.soft_constraints_dict)], ub).sort_index()
        p_lbA = pd.DataFrame(np_lbA, lbA).sort_index()
        p_ubA = pd.DataFrame(np_ubA, ubA).sort_index()
        p_weights = pd.DataFrame(np_H.dot(np.ones(np_H.shape[0])), weights).sort_index()
        if xdot_full is not None:
            p_xdot = pd.DataFrame(xdot_full, xdot).sort_index()
        p_A = pd.DataFrame(np_A, lbA, weights).sort_index(1).sort_index(0)
        if self.lbAs is None:
            self.lbAs = p_lbA
        else:
            self.lbAs = self.lbAs.T.append(p_lbA.T, ignore_index=True).T
            # self.lbAs.T[[c for c in self.lbAs.T.columns if 'dist' in c]].plot()
        pass

    def get_cmd(self, substitutions, nWSR=None):
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        :param substitutions: symbol -> value
        :type substitutions: dict
        :return: joint name -> joint command
        :rtype: dict
        """
        np_big_ass_M = self.cython_big_ass_M(**substitutions)
        # TODO create functions to extract the different matrices.
        np_H = np.array(np_big_ass_M[self.shape1:, :-2])
        np_A = np.array(np_big_ass_M[:self.shape1, :self.shape2])
        np_lb = np.array(np_big_ass_M[self.shape1:, -2])
        np_ub = np.array(np_big_ass_M[self.shape1:, -1])
        np_lbA = np.array(np_big_ass_M[:self.shape1, -2])
        np_ubA = np.array(np_big_ass_M[:self.shape1, -1])
        # self.debug_print(np_H, np_A, np_lb, np_ub, np_lbA, np_ubA)
        xdot_full = self.qp_solver.solve(np_H, self.np_g, np_A, np_lb, np_ub, np_lbA, np_ubA, nWSR)
        if xdot_full is None:
            return None
        # TODO enable debug print in an elegant way, preferably without slowing anything down
        self.debug_print(np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, xdot_full)
        return OrderedDict((observable, xdot_full[i]) for i, observable in enumerate(self.controlled_joints))

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df)
