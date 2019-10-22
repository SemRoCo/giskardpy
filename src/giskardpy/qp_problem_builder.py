import numpy as np
import pickle
import warnings
from collections import OrderedDict, namedtuple
from time import time

# from giskardpy import BACKEND
from giskardpy import logging, w
from giskardpy.exceptions import QPSolverException
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
                 free_symbols=None, path_to_functions='', backend='llvm', opt_level=0):
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
        self.make_matrices(backend, opt_level)

        self.shape1 = len(self.hard_constraints_dict) + len(self.soft_constraints_dict)
        self.shape2 = len(self.joint_constraints_dict) + len(self.soft_constraints_dict)

        self.qp_solver = QPSolver(len(self.joint_constraints_dict) + len(self.soft_constraints_dict),
                                  len(self.hard_constraints_dict) + len(self.soft_constraints_dict))
        self.lbAs = None  # for debugging purposes

    def get_expr(self):
        return self.cython_big_ass_M.str_params

    #
    def make_matrices(self, backend='llvm', opt_level=0):
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
            assert not w.is_matrix(c.expression), u'Matrices are not allowed as soft constraint expression'
            soft_expressions.append(c.expression)

        self.cython_big_ass_M = load_compiled_function(self.path_to_functions)
        self.np_g = np.zeros(len(weights))

        if self.cython_big_ass_M is None:
            logging.loginfo(u'new controller with {} constraints requested; compiling'.format(len(soft_expressions)))
            h = len(self.hard_constraints_dict)
            s = len(self.soft_constraints_dict)
            c = len(self.joint_constraints_dict)

            #       c           s       1      1
            #   |----------------------------------
            # h | A hard    |   0    |       |
            #   | -------------------| lbA   | ubA
            # s | A soft    |identity|       |
            #   |-----------------------------------
            #c+s| H                  | lb    | ub
            #   | ----------------------------------
            self.big_ass_M = w.zeros(h+s+s+c, c+s+2)

            self.big_ass_M[h+s:,:-2] = w.diag(*weights)

            self.lb = w.Matrix(lb)
            self.ub = w.Matrix(ub)

            # make A
            # hard part
            A_hard = w.Matrix(hard_expressions)
            A_hard = w.jacobian(A_hard, self.controlled_joints)
            self.big_ass_M[:h, :c] = A_hard

            # soft part
            A_soft = w.Matrix(soft_expressions)
            t = time()
            A_soft = w.jacobian(A_soft, self.controlled_joints)
            logging.loginfo(u'jacobian took {}'.format(time() - t))
            identity = w.eye(A_soft.shape[0])
            self.big_ass_M[h:h+s, :c] = A_soft
            self.big_ass_M[h:h+s, c:c+s] = identity


            self.lbA = w.Matrix(lbA)
            self.ubA = w.Matrix(ubA)

            self.big_ass_M[:h+s, c+s] = self.lbA
            self.big_ass_M[:h+s, c+s+1] = self.ubA
            self.big_ass_M[h+s:, c+s] = self.lb
            self.big_ass_M[h+s:, c+s+1] = self.ub

            t = time()
            if self.free_symbols is None:
                self.free_symbols = w.free_symbols(self.big_ass_M)
            self.cython_big_ass_M = w.speed_up(self.big_ass_M,
                                               self.free_symbols)
            if self.path_to_functions is not None:
                # safe_compiled_function(self.cython_big_ass_M, self.path_to_functions)
                logging.loginfo(u'autowrap took {}'.format(time() - t))
        else:
            logging.loginfo(u'controller loaded {}'.format(self.path_to_functions))
            logging.loginfo(u'controller ready {}s'.format(time() - t_total))

    def save_pickle(self, hash, f):
        with open(u'/tmp/{}'.format(hash), u'w') as file:
            pickle.dump(f, file)

    def load_pickle(self, hash):
        return pickle.load(hash)

    def debug_print(self, np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, xdot_full=None):
        import pandas as pd
        lb = []
        lbA = []
        weights = []
        xdot = []
        if xdot_full is not None:
            A_dot_x = np_A.dot(xdot_full)
        for iJ, k in enumerate(self.joint_constraints_dict.keys()):
            key = 'j -- ' + str(k)
            lb.append(key)
            weights.append(key)
            xdot.append(key)

        for iH, k in enumerate(self.hard_constraints_dict.keys()):
            key = 'h -- ' + str(k)
            lbA.append(key)
            # upper_bound = np_ubA[iH]
            # lower_bound = np_lbA[iH]
            # if np.sign(upper_bound) == np.sign(lower_bound):
            #     logging.logwarn(u'{} out of bounds'.format(k))
            #     if upper_bound > 0:
            #         logging.logwarn(u'{} value below lower bound by {}'.format(k, lower_bound))
            #         vel = np_ub[iH]
            #         if abs(vel) < abs(lower_bound):
            #             logging.logerr(u'joint vel of {} to low to get back into bound in one iteration'.format(vel))
            #     else:
            #         logging.logwarn(u'{} value above upper bound by {}'.format(k, abs(upper_bound)))
            #         vel = np_lb[iH]
            #         if abs(vel) < abs(lower_bound):
            #             logging.logerr(u'joint vel of {} to low to get back into bound in one iteration'.format(vel))

        for iS, k in enumerate(self.soft_constraints_dict.keys()):
            key = 's -- ' + str(k)
            lbA.append(key)
            weights.append(key)
            xdot.append(key)
        p_lb = pd.DataFrame(np_lb[:-len(self.soft_constraints_dict)], lb).sort_index()
        p_ub = pd.DataFrame(np_ub[:-len(self.soft_constraints_dict)], lb).sort_index()
        p_lbA = pd.DataFrame(np_lbA, lbA).sort_index()
        if xdot_full is not None:
            p_A_dot_x = pd.DataFrame(A_dot_x, lbA).sort_index()
        p_ubA = pd.DataFrame(np_ubA, lbA).sort_index()
        p_weights = pd.DataFrame(np_H.dot(np.ones(np_H.shape[0])), weights).sort_index()
        if xdot_full is not None:
            p_xdot = pd.DataFrame(xdot_full, xdot).sort_index()
        p_A = pd.DataFrame(np_A, lbA, weights).sort_index(1).sort_index(0)
        if self.lbAs is None:
            self.lbAs = p_lbA
        else:
            self.lbAs = self.lbAs.T.append(p_lbA.T, ignore_index=True).T
            # self.lbAs.T[[c for c in self.lbAs.T.columns if 'dist' in c]].plot()
        # arrays = [(p_weights, u'H'),
        #           (p_A, u'A'),
        #           (p_lbA, u'lbA'),
        #           (p_ubA, u'ubA'),
        #           (p_lb, u'lb'),
        #           (p_ub, u'ub')]
        # for a, name in arrays:
        #     self.check_for_nan(name, a)
        #     self.check_for_big_numbers(name, a)
        pass

    def check_for_nan(self, name, p_array):
        p_filtered = p_array.apply(lambda x: zip(x.index[x.isnull()].tolist(), x[x.isnull()]), 1)
        p_filtered = p_filtered[p_filtered.apply(lambda x: len(x)) > 0]
        if len(p_filtered) > 0:
            logging.logwarn(u'{} has the following nans:'.format(name))
            self.print_pandas_array(p_filtered)
        else:
            logging.loginfo(u'no nans')

    def check_for_big_numbers(self, name, p_array, big=1e5):
        # FIXME fails if condition is true on first entry
        p_filtered = p_array.apply(lambda x: zip(x.index[abs(x) > big].tolist(), x[x > big]), 1)
        p_filtered = p_filtered[p_filtered.apply(lambda x: len(x)) > 0]
        if len(p_filtered) > 0:
            logging.logwarn(u'{} has the following big numbers:'.format(name))
            self.print_pandas_array(p_filtered)
        else:
            logging.loginfo(u'no big numbers')

    def print_pandas_array(self, array):
        import pandas as pd
        if len(array) > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(array)

    def get_cmd(self, substitutions, nWSR=None):
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        :param substitutions: symbol -> value
        :type substitutions: dict
        :return: joint name -> joint command
        :rtype: dict
        """
        np_big_ass_M = self.cython_big_ass_M.call2(substitutions)
        np_H = np_big_ass_M[self.shape1:, :-2].copy()
        np_A = np_big_ass_M[:self.shape1, :self.shape2].copy()
        np_lb = np_big_ass_M[self.shape1:, -2].copy()
        np_ub = np_big_ass_M[self.shape1:, -1].copy()
        np_lbA = np_big_ass_M[:self.shape1, -2].copy()
        np_ubA = np_big_ass_M[:self.shape1, -1].copy()
        # self.debug_print(np_H, np_A, np_lb, np_ub, np_lbA, np_ubA)
        try:
            xdot_full = self.qp_solver.solve(np_H, self.np_g, np_A, np_lb, np_ub, np_lbA, np_ubA, nWSR)
        except QPSolverException as e:
            self.debug_print(np_H, np_A, np_lb, np_ub, np_lbA, np_ubA)
            raise e
        if xdot_full is None:
            return None
        # TODO enable debug print in an elegant way, preferably without slowing anything down
        self.debug_print(np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, xdot_full)
        return OrderedDict((observable, xdot_full[i]) for i, observable in enumerate(self.controlled_joints)), \
               np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, xdot_full

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df)
