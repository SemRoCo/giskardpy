from collections import OrderedDict
from time import time

import numpy as np

from giskardpy import logging, cas_wrapper as w
from giskardpy.data_types import SoftConstraint
from giskardpy.exceptions import QPSolverException
from giskardpy.qp_solver import QPSolver
from giskardpy.utils import make_filter_masks


class QProblemBuilder(object):
    """
    Wraps around QPOases. Builds the required matrices from constraints.
    """

    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict, controlled_joint_symbols,
                 path_to_functions=''):
        """
        :type joint_constraints_dict: dict
        :type hard_constraints_dict: dict
        :type soft_constraints_dict: dict
        :type controlled_joint_symbols: set
        :type free_symbols: set
        :param path_to_functions: location where the compiled functions can be safed.
        :type path_to_functions: str
        """
        assert (not len(controlled_joint_symbols) > len(joint_constraints_dict))
        assert (not len(controlled_joint_symbols) < len(joint_constraints_dict))
        assert (len(hard_constraints_dict) <= len(controlled_joint_symbols))
        self.path_to_functions = path_to_functions
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.controlled_joints = controlled_joint_symbols
        self.construct_big_ass_M()
        self.compile_big_ass_M()

        self.shape1 = len(self.hard_constraints_dict) + len(self.soft_constraints_dict)
        self.shape2 = len(self.joint_constraints_dict) + len(self.soft_constraints_dict)

        self.num_hard_constraints = len(self.hard_constraints_dict)
        self.num_joint_constraints = len(self.joint_constraints_dict)
        self.num_soft_constraints = len(self.soft_constraints_dict)

        self.qp_solver = QPSolver(self.num_hard_constraints,
                                  self.num_joint_constraints,
                                  self.num_soft_constraints)
        self.lbAs = None  # for debugging purposes

    def get_expr(self):
        return self.compiled_big_ass_M.str_params

    def construct_big_ass_M(self):
        # TODO cpu intensive
        weights = []
        lb = []
        ub = []
        lbA = []
        ubA = []
        soft_expressions = []
        hard_expressions = []
        for constraint_name, constraint in self.joint_constraints_dict.items():
            weights.append(constraint.weight)
            lb.append(constraint.lower)
            ub.append(constraint.upper)
        for constraint_name, constraint in self.hard_constraints_dict.items():
            lbA.append(constraint.lower)
            ubA.append(constraint.upper)
            hard_expressions.append(constraint.expression)
        for constraint_name, constraint in self.soft_constraints_dict.items(): # type: (str, SoftConstraint)
            weights.append(constraint.weight)
            lbA.append(constraint.lbA)
            ubA.append(constraint.ubA)
            lb.append(constraint.lb)
            ub.append(constraint.ub)
            assert not w.is_matrix(constraint.expression), u'Matrices are not allowed as soft constraint expression'
            soft_expressions.append(constraint.expression)

        self.np_g = np.zeros(len(weights))

        logging.loginfo(u'constructing new controller with {} soft constraints...'.format(len(soft_expressions)))
        self.h = len(self.hard_constraints_dict)
        self.s = len(self.soft_constraints_dict)
        self.j = len(self.joint_constraints_dict)

        self.init_big_ass_M()

        self.set_weights(weights)

        self.construct_A_hard(hard_expressions)
        self.construct_A_soft(soft_expressions)

        self.set_lbA(w.Matrix(lbA))
        self.set_ubA(w.Matrix(ubA))
        self.set_lb(w.Matrix(lb))
        self.set_ub(w.Matrix(ub))


    def compile_big_ass_M(self):
        t = time()
        self.free_symbols = w.free_symbols(self.big_ass_M)
        self.compiled_big_ass_M = w.speed_up(self.big_ass_M,
                                             self.free_symbols)
        logging.loginfo(u'compiled symbolic expressions in {:.5f}s'.format(time() - t))

    def init_big_ass_M(self):
        """
        #        j           s       1      1
        #    |----------------------------------
        # h  | A hard    |   0    |       |
        #    | -------------------| lbA   | ubA
        # s  | A soft    |identity|       |
        #    |-----------------------------------
        # j+s| H                  | lb    | ub
        #    | ----------------------------------
        """
        self.big_ass_M = w.zeros(self.h + self.s * 2 + self.j,
                                 self.j + self.s + 2)

    def construct_A_hard(self, hard_expressions):
        A_hard = w.Matrix(hard_expressions)
        A_hard = w.jacobian(A_hard, self.controlled_joints)
        self.set_A_hard(A_hard)

    def set_A_hard(self, A_hard):
        self.big_ass_M[:self.h, :self.j] = A_hard

    def construct_A_soft(self, soft_expressions):
        A_soft = w.zeros(self.s, self.j + self.s)
        t = time()
        A_soft[:, :self.j] = w.jacobian(w.Matrix(soft_expressions), self.controlled_joints)
        logging.loginfo(u'computed Jacobian in {:.5f}s'.format(time() - t))
        A_soft[:, self.j:] = w.eye(self.s)
        self.set_A_soft(A_soft)

    def set_A_soft(self, A_soft):
        self.big_ass_M[self.h:self.h + self.s, :self.j + self.s] = A_soft

    def set_lbA(self, lbA):
        self.big_ass_M[:self.h + self.s, self.j + self.s] = lbA

    def set_ubA(self, ubA):
        self.big_ass_M[:self.h + self.s, self.j + self.s + 1] = ubA

    def set_lb(self, lb):
        self.big_ass_M[self.h + self.s:, self.j + self.s] = lb

    def set_ub(self, ub):
        self.big_ass_M[self.h + self.s:, self.j + self.s + 1] = ub

    def set_weights(self, weights):
        self.big_ass_M[self.h + self.s:, :-2] = w.diag(*weights)

    def debug_print(self, unfiltered_H, A, lb, ub, lbA, ubA, xdot_full=None):
        import pandas as pd
        bA_mask, b_mask = make_filter_masks(unfiltered_H, self.num_joint_constraints, self.num_hard_constraints)
        b_names = []
        bA_names = []
        for iJ, k in enumerate(self.joint_constraints_dict.keys()):
            key = 'j -- ' + str(k)
            b_names.append(key)

        for iH, k in enumerate(self.hard_constraints_dict.keys()):
            key = 'h -- ' + str(k)
            bA_names.append(key)
            # upper_bound = ubA[iH]
            # lower_bound = lbA[iH]
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
            bA_names.append(key)
            b_names.append(key)

        b_names = np.array(b_names)
        filtered_b_names = b_names[b_mask]
        filtered_bA_names = np.array(bA_names)[bA_mask]
        filtered_H = unfiltered_H[b_mask][:,b_mask]

        p_lb = pd.DataFrame(lb, filtered_b_names, dtype=float).sort_index()
        p_ub = pd.DataFrame(ub, filtered_b_names, dtype=float).sort_index()
        p_lbA = pd.DataFrame(lbA, filtered_bA_names, dtype=float).sort_index()
        p_ubA = pd.DataFrame(ubA, filtered_bA_names, dtype=float).sort_index()
        p_weights = pd.DataFrame(unfiltered_H.dot(np.ones(unfiltered_H.shape[0])), b_names, dtype=float).sort_index()
        if xdot_full is not None:
            p_xdot = pd.DataFrame(xdot_full, filtered_b_names, dtype=float).sort_index()
            Ax = np.dot(A, xdot_full)
            p_Ax = pd.DataFrame(Ax, filtered_bA_names, dtype=float).sort_index()
            xH = np.dot(xdot_full.T, filtered_H)
            p_xH = pd.DataFrame(xH, filtered_b_names, dtype=float).sort_index()

        p_A = pd.DataFrame(A, filtered_bA_names, filtered_b_names, dtype=float).sort_index(1).sort_index(0)
        # if self.lbAs is None:
        #     self.lbAs = p_lbA
        # else:
        #     self.lbAs = self.lbAs.T.append(p_lbA.T, ignore_index=True).T
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

    def filter_zero_weight_constraints(self, H, A, lb, ub, lbA, ubA):
        bA_mask, b_mask = make_filter_masks(H, self.num_joint_constraints, self.num_hard_constraints)
        A = A[bA_mask][:, b_mask].copy()
        lbA = lbA[bA_mask]
        ubA = ubA[bA_mask]
        lb = lb[b_mask]
        ub = ub[b_mask]
        H = H[b_mask][:, b_mask]
        return H, A, lb, ub, lbA, ubA

    def get_cmd(self, substitutions, nWSR=None):
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        :param substitutions: symbol -> value
        :type substitutions: dict
        :return: joint name -> joint command
        :rtype: dict
        """
        np_big_ass_M = self.compiled_big_ass_M.call2(substitutions)
        np_H = np_big_ass_M[self.shape1:, :-2].copy()
        np_A = np_big_ass_M[:self.shape1, :self.shape2].copy()
        np_lb = np_big_ass_M[self.shape1:, -2].copy()
        np_ub = np_big_ass_M[self.shape1:, -1].copy()
        np_lbA = np_big_ass_M[:self.shape1, -2].copy()
        np_ubA = np_big_ass_M[:self.shape1, -1].copy()
        H, A, lb, ub, lbA, ubA = self.filter_zero_weight_constraints(np_H, np_A, np_lb, np_ub, np_lbA, np_ubA)
        # self.debug_print(np_H, A, lb, ub, lbA, ubA)
        try:
            g = np.zeros(H.shape[0])

            xdot_full = self.qp_solver.solve(H, g, A, lb, ub, lbA, ubA, nWSR)
        except QPSolverException as e:
            self.debug_print(np_H, A, lb, ub, lbA, ubA)
            raise e
        if xdot_full is None:
            return None
        # TODO enable debug print in an elegant way, preferably without slowing anything down
        self.debug_print(np_H, A, lb, ub, lbA, ubA, xdot_full)
        return OrderedDict((observable, xdot_full[i]) for i, observable in enumerate(self.controlled_joints)), \
               np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, xdot_full

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df)
