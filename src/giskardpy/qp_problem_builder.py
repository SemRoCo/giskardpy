import numpy as np
import sympy as sp


class QProblemBuilder(object):
    def __init__(self, controller):
        self.controller = controller

        self.observables = sp.Matrix(self.controller.get_observables())

        self.H = sp.diag(*self.controller.get_weights())

        self.np_g = np.zeros(self.get_num_controllables() + self.get_num_soft_constraints())

        self.lb = sp.Matrix(self.controller.get_lb())
        self.ub = sp.Matrix(self.controller.get_ub())

        # make A
        # hard part
        A_hard = sp.Matrix(self.controller.get_hard_expressions())
        A_hard = A_hard.jacobian(self.controller.get_robot_observables())
        zeros3x3 = sp.zeros(*A_hard.shape)
        A_hard = zeros3x3.col_insert(0, A_hard)

        # soft part
        A_soft = sp.Matrix(self.controller.get_soft_expressions())
        A_soft = A_soft.jacobian(self.controller.get_controller_observables())
        identity3x3 = sp.eye(3)
        A_soft = identity3x3.col_insert(0, A_soft)

        # final A
        self.A = A_soft.row_insert(0, A_hard)

        self.lbA = sp.Matrix(self.controller.get_lbA())
        self.ubA = sp.Matrix(self.controller.get_ubA())

    def get_problem_dimensions(self):
        return len(self.controller.get_weights()), len(self.get_lb())

    def get_num_controllables(self):
        return self.controller.get_num_controllables()

    def get_num_soft_constraints(self):
        return len(self.controller.get_soft_expressions())

    def update(self):
        updates_dict = self.controller.get_updates()
        self.update_H(updates_dict)
        self.update_lb(updates_dict)
        self.update_ub(updates_dict)
        self.update_lbA(updates_dict)
        self.update_ubA(updates_dict)
        self.update_A(updates_dict)

    def update_expression_matrix(self, matrix, updates_dict):
        m_sub = matrix.subs(updates_dict)
        return np.array(m_sub.tolist(), dtype=float)

    def update_expression_vector(self, vector, updates_dict):
        np_v = self.update_expression_matrix(vector, updates_dict)
        return np_v.reshape(len(np_v))

    def update_H(self, updates_dict):
        self.np_H = self.update_expression_matrix(self.H, updates_dict)

    def get_H(self):
        return self.np_H

    def update_A(self, updates_dict):
        self.np_A = self.update_expression_matrix(self.A, updates_dict)

    def get_A(self):
        return self.np_A

    def update_lb(self, updates_dict):
        self.np_lb = self.update_expression_vector(self.lb, updates_dict)

    def get_lb(self):
        return self.np_lb

    def update_ub(self, updates_dict):
        self.np_ub = self.update_expression_vector(self.ub, updates_dict)

    def get_ub(self):
        return self.np_ub

    def update_lbA(self, updates_dict):
        self.np_lbA = self.update_expression_vector(self.lbA, updates_dict)

    def get_lbA(self):
        return self.np_lbA

    def update_ubA(self, updates_dict):
        self.np_ubA = self.update_expression_vector(self.ubA, updates_dict)

    def get_ubA(self):
        return self.np_ubA

    def get_g(self):
        return self.np_g
