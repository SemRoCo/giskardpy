import numpy as np
from sympy import *

class QPProblemBuilder(object):
    def __init__(self, controllables_lower, controllables_upper, controllables_weights,
                       softs_lower, softs_upper, softs_weight, softs_expressions,
                       hards_lower, hards_upper, hards_expressions, observables, controllables):
        super(QPProblemBuilder, self).__init__(self)

        # Controllable constraint expressions
        if len(controllables_lower) != len(controllables_upper) or len(controllables_lower) != len(controllables_weights):
            raise Exception('There needs to be an equal amount of lower bounds, upper bounds and weights for controllable constraints.')

        self.controllable_lower_bounds = controllables_lower
        self.controllable_upper_bounds = controllables_upper
        self.controllable_weights      = controllables_weights

        # Soft constraint expressions
        if len(softs_expressions) != len(softs_lower) or len(softs_expressions) != len(softs_upper) or len(softs_expressions) != len(softs_weight):
            raise Exception('There needs to be an equal amount of expressions, lower bounds, upper bounds and weights for soft constraints.')

        self.soft_expressions          = softs_expressions
        self.soft_lower_bounds         = softs_lower
        self.soft_upper_bounds         = softs_upper
        self.soft_expressions          = softs_expressions

        # Hard constraint expressions
        if len(hards_expressions) != len(hards_lower) or len(hards_expressions) != len(hards_upper):
            raise Exception('There needs to be an equal amount of expressions, lower bounds and upper bounds for hard constraints.')

        self.hard_expressions          = hards_expressions
        self.hard_lower_bounds         = hards_lower
        self.hard_upper_bounds         = hards_upper

        self.np_H  = np.zeros(self.num_weights(), self.num_weights())
        self.np_A  = np.block([[np.zeros(self.num_hard_constraints(), self.num_hard_constraints()), np.zeros(self.num_soft_constraints(), self.num_hard_constraints())],
                            [np.zeros(self.num_hard_constraints(), self.num_soft_constraints()), np.identity(self.num_soft_constraints())]])

        self.np_g   = np.zeros(self.num_weights())
        self.np_lb  = np.zeros(self.num_weights())
        self.np_ub  = np.zeros(self.num_weights())
        self.np_lbA = np.zeros(self.num_constraints())
        self.np_ubA = np.zeros(self.num_constraints())

        num_c = len(controllables)

        # Determine names of controllables
        self.obs_names = list(observables)

        self.c_obs_names = tuple(controllables)

        # Define symbolic matrix H
        H_diagonal = controllables_weights + softs_weight
        self.H = zeros(len(H_diagonal))
        for x in range(len(H_diagonal)):
            self.H[x*(len(H_diagonal + 1))] = H_diagonal[x]

        Awidth = self.num_weights()
        Aheight = self.num_constraints()
        self.A = zeros(Aheight, Awidth)

        # Fill a with hard constraint derivatives
        for hidx in range(len(hards_expressions)):
            hx = hards_expressions[hidx]
            for cidx in range(num_c):
                cname = self.c_obs_names[cidx]
                if cname in hx.free_symbols:
                    self.A[hidx * Awidth + cidx] = diff(hx, cname, 1)

        # Fill the rest of A with soft constraint derivatives
        AsoftOffset = len(hards_expressions)
        for sidx in range(len(softs_expressions)):
            sx = softs_expressions[sidx]
            for cidx in range(num_c):
                cname = self.c_obs_names[cidx]
                if cname in hx.free_symbols:
                    self.A[(AsoftOffset + sidx) * Awidth + cidx] = diff(sx, cname, 1)

            # Fill the soft constriaint's weight columns with an identity matrix
            self.A[AsoftOffset * Awidth + num_c + (Awidth + 1) * sidx] = 1

        # Construct symbolic lower bound vectors
        self.lb = Matrix(controllables_lower + ([-1e+9] * self.num_soft_constraints()))
        self.ub = Matrix(controllables_upper + ([1e+9] * self.num_soft_constraints()))

        # Construct symbolic upper bound vectors
        self.lbA = Matrix(hards_lower + softs_lower)
        self.ubA = Matrix(hards_upper + softs_upper)



    def update(self, obs_dict):
        self.np_H   = np.array(self.H.subs( obs_dict ).tolist(), dtype=float)
        self.np_A   = np.array(self.A.subs( obs_dict ).tolist(), dtype=float)
        self.np_lb  = np.array(self.lb.subs( obs_dict ).tolist(), dtype=float)
        self.np_ub  = np.array(self.ub.subs( obs_dict ).tolist(), dtype=float)
        self.np_lbA = np.array(self.lbA.subs( obs_dict ).tolist(), dtype=float)
        self.np_ubA = np.array(self.ubA.subs( obs_dict ).tolist(), dtype=float)

    def num_controllables(self):
        return len(self.controllable_weights)

    def num_hard_constraints(self):
        return len(self.hard_expressions)

    def num_soft_constraints(self):
        return len(self.soft_expressions)

    def num_constraints(self):
        return self.num_soft_constraints() + self.num_hard_constraints()

    def num_weights(self):
        return self.num_soft_constraints() + self.num_controllables()

    def get_H(self):
        return self.np_H

    def printInternals(self):
        print('H:\n' + str(self.np_H))
        print('g:'   + str(self.np_g.transpose()))
        print('A:\n' + str(self.np_A))
        print('lb:\n' + str(self.np_lb.transpose()))
        print('ub:\n' + str(self.np_ub.transpose()))
        print('lbA:\n' + str(self.np_lbA.transpose()))
        print('ubA:\n' + str(self.np_ubA.transpose()))
