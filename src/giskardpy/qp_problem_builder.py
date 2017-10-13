import numpy as np
from sympy import *

def printSPMatrix(matrix):
    out = ''
    for y in range(matrix.rows):
        out += '| '
        for x in range(matrix.cols):
           out += '{:>10.10}'.format(str(matrix[y * matrix.cols + x])) + ', '
        out += ' |\n'
    print(out)

class QPProblemBuilder(object):
    def __init__(self, controllables_lower, controllables_upper, controllables_weights,
                       softs_lower, softs_upper, softs_weight, softs_expressions,
                       hards_lower, hards_upper, hards_expressions, observables, controllables):
        super(QPProblemBuilder, self).__init__()

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

        self.np_H  = np.zeros((self.num_weights(), self.num_weights()))
        self.np_A  = np.bmat([[np.zeros((self.num_hard_constraints(), self.num_controllables())), np.zeros((self.num_hard_constraints(), self.num_soft_constraints()))],
                            [np.zeros((self.num_soft_constraints(), self.num_controllables())), np.identity(self.num_soft_constraints())]])

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
            self.H[x*(len(H_diagonal) + 1)] = H_diagonal[x]

        Awidth = self.num_weights()
        Aheight = self.num_constraints()
        self.A = zeros(Aheight, Awidth)

        # Fill a with hard constraint derivatives
        for hidx in range(len(hards_expressions)):
            hx = hards_expressions[hidx]
            for cidx in range(num_c):
                csym = Symbol(self.c_obs_names[cidx])
                if csym in hx.free_symbols:
                    self.A[hidx * Awidth + cidx] = diff(hx, csym, 1)

        # Fill the rest of A with soft constraint derivatives
        AsoftOffset = len(hards_expressions)
        for sidx in range(len(softs_expressions)):
            sx = softs_expressions[sidx]
            for cidx in range(num_c):
                csym = Symbol(self.c_obs_names[cidx])
                if csym in sx.free_symbols:
                    self.A[(AsoftOffset + sidx) * Awidth + cidx] = diff(sx, csym, 1)

            # Fill the soft constriaint's weight columns with an identity matrix
            self.A[AsoftOffset * Awidth + num_c + (Awidth + 1) * sidx] = 1

        # Construct symbolic lower bound vectors
        self.lb = Array(controllables_lower + ([-1e+9] * self.num_soft_constraints()))
        self.ub = Array(controllables_upper + ([1e+9] * self.num_soft_constraints()))

        # Construct symbolic upper bound vectors
        self.lbA = Array(hards_lower + softs_lower)
        self.ubA = Array(hards_upper + softs_upper)



    def update(self, obs_dict):
        nextArray = 'H'
        try:
            self.np_H   = np.array(self.H.subs( obs_dict ).tolist(), dtype=float)
            nextArray = 'A'
            self.np_A   = np.array(self.A.subs( obs_dict ).tolist(), dtype=float)
            nextArray = 'lb'
            self.np_lb  = np.array(self.lb.subs( obs_dict ).tolist(), dtype=float)
            nextArray = 'ub'
            self.np_ub  = np.array(self.ub.subs( obs_dict ).tolist(), dtype=float)
            nextArray = 'lbA'
            self.np_lbA = np.array(self.lbA.subs( obs_dict ).tolist(), dtype=float)
            nextArray = 'ubA'
            self.np_ubA = np.array(self.ubA.subs( obs_dict ).tolist(), dtype=float)
        except Exception as e:
            print('An exception occurred during update of numpy matrix ' + nextArray + ': ' + str(e))
            self.printInternals()
            raise Exception('An exception occurred during update of numpy matrices ' + nextArray + ': ' + str(e))

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

    def printInternals(self):
        print('H: ')
        printSPMatrix(self.H)
        print('A: ')
        printSPMatrix(self.A)
        print('lb: ')
        print(self.lb)
        print('ub: ')
        print(self.ub)
        print('lbA: ')
        print(self.lbA)
        print('ubA: ')
        print(self.ubA)

    def printNPInternals(self):
        print('H:\n' + str(self.np_H))
        print('g:'   + str(self.np_g.transpose()))
        print('A:\n' + str(self.np_A))
        print('lb: ' + str(self.np_lb.transpose()))
        print('ub: ' + str(self.np_ub.transpose()))
        print('lbA: ' + str(self.np_lbA.transpose()))
        print('ubA: ' + str(self.np_ubA.transpose()))
