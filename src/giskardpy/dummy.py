#!/usr/bin/env python
import numpy as np
from qpoases import PyQProblem as QProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel

import sympy as sp


def check_sympy_installation(a_value, b_value):
    a = sp.Symbol('a')
    b = sp.Symbol('b')
    f = a ** 2 + b
    f_a = sp.diff(f, a)
    return f_a.subs({a: a_value, b: b_value})


def check_qpoaes(x_start,
                 x_goal,
                 control_constraints_l,
                 control_constraints_u,
                 hard_constraints_l,
                 hard_constraints_u,
                 w_joints,
                 w_tasks):
    weights = np.concatenate((w_joints, w_tasks))
    H = np.diag(weights)
    A = np.array([
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 1.],
    ])
    g = np.zeros(6)
    inf3x = np.ones(3) * 10e6
    lb = np.concatenate((control_constraints_l, -inf3x))
    ub = np.concatenate((control_constraints_u, inf3x))
    lbA = np.concatenate((hard_constraints_l - x_start, x_goal - x_start))
    ubA = np.concatenate((hard_constraints_u - x_start, x_goal - x_start))

    # Setting up QProblem object.

    example = QProblem(len(weights), len(hard_constraints_l) + 3)
    options = Options()
    options.printLevel = PrintLevel.NONE
    example.setOptions(options)

    # Solve first QP.
    nWSR = np.array([10])
    example.init(H, g, A, lb, ub, lbA, ubA, nWSR)

    xOpt = np.zeros(6)
    example.getPrimalSolution(xOpt)
    return xOpt
