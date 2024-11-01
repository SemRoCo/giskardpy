from enum import IntEnum


class SupportedQPSolver(IntEnum):
    qpSWIFT = 1
    qpalm = 2
    gurobi = 3
    clarabel = 4
    # qpOASES = 5
    osqp = 6
    # quadprog = 7
    # cplex = 3
    # cvxopt = 7
    # qp_solvers = 8
    mosek = 9
    scs = 11
    # casadi = 12
    # super_csc = 14
    # cvxpy = 15
    proxsuite = 16
