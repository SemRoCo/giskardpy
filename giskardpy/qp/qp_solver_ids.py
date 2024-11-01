from enum import IntEnum


class SupportedQPSolver(IntEnum):
    qpSWIFT = 1
    qpalm = 2
    gurobi = 3
    clarabel = 4
    # qpOASES = 5
    osqp = 6
    quadprog = 7
    # cplex = 3
    qp_solvers = 8
    mosek = 9
    scs = 11
    # super_csc = 14
    proxsuite = 16
    piqp = 17
    daqp = 18
