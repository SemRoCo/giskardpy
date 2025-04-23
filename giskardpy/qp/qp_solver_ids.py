from enum import IntEnum


class SupportedQPSolver(IntEnum):
    qpSWIFT = 1
    gurobi = 2
    clarabel = 3
    qpalm = 4
    # qpOASES = 5
    # osqp = 6
    # quadprog = 7 # manages simple problems, but fails often
    # cplex = 3
    # qp_solvers = 8
    # mosek = 9
    # scs = 11
    # super_csc = 14
    # proxsuite = 16
    # piqp = 17  # does not like inf bounds
    daqp = 18
    # cvxopt = 19
    # qpax   # very slow with qp_solvers, so i didn't bother implementing
    # hpipm # already slow with qp_solver
    # highs # struggles for some reason
