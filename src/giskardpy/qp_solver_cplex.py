import sys

import numpy as np
import cplex

from giskardpy.exceptions import QPSolverException, InfeasibleException
from giskardpy.utils import logging
from giskardpy.qp.qp_solver import QPSolver

error_info = {
    # https://www.ibm.com/docs/en/icos/20.1.0?topic=manual-cplexexceptionserror-codes
}


class QPSolverCplex(QPSolver):

    @profile
    def init(self, H, g, A, lb, ub, lbA, ubA):
        # TODO potential speed up by reusing model
        self.qpProblem = cplex.Cplex()
        # self.qpProblem.set_problem_type(self.qpProblem.problem_type.QP)
        # self.qpProblem.parameters.barrier.qcpconvergetol.set(1e-10)
        self.qpProblem.objective.set_sense(self.qpProblem.objective.sense.minimize)
        self.set_qp(H, g, A, lb, ub, lbA, ubA)
        # H = sparse.csc_matrix(H)
        # A = sparse.csc_matrix(A)
        self.started = False
        self.qpProblem.set_log_stream(None)
        self.qpProblem.set_results_stream(None)

    def set_qp(self, H, g, A, lb, ub, lbA, ubA):
        x_names = ['x' + str(int(i)) for i in range(0, len(lb))]
        c_names = ['c' + str(int(i)) for i in range(0, len(lbA))]
        # Add vars with limits and linear objective if valid
        self.qpProblem.variables.add(obj=g.tolist() if len(g) == len(lb) else None,
                                     lb=lb.tolist(), ub=ub.tolist(),
                                     types=[self.qpProblem.variables.type.continuous] * len(lb),
                                     names=x_names)
        # Set quadratic objective term
        H_doubled = H * 2.0
        self.qpProblem.objective.set_quadratic(H_doubled.tolist())
        # Add linear constraints
        Gs = ''.join(["G"] * len(lbA))
        Ls = ''.join(["L"] * len(lbA))
        A_with_x_names = np.zeros((A.shape[0], 2, A.shape[1])).tolist()
        for c_i in range(0, A.shape[0]):
            A_with_x_names[c_i][0] = x_names[:]
            A_with_x_names[c_i][1] = A[c_i][:].tolist()
        self.qpProblem.linear_constraints.add(lin_expr=A_with_x_names[:], senses=Gs, rhs=lbA.tolist(),
                                              range_values=np.zeros(len(lbA)).tolist(), names=c_names)
        self.qpProblem.linear_constraints.add(lin_expr=A_with_x_names[:], senses=Ls, rhs=ubA.tolist(),
                                              range_values=np.zeros(len(ubA)).tolist(), names=c_names)

    @profile
    def update(self, H, g, A, lb, ub, lbA, ubA):
        self.qpProblem.linear_constraints.delete()
        self.qpProblem.variables.delete()
        self.set_qp(H, g, A, lb, ub, lbA, ubA)

    def print_debug(self):
        # TODO use MinRHS etc to analyse solution
        # self.qpProblem.set_log_stream(sys.stdout)
        # self.qpProblem.set_results_stream(sys.stdout)
        logging.logwarn(self.qpProblem.solution.get_status_string())
        logging.logwarn(str(self.qpProblem.get_stats()))
        #self.qpProblem.write("log.lp")
        #self.qpProblem.solution.write("solution.lp")
        # self.qpProblem.set_log_stream(None)
        # self.qpProblem.set_results_stream(None)

    def round(self, data, decimal_places):
        return np.round(data, decimal_places)

    @profile
    def solve(self, H, g, A, lb, ub, lbA, ubA, tries=5, decimal_places=4):
        """
        x^T*H*x + x^T*g
        s.t.: lbA < A*x < ubA
        and    lb <  x  < ub
        :param H: 2d diagonal weight matrix, shape = (jc (joint constraints) + sc (soft constraints)) * (jc + sc)
        :type np.array
        :param g: 1d zero vector of len joint constraints + soft constraints
        :type np.array
        :param A: 2d jacobi matrix of hc (hard constraints) and sc, shape = (hc + sc) * (number of joints)
        :type np.array
        :param lb: 1d vector containing lower bound of x, len = jc + sc
        :type np.array
        :param ub: 1d vector containing upper bound of x, len = js + sc
        :type np.array
        :param lbA: 1d vector containing lower bounds for the change of hc and sc, len = hc+sc
        :type np.array
        :param ubA: 1d vector containing upper bounds for the change of hc and sc, len = hc+sc
        :type np.array
        :param nWSR:
        :type np.array
        :return: x according to the equations above, len = joint constraints + soft constraints
        :type np.array
        """
        for i in range(tries):
            if self.started:
                self.update(H, g, A, lb, ub, lbA, ubA)
            else:
                self.init(H, g, A, lb, ub, lbA, ubA)
            self.qpProblem.solve()
            success = self.qpProblem.solution.get_status()
            if 'optimal' in self.qpProblem.solution.get_status_string():
                # if success == gurobipy.GRB.SUBOPTIMAL:
                #    logging.logwarn('warning, suboptimal!')
                self.xdot_full = np.array(self.qpProblem.solution.get_values())
                break
            elif i < tries - 1:
                self.print_debug()
                logging.logwarn(u'Solver returned \'{}\', retrying with data rounded to \'{}\' decimal places'.format(
                    self.qpProblem.solution.get_status_string(),
                    decimal_places
                ))
                H = self.round(H, decimal_places)
                A = self.round(A, decimal_places)
                lb = self.round(lb, decimal_places)
                ub = self.round(ub, decimal_places)
                lbA = self.round(lbA, decimal_places)
                ubA = self.round(ubA, decimal_places)
        else:
            self.print_debug()
            self.started = False
            error_message = u'{}'.format(self.qpProblem.solution.get_status_string())
            if success == self.qpProblem.solution.status.infeasible:
                raise InfeasibleException(error_message)
            raise QPSolverException(error_message)

        return self.xdot_full
