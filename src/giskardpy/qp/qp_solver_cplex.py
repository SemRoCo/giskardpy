import sys

import numpy as np
import cplex

from giskardpy.exceptions import QPSolverException, InfeasibleException
from giskardpy.utils import logging
from giskardpy.qp.qp_solver import QPSolver

error_info = {
    # https://www.ibm.com/docs/en/icos/20.1.0?topic=manual-cplexexceptionserror-codes
}

feasible = [cplex.SolutionInterface.status.feasible, cplex.SolutionInterface.status.feasible_relaxed_sum,
            cplex.SolutionInterface.status.optimal_relaxed_sum, cplex.SolutionInterface.status.feasible_relaxed_inf,
            cplex.SolutionInterface.status.optimal_relaxed_inf, cplex.SolutionInterface.status.feasible_relaxed_quad,
            cplex.SolutionInterface.status.optimal_relaxed_quad, cplex.SolutionInterface.status.first_order,
            cplex.SolutionInterface.status.optimal_tolerance, cplex.SolutionInterface.status.node_limit_feasible,
            cplex.SolutionInterface.status.MIP_feasible_relaxed_sum, cplex.SolutionInterface.status.MIP_optimal_relaxed_sum,
            cplex.SolutionInterface.status.MIP_feasible_relaxed_inf, cplex.SolutionInterface.status.MIP_optimal_relaxed_inf,
            cplex.SolutionInterface.status.MIP_feasible_relaxed_quad, cplex.SolutionInterface.status.MIP_optimal_relaxed_quad,
            cplex.SolutionInterface.status.MIP_feasible, cplex.SolutionInterface.status.multiobj_non_optimal]

optimal = [cplex.SolutionInterface.status.optimal, cplex.SolutionInterface.status.MIP_optimal,
           cplex.SolutionInterface.status.multiobj_optimal]

infeasible = [cplex.SolutionInterface.status.infeasible,  cplex.SolutionInterface.status.infeasible_or_unbounded,
              cplex.SolutionInterface.status.fail_infeasible,  cplex.SolutionInterface.status.fail_infeasible_no_tree,
              cplex.SolutionInterface.status.multiobj_infeasible, cplex.SolutionInterface.status.MIP_optimal_infeasible,
              cplex.SolutionInterface.status.MIP_infeasible_or_unbounded, cplex.SolutionInterface.status.multiobj_infeasible]

limit_infeasible = [cplex.SolutionInterface.status.mem_limit_infeasible, cplex.SolutionInterface.status.node_limit_infeasible]


class QPSolverCplex(QPSolver):

    @profile
    def init(self, H, g, A, lb, ub, lbA, ubA):
        # TODO potential speed up by reusing model
        self.qpProblem = cplex.Cplex()
        self.qpProblem.objective.set_sense(self.qpProblem.objective.sense.minimize)
        self.set_qp(H, g, A, lb, ub, lbA, ubA)
        self.started = False
        self.qpProblem.set_log_stream(None)
        self.qpProblem.set_results_stream(None)

    def set_qp(self, H, g, A, lb, ub, lbA, ubA):
        # Add vars with limits and linear objective if valid
        self.qpProblem.variables.add(obj=g if len(g) == len(lb) else None,
                                     lb=lb, ub=ub,
                                     types=[self.qpProblem.variables.type.continuous] * len(lb))
        # Set quadratic objective term
        H_doubled = H * 2.0
        self.qpProblem.objective.set_quadratic(H_doubled)
        # Add linear constraints
        Gs = ''.join(["G"] * len(lbA))
        Ls = ''.join(["L"] * len(lbA))
        x_inds = list(range(0, len(lb)))
        A_with_x_inds = list(zip([x_inds]*A.shape[0], A))
        self.qpProblem.linear_constraints.add(lin_expr=A_with_x_inds, senses=Gs, rhs=lbA)
        self.qpProblem.linear_constraints.add(lin_expr=A_with_x_inds, senses=Ls, rhs=ubA)

    @profile
    def update(self, H, g, A, lb, ub, lbA, ubA):
        self.qpProblem.linear_constraints.delete()
        self.qpProblem.variables.delete()
        self.set_qp(H, g, A, lb, ub, lbA, ubA)

    def print_debug(self):
        # Print QP problem stats
        logging.logwarn(u'Problem Definition:')
        logging.logwarn(u'Problem Type: {}'.format(self.qpProblem.problem_type[self.qpProblem.get_problem_type()]))
        logging.logwarn(str(self.qpProblem.get_stats()))
        # Print solution type and stats if solution exists
        logging.logwarn(u'Solving method: {}'.format(self.qpProblem.solution.method[self.qpProblem.solution.get_method()]))
        logging.logwarn(u'Solution status: {}'.format(self.qpProblem.solution.get_status_string()))
        if self.qpProblem.solution.get_status() not in infeasible:
            m = self.qpProblem.solution.quality_metric
            arr = self.qpProblem.solution.get_float_quality([m.max_x, m.max_dual_infeasibility])
            if len(arr) == 0:
                arr = self.qpProblem.solution.get_integer_quality([m.max_x, m.max_dual_infeasibility])
            logging.logwarn(u'Solution quality [max, max_dual_infeasibility]: {}'.format(str(arr)))
            logging.logwarn(u'Solution objective value'.format(str(self.qpProblem.solution.get_objective_value())))
        # Write QP problem in prob.lp and solution in solution.lp
        #self.qpProblem.write("prob.lp")
        #self.qpProblem.solution.write("solution.lp")

    def round(self, data, decimal_places):
        return np.round(data, decimal_places)

    @profile
    def solve(self, H, g, A, lb, ub, lbA, ubA, tries=1, decimal_places=4):
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
            if success in optimal or success in feasible:
                if success in feasible:
                    logging.logwarn(u'Solution may be suboptimal!')
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
            if success == cplex.SolutionInterface.status.optimal_infeasible:
                error_message = u'{}: problem is optimally infeasible'.format(self.qpProblem.solution.get_status_string())
                raise InfeasibleException(error_message)
            elif success in infeasible:
                error_message = u'{}: problem is infeasible'.format(self.qpProblem.solution.get_status_string())
                raise InfeasibleException(error_message)
            elif success in limit_infeasible:
                error_message = u'{}: problem is due to limits infeasible'.format(self.qpProblem.solution.get_status_string())
                raise InfeasibleException(error_message)
            raise QPSolverException(u'{}'.format(self.qpProblem.solution.get_status_string()))
        return self.xdot_full
