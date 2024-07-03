from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np


class TimeCollector:
    qp_solver_times: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    separator = ';'

    def add_qp_solve_time(self, class_name, number_variables, number_constraints, time):
        self.qp_solver_times[class_name, number_variables, number_constraints].append(time)

    def print_qp_solver_times(self):
        print('solver, variables, constraints, avg, std, samples')
        for dims, times in sorted(self.qp_solver_times.items()):
            print(self.separator.join([str(dims[0].split(".")[1]),
                                       str(dims[1]),
                                       str(dims[2]),
                                       str(np.average(times)),
                                       str(np.std(times)),
                                       str(times)]))

    def pretty_print(self, filter=None):
        print('-------------------------------------------------')
        self.print_qp_solver_times()
        print('-------------------------------------------------')
