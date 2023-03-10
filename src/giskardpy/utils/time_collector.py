from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np

from giskardpy import identifier
from giskardpy.god_map import GodMap


class TimeCollector:
    qp_solver_times: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)

    def __init__(self):
        self.god_map = GodMap()

    def add_qp_solve_time(self, class_name, number_variables, number_constraints, time):
        self.qp_solver_times[class_name, number_variables, number_constraints].append(time)

    def print_qp_solver_times(self):
        for dims, times in sorted(self.qp_solver_times.items()):
            print(f'{dims}: {np.average(times)}')

    def pretty_print(self, filter=None):
        print('-------------------------------------------------')
        self.print_qp_solver_times()
        print('-------------------------------------------------')
