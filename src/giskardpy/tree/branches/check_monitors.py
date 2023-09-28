from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.curcial_monitors_satisfied import CrucialMonitorsSatisfied
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.composites.running_selector import RunningSelector


class CheckMonitors(RunningSelector):

    def __init__(self, name: str = 'check monitors'):
        super().__init__(name)
        self.add_child(CrucialMonitorsSatisfied())
        # self.add_child(LoopDetector('loop detector'))
        # self.add_child(LocalMinimum('local minimum', real_time=god_map.is_closed_loop))
        self.add_child(MaxTrajectoryLength('traj length check', real_time=god_map.is_closed_loop()))
        # self.add_child(GoalDone('goal done check'))
