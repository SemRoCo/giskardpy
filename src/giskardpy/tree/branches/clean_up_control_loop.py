from py_trees import Sequence

from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy.tree.behaviors.goal_cleanup import GoalCleanUp
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.time import TimePlugin


class CleanupControlLoop(Sequence):
    def __init__(self, name: str = 'clean up control loop'):
        super().__init__(name)
        self.add_child(TimePlugin('increase time plan post processing'))
        self.add_child(SetZeroVelocity('set zero vel 1'))
        self.add_child(LogTrajPlugin('log post processing'))
        self.add_child(GoalCleanUp('clean up goals'))
