from py_trees import Sequence

from giskardpy import identifier
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.tree.behaviors.collision_checker import CollisionChecker
from giskardpy.tree.behaviors.instantaneous_controller import ControllerPlugin
from giskardpy.tree.behaviors.kinematic_sim import KinSimPlugin
from giskardpy.tree.behaviors.local_minimum import LocalMinimum
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.loop_detector import LoopDetector
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.tree.decorators import success_is_running


class ControlLoopBranch(AsyncBehavior, GodMapWorshipper):
    publish_state: PublishState

    def __init__(self, name: str = 'control_loop'):
        super().__init__(name)
        self.publish_state = success_is_running(PublishState)('publish state 2')
        stoppers = success_is_running(Sequence)('stop conditions')
        stoppers.add_child(LoopDetector('loop detector'))
        stoppers.add_child(LocalMinimum('local minimum'))

        if self.god_map.get_data(identifier.collision_checker) != CollisionCheckerLib.none:
            self.add_child(success_is_running(CollisionChecker)('collision checker'))
        self.add_child(ControllerPlugin('controller'))
        self.add_child(success_is_running(KinSimPlugin)('kin sim'))
        self.add_child(success_is_running(LogTrajPlugin)('log closed loop control'))
        self.add_child(stoppers)
        self.add_child(success_is_running(TimePlugin)('increase time closed loop'))
        self.add_child(success_is_running(MaxTrajectoryLength)('traj length check'))
        self.add_child(self.publish_state)

