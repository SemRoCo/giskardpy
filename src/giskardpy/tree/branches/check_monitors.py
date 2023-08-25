from py_trees import Sequence, Selector

from giskardpy import identifier
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.tree.behaviors.collision_checker import CollisionChecker
from giskardpy.tree.behaviors.curcial_monitors_satisfied import CrucialMonitorsSatisfied
from giskardpy.tree.behaviors.goal_done import GoalDone
from giskardpy.tree.behaviors.instantaneous_controller import ControllerPlugin
from giskardpy.tree.behaviors.kinematic_sim import KinSimPlugin
from giskardpy.tree.behaviors.local_minimum import LocalMinimum
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.loop_detector import LoopDetector
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.behaviors.notify_state_change import NotifyStateChange
from giskardpy.tree.behaviors.real_kinematic_sim import RealKinSimPlugin
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.behaviors.time_real import RosTime
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.send_controls import SendControls
from giskardpy.tree.branches.synchronization import Synchronization
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.tree.control_modes import ControlModes
from giskardpy.tree.decorators import success_is_running


class CheckMonitors(Selector, GodMapWorshipper):

    def __init__(self, name: str = 'check monitors'):
        super().__init__(name)
        self.add_child(CrucialMonitorsSatisfied())
        self.add_child(LoopDetector('loop detector'))
        self.add_child(LocalMinimum('local minimum', real_time=self.is_closed_loop))
        self.add_child(MaxTrajectoryLength('traj length check', real_time=self.is_closed_loop))
        # self.add_child(GoalDone('goal done check'))
