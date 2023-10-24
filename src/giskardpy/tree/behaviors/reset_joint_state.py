import traceback
from py_trees import Status

from giskardpy.exceptions import InvalidGoalException, GiskardException
from giskardpy.goals.base_traj_follower import BaseTrajFollower
from giskardpy.goals.monitors.monitors import TimeAbove
from giskardpy.god_map import god_map
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.logging import loginfo
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class ResetWorldState(GiskardBehavior):
    @record_time
    @profile
    def __init__(self, name: str = 'reset world state'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        js = god_map.trajectory.get_exact(0)
        god_map.world.state = js
        god_map.world.notify_state_change()
        return Status.SUCCESS

