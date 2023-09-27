from typing import List

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.goals.goal import Goal
from giskardpy.god_map_user import GodMap
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class SetDriveGoals(GiskardBehavior):
    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        GodMap.god_map.set_data(identifier.motion_goals, {})
        drive_goals: List[Goal] = GodMap.god_map.get_data(identifier.drive_goals)
        return Status.SUCCESS
