from typing import List

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.goals.goal import Goal
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class SetDriveGoals(GiskardBehavior):
    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        self.god_map.set_data(identifier.goals, {})
        drive_goals: List[Goal] = self.god_map.get_data(identifier.drive_goals)
        for drive_goal in drive_goals:
            drive_goal._save_self_on_god_map()
        return Status.SUCCESS
