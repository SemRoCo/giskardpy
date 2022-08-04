from typing import List

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.goals.goal import Goal
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.utils import catch_and_raise_to_blackboard


class SetDriveGoals(GiskardBehavior):
    @profile
    @catch_and_raise_to_blackboard
    def update(self):
        self.god_map.set_data(identifier.goals, {})
        drive_goals: List[Goal] = self.god_map.get_data(identifier.drive_goals)
        for drive_goal in drive_goals:
            drive_goal.save_self_on_god_map()
        return Status.SUCCESS
