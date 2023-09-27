from typing import Optional

import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.goals.monitors.monitor_manager import MonitorManager
from giskardpy.god_map_user import GodMap
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class EvaluateMonitors(GiskardBehavior):

    def __init__(self, name: str = 'evaluate monitors'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        GodMap.get_monitor_manager().evaluate_monitors()
        return Status.SUCCESS
