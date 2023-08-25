from typing import Optional

import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.goals.monitors.monitor_manager import MonitorManager
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class EvaluateMonitors(GiskardBehavior):
    monitor_manager: MonitorManager

    def __init__(self, name: str = 'evaluate monitors'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    def update(self):
        self.monitor_manager.evaluate_monitors()
        return Status.SUCCESS
