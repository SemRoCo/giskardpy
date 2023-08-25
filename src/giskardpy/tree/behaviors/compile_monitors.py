import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.goals.monitors.monitor_manager import MonitorManager
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class CompileMonitors(GiskardBehavior):
    def __init__(self, name: str = 'init monitor manager'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    def update(self):
        monitor_manager = MonitorManager()
        self.god_map.set_data(identifier.monitor_manager, monitor_manager)
        monitor_manager.compile_monitors()
        return Status.SUCCESS
