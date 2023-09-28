import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.goals.monitors.monitor_manager import MonitorManager
from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class CompileMonitors(GiskardBehavior):
    def __init__(self, name: str = 'compile monitors'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        god_map.monitor_manager.compile_monitors()
        return Status.SUCCESS
