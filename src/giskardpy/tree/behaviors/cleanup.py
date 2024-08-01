import rospy
from py_trees import Status
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy.debug_expression_manager import DebugExpressionManager
from giskardpy.monitors.monitor_manager import MonitorManager
from giskardpy.goals.motion_goal_manager import MotionGoalManager
from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class CleanUp(GiskardBehavior):
    @profile
    def __init__(self, name, clear_markers=True):
        super().__init__(name)
        self.clear_markers_ = clear_markers
        self.marker_pub = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=10)

    def clear_markers(self):
        msg = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        msg.markers.append(marker)
        self.marker_pub.publish(msg)

    @record_time
    @profile
    def initialise(self):
        if self.clear_markers_:
            self.clear_markers()
        god_map.giskard.set_defaults()
        god_map.world.fast_all_fks = None
        god_map.collision_scene.reset_cache()
        god_map.collision_scene.clear_collision_matrix()
        god_map.closest_point = Collisions(1)
        god_map.time = 0
        god_map.control_cycle_counter = 1
        god_map.monitor_manager = MonitorManager()
        god_map.motion_goal_manager = MotionGoalManager()
        god_map.debug_expression_manager = DebugExpressionManager()

        if hasattr(self.get_blackboard(), 'runtime'):
            del self.get_blackboard().runtime

    def update(self):
        return Status.SUCCESS


class CleanUpPlanning(CleanUp):
    def initialise(self):
        super().initialise()
        god_map.fill_trajectory_velocity_values = None
        god_map.free_variables = []

    @catch_and_raise_to_blackboard
    def update(self):
        return super().update()
