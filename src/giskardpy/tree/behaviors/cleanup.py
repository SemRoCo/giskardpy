import rospy
from py_trees import Status
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy import identifier
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


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
        self.god_map.clear_cache()
        self.god_map.get_data(identifier.giskard)._reset_config()
        self.god_map.set_data(identifier.goal_msg, None)
        self.world.fast_all_fks = None
        self.collision_scene.reset_cache()
        self.god_map.set_data(identifier.closest_point, Collisions(1))
        # self.get_god_map().safe_set_data(identifier.closest_point, None)
        self.god_map.set_data(identifier.time, 1)

        # to reverse update godmap changes
        # self.get_god_map().set_data(identifier.giskard, deepcopy(self.rosparams))
        # self.world.apply_default_limits_and_weights()
        self.god_map.set_data(identifier.next_move_goal, None)
        if hasattr(self.get_blackboard(), 'runtime'):
            del self.get_blackboard().runtime

    def update(self):
        return Status.SUCCESS


class CleanUpPlanning(CleanUp):
    def initialise(self):
        super().initialise()
        self.god_map.set_data(identifier.fill_trajectory_velocity_values, None)


class CleanUpBaseController(CleanUp):
    pass
