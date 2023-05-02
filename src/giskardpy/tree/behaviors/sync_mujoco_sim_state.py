from queue import Queue, Empty

import rospy
from py_trees import Status
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose

import giskardpy.utils.tfwrapper as tf
from giskardpy.model.utils import make_world_body_box, make_world_body_cylinder, make_world_body_sphere
from giskardpy.model.world import WorldBranch
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class SyncMujocoSim(GiskardBehavior):
    """
    Listens to the visualization_marker_array topic of a mujoco simulation and adds object to the
    Giskard world and updates their position.
    """

    @record_time
    @profile
    def __init__(self, name, group_name='hsrb4s', marker_array_topic='/mujoco/visualization_marker_array', tf_root_link_name=None):
        """
        :type js_identifier: str
        """
        super().__init__(name)
        self.map_frame = tf.get_tf_root()
        self.marker_topic = marker_array_topic
        self.group_name = group_name
        self.group = self.world.groups[self.group_name]  # type: WorldBranch
        if tf_root_link_name is None:
            self.tf_root_link_name = self.group.root_link_name
        else:
            self.tf_root_link_name = tf_root_link_name
        self.markerArray = None
        self.added = []

    @record_time
    @profile
    def setup(self, timeout=0.0):
        self.marker_sub = rospy.Subscriber(self.marker_topic, MarkerArray, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        self.markerArray = data

    @profile
    def initialise(self):
        self.last_time = rospy.get_rostime()
        super().initialise()

    @record_time
    @profile
    def update(self):
        for marker in self.markerArray.markers:
            name = marker.ns + str(marker.id)
            if 'sync' in name:
                if name not in self.added:
                    if marker.type == 1: #CUBE
                        scale = marker.scale
                        obj_body = make_world_body_box(scale.x, scale.y, scale.z)
                        self.world.add_world_body(name, obj_body, marker.pose, marker.header.frame_id)
                        self.added.append(name)
                    elif marker.type == 3: #CYLINDER
                        obj_body = make_world_body_cylinder(marker.scale.x, marker.scale.y/2)
                        self.world.add_world_body(name, obj_body, marker.pose, marker.header.frame_id)
                        self.added.append(name)
                    elif marker.type == 2: #SPHERE
                        obj_body = make_world_body_sphere(marker.scale.x)
                        self.world.add_world_body(name, obj_body, marker.pose, marker.header.frame_id)
                        self.added.append(name)
                else:
                    joint = self.world.joints['connection/' + name]
                    joint.update_transform(marker.pose)
        return Status.RUNNING
