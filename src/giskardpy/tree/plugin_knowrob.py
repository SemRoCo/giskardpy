from Queue import Empty, Queue

import pybullet
import rospy
from knowrob_objects.msg import ObjectStateArray, ObjectState
from py_trees import Status
from std_srvs.srv import Trigger, TriggerRequest

from giskardpy import MAP
from giskardpy.exceptions import CorruptShapeException
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import transform_pose
from giskardpy.model.world_object import WorldObject
from giskardpy.utils import logging


class KnowrobPlugin(GiskardBehavior):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    def __init__(self, name, object_state_topic=u'object_state'):
        """
        :type js_identifier: str
        """
        super(KnowrobPlugin, self).__init__(name)
        self.mjs = None
        self.object_state_topic = object_state_topic
        self.lock = Queue()

    def setup(self, timeout=10.0):
        self.object_state_sub = rospy.Subscriber(self.object_state_topic, ObjectStateArray, self.cb, queue_size=1)
        self.get_update_srv = rospy.ServiceProxy(u'/object_state_publisher/update_object_positions', Trigger)
        self.get_update_srv.call(TriggerRequest())
        super(KnowrobPlugin, self).setup(timeout)

    def cb(self, data):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    def update(self):
        try:
            while not self.lock.empty():
                updates = self.lock.get()
                for object_state in updates.object_states:  # type: ObjectState
                    object_name = object_state.object_id
                    if not self.get_world().has_object(object_name):
                        try:
                            world_object = WorldObject.from_object_state(object_state)
                        except CorruptShapeException:
                            # skip objects without visual
                            continue
                        try:
                            self.get_world().add_object(world_object)
                        except pybullet.error as e:
                            logging.logwarn(u'mesh \'{}\' does not exist'.format(object_state.mesh_path))
                            continue
                    pose_in_map = transform_pose(MAP, object_state.pose).pose
                    self.get_world().set_object_pose(object_name, pose_in_map)

        except Empty:
            pass

        return Status.RUNNING
