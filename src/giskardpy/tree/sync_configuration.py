from giskardpy.data_types import JointStates
from giskardpy.model.world import SubWorldTree

from queue import Queue, Empty

import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

import giskardpy.identifier as identifier
from giskardpy.tree.plugin import GiskardBehavior


class SyncConfiguration(GiskardBehavior):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    def __init__(self, name, group_name, joint_state_topic=u'joint_states', tf_root_link_name=None):
        """
        :type js_identifier: str
        """
        super(SyncConfiguration, self).__init__(name)
        self.mjs = None
        self.map_frame = self.get_god_map().unsafe_get_data(identifier.map_frame)
        self.joint_state_topic = joint_state_topic
        self.group_name = group_name
        self.group = self.world.groups[self.group_name]  # type: SubWorldTree
        if tf_root_link_name is None:
            self.tf_root_link_name = self.group.root_link_name
        else:
            self.tf_root_link_name = tf_root_link_name
        self.lock = Queue(maxsize=1)

    def setup(self, timeout=0.0):
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)
        return super(SyncConfiguration, self).setup(timeout)

    def cb(self, data):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    def update(self):
        try:
            if self.mjs is None:
                js = self.lock.get()
            else:
                js = self.lock.get_nowait()
            self.mjs = JointStates.from_msg(js, None)
            self.get_god_map().set_data(identifier.old_joint_states, self.mjs)
        except Empty:
            pass

        self.get_world().state.update(self.mjs)
        return Status.SUCCESS
