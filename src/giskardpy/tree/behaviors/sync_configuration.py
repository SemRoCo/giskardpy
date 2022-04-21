from rospy import ROSException

from giskardpy.data_types import JointStates, PrefixName
from giskardpy.model.world import SubWorldTree

from queue import Queue, Empty

import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging


class SyncConfiguration(GiskardBehavior):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    @profile
    def __init__(self, name, group_name, prefix=None, joint_state_topic='joint_states', tf_root_link_name=None):
        """
        :type js_identifier: str
        """
        super(SyncConfiguration, self).__init__(name)
        self.mjs = None
        self.prefix = prefix
        self.map_frame = self.get_god_map().unsafe_get_data(identifier.map_frame)
        if self.prefix is not None:
            self.joint_state_topic = u'/{}'.format(PrefixName(joint_state_topic, self.prefix))
        else:
            self.joint_state_topic = joint_state_topic
        self.group_name = group_name
        self.group = self.world.groups[self.group_name]  # type: SubWorldTree
        if tf_root_link_name is None:
            self.tf_root_link_name = self.group.root_link_name
        else:
            self.tf_root_link_name = tf_root_link_name
        self.lock = Queue(maxsize=1)

    @profile
    def setup(self, timeout=0.0):
        msg = None
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(self.joint_state_topic, JointState, rospy.Duration(1))
                self.lock.put(msg)
            except ROSException as e:
                logging.logwarn('Waiting for topic \'/{}\' to appear.'.format(self.joint_state_topic))
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)
        return super(SyncConfiguration, self).setup(timeout)

    def cb(self, data):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    @profile
    def update(self):
        try:
            if self.mjs is None:
                js = self.lock.get()
            else:
                js = self.lock.get_nowait()
            self.mjs = JointStates.from_msg(js, self.prefix)
        except Empty:
            pass

        self.get_world().state.update(self.mjs)
        self.world.notify_state_change()
        return Status.RUNNING
