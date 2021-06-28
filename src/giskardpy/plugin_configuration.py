try:
    # Python 2
    from Queue import Empty, Queue
except ImportError:
    # Python 3
    from queue import Queue, Empty

import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.tfwrapper import lookup_pose, wait_for_transform
from giskardpy.utils import to_joint_state_dict


class ConfigurationPlugin(GiskardBehavior):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    def __init__(self, name, joint_state_topic=u'joint_states'):
        """
        :type js_identifier: str
        """
        super(ConfigurationPlugin, self).__init__(name)
        self.mjs = None
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.joint_state_topic = joint_state_topic
        self.lock = Queue(maxsize=1)

    def setup(self, timeout=0.0):
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)
        return super(ConfigurationPlugin, self).setup(timeout)

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
            self.mjs = to_joint_state_dict(js)
        except Empty:
            pass

        robot_frame = self.get_robot().get_root()
        base_pose = lookup_pose(self.map_frame, robot_frame)
        self.get_robot().base_pose = base_pose.pose

        self.god_map.set_data(identifier.joint_states, self.mjs)
        return Status.RUNNING
