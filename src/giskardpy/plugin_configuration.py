from Queue import Empty, Queue

import rospy
from sensor_msgs.msg import JointState
from giskardpy.identifier import js_identifier
from giskardpy.plugin import PluginBase
from giskardpy.tfwrapper import lookup_transform, lookup_pose
from giskardpy.utils import to_joint_state_dict


class ConfigurationPlugin(PluginBase):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    def __init__(self, map_frame, joint_state_topic=u'joint_states'):
        """
        :type js_identifier: str
        """
        super(ConfigurationPlugin, self).__init__()
        self.mjs = None
        self.map_frame = map_frame
        self.joint_state_topic = joint_state_topic
        self.lock = Queue(maxsize=1)

    def setup(self):
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)

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

        base_pose = lookup_pose(self.map_frame, self.get_robot().get_root())
        self.get_robot().base_pose = base_pose.pose

        self.god_map.safe_set_data(js_identifier, self.mjs)
        return None
