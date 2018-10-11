from Queue import Empty, Queue

import rospy
from iai_wsg_50_msgs.msg import Status
from py_trees import Status

from sensor_msgs.msg import JointState

from giskardpy.plugin import NewPluginBase
from giskardpy.utils import to_joint_state_dict


class JointStatePlugin2(NewPluginBase):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    def __init__(self, js_identifier):
        """
        :type js_identifier: str
        """
        super(JointStatePlugin2, self).__init__()
        self.js_identifier = js_identifier
        self.mjs = None
        self.lock = Queue(maxsize=1)

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
        self.god_map.safe_set_data([self.js_identifier], self.mjs)
        return None

    def setup(self):
        self.joint_state_sub = rospy.Subscriber(u'joint_states', JointState, self.cb, queue_size=1)
