from queue import Queue, Empty

import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

import giskardpy.utils.tfwrapper as tf
from giskardpy.data_types import JointStates
from giskardpy.model.world import WorldBranch
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class SyncConfiguration2(GiskardBehavior):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    @record_time
    @profile
    def __init__(self, name, group_name, joint_state_topic='joint_states', tf_root_link_name=None):
        """
        :type js_identifier: str
        """
        super().__init__(name)
        self.mjs = None
        self.map_frame = tf.get_tf_root()
        self.joint_state_topic = joint_state_topic
        self.group_name = group_name
        self.group = self.world.groups[self.group_name]  # type: WorldBranch
        if tf_root_link_name is None:
            self.tf_root_link_name = self.group.root_link_name
        else:
            self.tf_root_link_name = tf_root_link_name
        self.lock = Queue(maxsize=1)

    @record_time
    @profile
    def setup(self, timeout=0.0):
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    @profile
    def initialise(self):
        self.last_time = rospy.get_rostime()
        super().initialise()

    @record_time
    @profile
    def update(self):
        try:
            if self.mjs is None:
                js = self.lock.get()
            else:
                js = self.lock.get_nowait()
            dt = (js.header.stamp - self.last_time).to_sec()
            self.mjs = JointStates.from_msg(js, None)
            self.last_time = js.header.stamp
            # self.world.state.update(self.mjs)
            for joint_name, next_state in self.mjs.items():
                # self.world.state[joint_name].acceleration = (next_state.velocity - self.world.state[joint_name].velocity)/dt
                # self.world.state[joint_name].velocity = next_state.velocity
                self.world.state[joint_name].position = next_state.position
            self.world.notify_state_change()
        except Empty:
            pass

        return Status.RUNNING
