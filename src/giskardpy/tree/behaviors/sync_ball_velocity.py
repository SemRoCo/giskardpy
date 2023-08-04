from mujoco_msgs.msg import ObjectStateArray

import rospy
from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class SyncBallVel(GiskardBehavior):

    @profile
    def __init__(self, group_name: str, state_topic='/mujoco/object_states'):
        """
        :type js_identifier: str
        """
        super().__init__(str(self))
        self.state_topic = state_topic
        if not self.state_topic.startswith('/'):
            self.ft_topic = '/' + self.state_topic
        super().__init__(str(self))
        self.ball_velocity = None
        self.group_name = group_name

    @profile
    def setup(self, timeout=0.0):
        self.sub = rospy.Subscriber(self.state_topic, ObjectStateArray, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        for state in data.object_states:
            if state.name == 'sync_ball':
                self.ball_velocity = state.velocity.linear

    @profile
    def update(self):
        self.god_map.set_data(identifier.ball_velocity, self.ball_velocity)
        return Status.RUNNING