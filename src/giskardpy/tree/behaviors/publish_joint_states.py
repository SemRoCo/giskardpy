
from copy import deepcopy
from typing import Optional

import rospy
from sensor_msgs.msg import JointState
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class PublishJointState(GiskardBehavior):
    @profile
    def __init__(self, name: Optional[str] = None, topic_name: Optional[str] = None, include_prefix=False):
        if name is None:
            name = self.__class__.__name__
        if topic_name is None:
            topic_name = '/joint_states'
        super().__init__(name)
        self.include_prefix = include_prefix
        self.cmd_topic = topic_name
        self.cmd_pub = rospy.Publisher(self.cmd_topic, JointState, queue_size=10)

    def update(self):
        msg = JointState()
        js = deepcopy(god_map.world.state)
        for joint_name in js:
            if 'localization' in joint_name.long_name:
                continue
            if self.include_prefix:
                msg.name.append(joint_name.long_name)
            else:
                msg.name.append(joint_name.short_name)
            position = js[joint_name].position
            msg.position.append(position)
        msg.header.stamp = rospy.get_rostime()
        self.cmd_pub.publish(msg)
        return Status.SUCCESS
