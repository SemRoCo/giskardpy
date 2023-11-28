
from copy import deepcopy

import rospy
from sensor_msgs.msg import JointState
from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior


class PublishJointState(GiskardBehavior):
    @profile
    def __init__(self, name: str, js_topic: str, use_prefix=False):
        super().__init__(name)
        self.use_prefix = use_prefix
        self.cmd_topic = js_topic
        self.cmd_pub = rospy.Publisher(self.cmd_topic, JointState, queue_size=10)

    def update(self):
        msg = JointState()
        js = deepcopy(self.world.state)
        for joint_name in js:
            if 'localization' in joint_name.long_name:
                continue
            if self.use_prefix:
                msg.name.append(joint_name.long_name)
            else:
                msg.name.append(joint_name.short_name)
            position = js[joint_name].position
            msg.position.append(position)
        msg.header.stamp = rospy.get_rostime()
        self.cmd_pub.publish(msg)
        return Status.RUNNING
