from copy import deepcopy

import rospy
from sensor_msgs.msg import JointState
from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior


class PublishJointState(GiskardBehavior):
    @profile
    def __init__(self, name, namespace):
        super().__init__(name)
        self.namespace = namespace
        self.cmd_topic = f'{self.namespace}'
        self.cmd_pub = rospy.Publisher(self.cmd_topic, JointState, queue_size=10)

    def update(self):
        msg = JointState()
        js = deepcopy(self.world.state)
        for joint_name in js:
            msg.name.append(joint_name.long_name)
            position = js[joint_name].position
            msg.position.append(position)
        msg.header.stamp = rospy.get_rostime()
        self.cmd_pub.publish(msg)
        return Status.RUNNING
