import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class CommandPublisher(GiskardBehavior):
    joint_names = []

    @record_time
    @profile
    def __init__(self, name, hz=100):
        super().__init__(name)
        self.hz = hz
        if not hasattr(self, 'joint_names'):
            raise NotImplementedError('you need to set joint names')
        self.world.register_controlled_joints(self.joint_names)
        self.stamp = None

    @profile
    def initialise(self):
        self.sample_period = self.god_map.get_data(identifier.sample_period)
        self.stamp = rospy.get_rostime()
        self.timer = rospy.Timer(period=rospy.Duration(1/self.hz), callback=self.publish_joint_state)
        super().initialise()

    def update(self):
        self.stamp = rospy.get_rostime()
        return Status.RUNNING

    def terminate(self, new_status):
        self.timer.shutdown()

    def publish_joint_state(self, time):
        raise NotImplementedError()
