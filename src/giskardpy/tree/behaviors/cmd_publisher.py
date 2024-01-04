import rospy
from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class CommandPublisher(GiskardBehavior):
    joint_names = []

    @record_time
    @profile
    def __init__(self, name, hz=100):
        super().__init__(name)
        self.hz = hz
        # if not hasattr(self, 'joint_names'):
        #     raise NotImplementedError('you need to set joint names')
        self.stamp = None

    @profile
    def initialise(self):
        self.timer = rospy.Timer(period=rospy.Duration(1/self.hz), callback=self.publish_joint_state)
        super().initialise()

    def update(self):
        self.stamp = rospy.get_rostime()
        return Status.SUCCESS

    def terminate(self, new_status):
        try:
            self.timer.shutdown()
        except AttributeError as e:
            # terminate might be called before initialise
            pass

    def publish_joint_state(self, time):
        raise NotImplementedError()
