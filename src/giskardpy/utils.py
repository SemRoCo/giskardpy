from collections import defaultdict, OrderedDict
from sensor_msgs.msg import JointState
from giskardpy.trajectory import SingleJointState


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

#
# CONVERSION FUNCTIONS FOR ROS MESSAGES
#


def to_joint_state_dict(msg):
    """
    Converts a ROS message of type sensor_msgs/JointState into an instance of MultiJointState.
    :param msg: ROS message to convert.
    :type msg: JointState
    :return: Corresponding MultiJointState instance.
    :rtype: OrderedDict[str, SingleJointState]
    """
    mjs = OrderedDict()
    for i, joint_name in enumerate(msg.name):
        sjs = SingleJointState()
        sjs.name = joint_name
        sjs.position = msg.position[i]
        sjs.velocity = msg.velocity[i]
        sjs.effort = msg.effort[i]
        mjs[joint_name] = sjs
    return mjs
