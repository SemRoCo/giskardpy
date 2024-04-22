from inspect import currentframe, getframeinfo

import rospy

from giskardpy.middleware.interface.interface import Logger


@profile
def generate_debug_msg(msg):
    node_name = rospy.get_name()
    frameinfo = getframeinfo(currentframe().f_back.f_back)
    file_info = frameinfo.filename.split('/')[-1] + ' line ' + str(frameinfo.lineno)
    new_msg = '\nnode: {}\n file: {}\n message: {}\n'.format(node_name, file_info, msg)
    return new_msg


@profile
def generate_msg(msg):
    node_name = rospy.get_name()
    new_msg = '[{}]: {}'.format(node_name, msg)
    if node_name == '/unnamed':
        print(new_msg)
    return new_msg


class ROS1Logger(Logger):

    @classmethod
    def loginfo(self, msg: str):
        final_msg = generate_msg(msg)
        rospy.loginfo(final_msg)

    @classmethod
    def logwarn(self, msg: str):
        final_msg = generate_msg(msg)
        rospy.logwarn(final_msg)

    @classmethod
    def logerr(self, msg: str):
        final_msg = generate_msg(msg)
        rospy.logerr(final_msg)

    @classmethod
    def logdebug(self, msg: str):
        final_msg = generate_msg(msg)
        rospy.logdebug(final_msg)

    @classmethod
    def logfatal(self, msg: str):
        final_msg = generate_msg(msg)
        rospy.logfatal(final_msg)


