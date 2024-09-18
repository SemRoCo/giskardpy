from inspect import currentframe, getframeinfo

import rospkg
import rospy
from line_profiler import profile

from giskardpy.middleware import MiddlewareWrapper

rospack = rospkg.RosPack()


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


class ROS1Wrapper(MiddlewareWrapper):

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

    @classmethod
    def resolve_iri(cls, path: str) -> str:
        """
        e.g. 'package://giskardpy/data'
        """
        if 'package://' in path:
            split = path.split('package://')
            prefix = split[0]
            result = prefix
            for suffix in split[1:]:
                package_name, suffix = suffix.split('/', 1)
                real_path = rospack.get_path(package_name)
                result += f'{real_path}/{suffix}'
            return result
        else:
            return path
