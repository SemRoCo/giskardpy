from inspect import currentframe, getframeinfo

import rospy

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

@profile
def logdebug(msg):
    # generating debug msg in python3 is slow af
    final_msg = generate_msg(msg)
    rospy.logdebug(final_msg)

@profile
def loginfo(msg):
    final_msg = generate_msg(msg)
    rospy.loginfo(final_msg)

def logwarn(msg):
    final_msg = generate_msg(msg)
    rospy.logwarn(final_msg)

def logerr(msg):
    final_msg = generate_msg(msg)
    rospy.logerr(final_msg)

def logfatal(msg):
    final_msg = generate_msg(msg)
    rospy.logfatal(final_msg)

