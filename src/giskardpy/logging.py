import rospy
from inspect import currentframe, getframeinfo
from giskardpy import identifier

@profile
def debug():
    try:
        if debug.param == None:
            l = identifier.debug
            param_name = '/giskard/' + '/'.join(s for s in l[1:])
            debug.param = rospy.get_param(param_name)
            return debug.param
        else:
            return debug.param
    except KeyError:
        pass
    return False

debug.param = None

# @profile
def generate_debug_msg(msg):
    node_name = rospy.get_name()
    frameinfo = getframeinfo(currentframe().f_back.f_back)
    file_info = frameinfo.filename.split('/')[-1] + ' line ' + str(frameinfo.lineno)
    new_msg = '\nnode: {}\n file: {}\n message: {}\n'.format(node_name, file_info, msg)
    return new_msg

@profile
def generate_msg(msg):
    if(debug()):
        return generate_debug_msg(msg)
    else:
        node_name = rospy.get_name()
        new_msg = '[{}]: {}'.format(node_name, msg)
        return new_msg

# @profile
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

