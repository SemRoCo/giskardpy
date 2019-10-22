import rospy
from rosgraph_msgs.msg import Log

jac = 0
compiling = 0
plan = 0

def cb(data):
    """
    :type data: Log
    """
    global jac
    global compiling
    global plan
    if data.name == '/tests':
        msg = data.msg
        if 'jacobian' in msg:
            jac += float(msg.split('jacobian took ')[-1])
        elif 'autowrap' in msg:
            compiling += float(msg.split('autowrap took ')[-1])
        elif 'found goal traj' in msg:
            plan += float(msg.split(' in ')[-1][:-1])


rospy.init_node('asdf', anonymous=True)
s = rospy.Subscriber('/rosout', Log, cb, queue_size=10)
raw_input('kill?')
print(jac)
print(compiling)
print(plan-jac-compiling)