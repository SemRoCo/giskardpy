import rospy
from sensor_msgs.msg._JointState import JointState
from giskardpy.robot import Robot


class RobotRos(Robot):
    def __init__(self, default_joint_value=0.0, default_joint_weight=1.0, urdf_str=None, root_link='base_footprint',
                 tip_links=()):
        super(RobotRos, self).__init__(default_joint_value, default_joint_weight, urdf_str, root_link, tip_links)
        self.joint_sub = rospy.Subscriber('joint_states', JointState, self.joint_state_sub, queue_size=100)

    def joint_state_sub(self, joint_state):
        self.set_joint_state(self.joint_state_msg_to_dict(joint_state))

    def joint_state_msg_to_dict(self, joint_state_msg):
        joint_state_dict = {}
        for i, joint_name in enumerate(joint_state_msg.name):
            joint_state_dict[joint_name] = joint_state_msg.position[i]
        return joint_state_dict

    def joint_vel_dict_to_msg(self, joint_vel_dict):
        joint_vel_msg = JointState()
        for joint_name, joint_vel in joint_vel_dict.items():
            joint_vel_msg.name.append(joint_name)
            joint_vel_msg.velocity.append(joint_vel)
        return joint_vel_msg