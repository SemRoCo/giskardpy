from copy import deepcopy

import rospy
from sensor_msgs.msg import JointState

import giskardpy.identifier as identifier
from giskardpy.god_map_user import GodMap
from giskardpy.tree.behaviors.cmd_publisher import CommandPublisher


class JointStatePublisher(CommandPublisher):
    @profile
    def __init__(self, name, namespace, hz=100):
        self.namespace = namespace
        self.cmd_topic = f'{self.namespace}/command'
        self.cmd_pub = rospy.Publisher(self.cmd_topic, JointState, queue_size=10)
        self.joint_names = rospy.get_param('{}/controlled_joints'.format(self.namespace))
        super().__init__(name, hz)

    def publish_joint_state(self, time):
        msg = JointState()
        js = deepcopy(GodMap.world.state)
        try:
            qp_data = GodMap.god_map.get_data(identifier.qp_solver_solution)
        except Exception:
            return
        for joint_name in self.joint_names:
            msg.name.append(joint_name)
            try:
                key = str(GodMap.god_map.key_to_expr[tuple(identifier.joint_states + [joint_name, 'position'])])
                dt = ((time.current_real - self.stamp).to_sec())# - 1/self.hz)
                # if joint_name == 'neck_shoulder_pan_joint':
                #     print(dt)
                jerk = qp_data[2][key]
                # acceleration = qp_data[1][key]
                # velocity = qp_data[0][key]
                acceleration = js[joint_name].acceleration + jerk * dt
                velocity = js[joint_name].velocity + acceleration * dt
                position = js[joint_name].position + velocity * dt
            except KeyError:
                position = js[joint_name].position
                velocity = js[joint_name].velocity
            msg.position.append(position)
            msg.velocity.append(velocity)
        msg.header.stamp = time.current_real
        self.cmd_pub.publish(msg)
