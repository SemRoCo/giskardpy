from copy import deepcopy

import rospy
from std_msgs.msg import Float64MultiArray

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.tree.behaviors.cmd_publisher import CommandPublisher


class JointGroupPosController(CommandPublisher):
    @profile
    def __init__(self, namespace='/joint_group_pos_controller', group_name: str = None, hz=100):
        super().__init__(namespace, hz)
        self.namespace = namespace
        self.cmd_topic = f'{self.namespace}/command'
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Float64MultiArray, queue_size=10)
        self.joint_names = rospy.get_param(f'{self.namespace}/joints')
        for i in range(len(self.joint_names)):
            self.joint_names[i] = self.world.get_joint_name(self.joint_names[i])
        self.world.register_controlled_joints(self.joint_names)

    @profile
    def initialise(self):
        def f(joint_symbol):
            return self.god_map.expr_to_key[joint_symbol][-2]

        self.symbol_to_joint_map = KeyDefaultDict(f)
        super().initialise()

    def publish_joint_state(self, time):
        msg = Float64MultiArray()
        js = deepcopy(self.world.state)
        try:
            qp_data = self.god_map.get_data(identifier.qp_solver_solution)
        except Exception:
            return
        for joint_name in self.joint_names:
            try:
                key = str(self.god_map.key_to_expr[tuple(identifier.joint_states + [joint_name, 'position'])])
                velocity = qp_data[0][key]
                position = js[joint_name].position + velocity * ((time.current_real - self.stamp).to_sec() - 1/self.hz)
            except KeyError:
                position = js[joint_name].position
            msg.data.append(position)
        self.cmd_pub.publish(msg)
