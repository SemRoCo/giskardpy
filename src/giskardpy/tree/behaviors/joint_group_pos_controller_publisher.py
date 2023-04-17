from copy import deepcopy

import rospy
from std_msgs.msg import Float64MultiArray

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.tree.behaviors.cmd_publisher import CommandPublisher
from giskardpy.utils.decorators import record_time


class JointGroupPosController(CommandPublisher):
    @record_time
    @profile
    def __init__(self, name, namespace='/joint_group_pos_controller', hz=100):
        self.namespace = namespace
        self.cmd_topic = '{}/command'.format(self.namespace)
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Float64MultiArray, queue_size=10)
        self.joint_names = rospy.get_param('{}/joints'.format(self.namespace))
        super().__init__(name, hz)

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
