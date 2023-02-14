from copy import deepcopy

import rospy
from std_msgs.msg import Float64MultiArray, Float64

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.my_types import Derivatives
from giskardpy.tree.behaviors.cmd_publisher import CommandPublisher


class JointPosController(CommandPublisher):
    @profile
    def __init__(self, namespaces=None, group_name: str = None, hz=100):
        super().__init__('joint position publisher', hz)
        self.namespaces = namespaces
        self.publishers = []
        self.cmd_topics = []
        self.joint_names = []
        for namespace in self.namespaces:
            cmd_topic = f'{namespace}/command'
            self.cmd_topics.append(cmd_topic)
            self.publishers.append(rospy.Publisher(cmd_topic, Float64, queue_size=10))
            self.joint_names.append(rospy.get_param(f'{namespace}/joint'))
        for i in range(len(self.joint_names)):
            self.joint_names[i] = self.world.search_for_joint_name(self.joint_names[i])
        self.world.register_controlled_joints(self.joint_names)

    @profile
    def initialise(self):
        def f(joint_symbol):
            return self.god_map.expr_to_key[joint_symbol][-2]

        self.symbol_to_joint_map = KeyDefaultDict(f)
        super().initialise()

    def publish_joint_state(self, time):
        msg = Float64()
        js = deepcopy(self.world.state)
        try:
            qp_data = self.god_map.get_data(identifier.qp_solver_solution)
        except Exception:
            return
        for i, joint_name in enumerate(self.joint_names):
            try:
                key = self.world.joints[joint_name].free_variables[0].position_name
                velocity = qp_data[Derivatives.velocity][key]
                dt = (time.current_real - self.stamp).to_sec()
                dt -= 1/self.hz
                position = js[joint_name].position + velocity * 0.001
            except KeyError:
                position = js[joint_name].position
            msg.data = position
            self.publishers[i].publish(msg)
