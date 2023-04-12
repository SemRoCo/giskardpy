from copy import deepcopy

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict, JointStates
from giskardpy.my_types import Derivatives
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
            self.joint_names[i] = self.world.search_for_joint_name(self.joint_names[i])
        self.world.register_controlled_joints(self.joint_names)
        self.js = None
        self.msg = None
        self.new_stamp = None

    # @profile
    # def setup(self, timeout=0.0):
    #     self.joint_state_sub = rospy.Subscriber('hsrb/joint_states', JointState, self.cb, queue_size=1)
    #     return super().setup(timeout)
    #
    # def cb(self, data):
    #     self.msg = data
    #     self.new_stamp = rospy.get_rostime()

    @profile
    def initialise(self):
        def f(joint_symbol):
            return self.god_map.expr_to_key[joint_symbol][-2]

        self.symbol_to_joint_map = KeyDefaultDict(f)
        self.js = deepcopy(self.world.state)
        super().initialise()

    def publish_joint_state(self, time):
        msg = Float64MultiArray()
        # js = JointStates.from_msg(self.msg, 'hsrb')
        try:
            qp_data = self.god_map.get_data(identifier.qp_solver_solution)
            # dt = (time.current_real - self.new_stamp).to_sec() #- 1 / self.hz
        except Exception:
            return
        for joint_name in self.joint_names:
            try:
                key = self.world.joints[joint_name].free_variables[0].position_name
                velocity = qp_data[Derivatives.velocity][key]
                # acc = qp_data[Derivatives.acceleration][key]
                # jerk = qp_data[Derivatives.jerk][key]
                # delta = velocity * dt #+ (acc * dt**2) / 2 + (jerk * dt**3) / 6

                # position = self.js[joint_name].position + delta * 0.7
                # implicit integration uses position from the current and velocity from the next time step
                # it should give stable solutions irrespective of the time step
                # position = js[joint_name].position + qp_data[Derivatives.velocity][key] * dt
                position = velocity
            except KeyError:
                # position = js[joint_name].position
                position = 0
            msg.data.append(position)
            self.js[joint_name].position = position

        self.cmd_pub.publish(msg)

    def terminate(self, new_status):
        msg = Float64MultiArray()
        for joint_name in self.joint_names:
            msg.data.append(0.0)
        self.cmd_pub.publish(msg)
        super().terminate(new_status)
