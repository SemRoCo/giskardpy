from copy import deepcopy

import rospy
from py_trees import Status
from rospy.timer import TimerEvent
from std_msgs.msg import Float64MultiArray, Float64

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.my_types import Derivatives
from giskardpy.tree.behaviors.cmd_publisher import CommandPublisher
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class JointVelController(GiskardBehavior):
    last_time: float

    @record_time
    @profile
    def __init__(self, namespaces=None):
        super().__init__('joint velocity publisher')
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

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        # next_time = self.god_map.get_data(identifier.time)
        # if next_time <= 0.0 or not hasattr(self, 'last_time'):
        #     self.last_time = next_time
        #     return Status.RUNNING
        # # if self.last_time is None:
        # next_cmds = self.god_map.get_data(identifier.qp_solver_solution)
        # # joints = self.world.joints
        # # next_time = rospy.get_rostime()
        # dt = next_time - self.last_time
        # print(f'dt: {dt}')
        # self.world.update_state(next_cmds, dt)
        # self.last_time = next_time
        # self.world.notify_state_change()

        # next_cmds = self.god_map.get_data(identifier.qp_solver_solution)
        # self.world.update_state(next_cmds, self.sample_period)
        msg = Float64()
        # js = deepcopy(self.world.state)
        # try:
        #     qp_data = self.god_map.get_data(identifier.qp_solver_solution)
        #     if qp_data is None:
        #         return
        # except Exception:
        #     return
        # try:
        #     dt = (time.current_real - time.last_real).to_sec()
        # except:
        #     dt = 0
        for i, joint_name in enumerate(self.joint_names):
            msg.data = self.world.state[joint_name].velocity
            self.publishers[i].publish(msg)
        return Status.RUNNING

    def terminate(self, new_status):
        super().terminate(new_status)
        msg = Float64()
        for i, joint_name in enumerate(self.joint_names):
            self.publishers[i].publish(msg)


