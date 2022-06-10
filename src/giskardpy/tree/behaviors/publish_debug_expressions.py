import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

from giskardpy import identifier
from giskardpy.qp.qp_controller import QPController
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class PublishDebugExpressions(GiskardBehavior):
    def __init__(self, name, enabled, expression_filter=None, **kwargs):
        super().__init__(name)
        self.expression_filter = expression_filter

    def setup(self, timeout):
        self.publisher = rospy.Publisher('~qp_data', JointState, queue_size=1)
        return super().setup(timeout)

    @profile
    def update(self):
        # print('hi')
        debug_pandas = self.god_map.get_data(identifier.debug_expressions_evaluated)
        qp_controller: QPController = self.god_map.get_data(identifier.qp_controller)
        qp_controller._create_debug_pandas()
        msg = JointState()
        msg.header.stamp = rospy.get_rostime()
        msg.name = [f'debug_expressions/{x}' for x in debug_pandas.keys()]
        msg.position = list(debug_pandas.values())
        for name, thing in zip(['lbA', 'ubA', 'lb', 'ub', 'weights', 'xdot'],
                         [qp_controller.p_lbA, qp_controller.p_ubA, qp_controller.p_lb, qp_controller.p_ub,
                          qp_controller.p_weights, qp_controller.p_xdot]):
            msg.name.extend([f'{name}/{x}' for x in thing.index])
            msg.position.extend(list(thing.values.T[0]))

        self.publisher.publish(msg)
        return Status.RUNNING
