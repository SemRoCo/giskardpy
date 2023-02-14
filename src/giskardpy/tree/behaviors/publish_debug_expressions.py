import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

from giskardpy import identifier
from giskardpy.qp.qp_controller import QPController
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class PublishDebugExpressions(GiskardBehavior):
    @profile
    def __init__(self, name, enabled, expression_filter=None, **kwargs):
        super().__init__(name)
        self.expression_filter = expression_filter

    @profile
    def setup(self, timeout):
        self.publisher = rospy.Publisher('~qp_data', JointState, queue_size=1)
        return super().setup(timeout)

    @profile
    def update(self):
        qp_controller: QPController = self.god_map.get_data(identifier.qp_controller)
        p_lbA, p_ubA, p_lb, p_ub, p_weights, p_xdot, p_Ax_without_slack, p_debug = qp_controller.create_debug_pandas2()
        msg = JointState()
        msg.header.stamp = rospy.get_rostime()
        for name, thing in zip(['lbA', 'ubA', 'lb', 'ub', 'weights', 'xdot', 'Ax no slack', 'debug_expressions'],
                         [p_lbA, p_ubA, p_lb, p_ub, p_weights, p_xdot, p_Ax_without_slack, p_debug]):
            msg.name.extend([f'{name}/{x}' for x in thing.index])
            try:
                msg.position.extend(list(thing.values.T[0]))
            except IndexError as e:
                msg.position = []

        self.publisher.publish(msg)
        return Status.RUNNING
