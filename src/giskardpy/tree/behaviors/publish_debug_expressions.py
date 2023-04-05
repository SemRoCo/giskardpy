from copy import deepcopy

import numpy as np
import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

from giskardpy import identifier
from giskardpy.data_types import JointStates
from giskardpy.model.trajectory import Trajectory
from giskardpy.qp.qp_controller import QPProblemBuilder
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class PublishDebugExpressions(GiskardBehavior):
    @profile
    def __init__(self, name, publish_lb: bool = False, publish_ub: bool = False, publish_lbA: bool = False,
                 publish_ubA: bool = False, publish_Ax: bool = False, publish_xdot: bool = False,
                 publish_weights: bool = False, publish_debug: bool = False, **kwargs):
        super().__init__(name)
        self.publish_lb = publish_lb
        self.publish_ub = publish_ub
        self.publish_lbA = publish_lbA
        self.publish_ubA = publish_ubA
        self.publish_weights = publish_weights
        self.publish_Ax = publish_Ax
        self.publish_xdot = publish_xdot
        self.publish_debug = publish_debug

    @profile
    def setup(self, timeout):
        self.publisher = rospy.Publisher('~qp_data', JointState, queue_size=1)
        return super().setup(timeout)

    @profile
    def create_msg(self, qp_controller: QPProblemBuilder):
        # FIXME
        msg = JointState()
        msg.header.stamp = rospy.get_rostime()

        sample_period = self.god_map.get_data(identifier.sample_period)
        b_names = qp_controller.b_names()
        bA_names = qp_controller.bA_names()
        filtered_b_names = np.array(b_names)[qp_controller.b_filter]
        filtered_bA_names = np.array(bA_names)[qp_controller.bA_filter]
        num_vel_constr = len(qp_controller.derivative_constraints) * (qp_controller.prediction_horizon - 2)
        num_task_constr = len(qp_controller.constraints)
        num_constr = num_vel_constr + num_task_constr

        if self.publish_debug:
            for name, value in qp_controller.evaluated_debug_expressions.items():
                if isinstance(value, np.ndarray):
                    if len(value) > 1:
                        for x in range(value.shape[0]):
                            for y in range(value.shape[1]):
                                tmp_name = f'{name}|{x}_{y}'
                                msg.name.append(tmp_name)
                                msg.position.append(value[x, y])
                    else:
                        msg.name.append(name)
                        msg.position.append(value.flatten())
                else:
                    msg.name.append(name)
                    msg.position.append(value)

        if self.publish_lb:
            lb_names = [f'lb/{entry_name}' for entry_name in filtered_b_names]
            msg.name.extend(lb_names)
            msg.position.extend(qp_controller.np_lb_filtered.tolist())

        if self.publish_ub:
            ub_names = [f'ub/{entry_name}' for entry_name in filtered_b_names]
            msg.name.extend(ub_names)
            msg.position.extend(qp_controller.np_ub_filtered.tolist())

        if self.publish_lbA:
            lbA_names = [f'lbA/{entry_name}' for entry_name in filtered_bA_names]
            msg.name.extend(lbA_names)
            msg.position.extend(qp_controller.np_lbA_filtered.tolist())

        if self.publish_ubA:
            ubA_names = [f'ubA/{entry_name}' for entry_name in filtered_bA_names]
            msg.name.extend(ubA_names)
            msg.position.extend(qp_controller.np_ubA_filtered.tolist())

        if self.publish_weights:
            weight_names = [f'weights/{entry_name}' for entry_name in b_names]
            msg.name.extend(weight_names)
            msg.position.extend(qp_controller.np_weights.tolist())

        if self.publish_xdot:
            xdot_names = [f'xdot/{entry_name}' for entry_name in filtered_b_names]
            msg.name.extend(xdot_names)
            msg.position.extend(qp_controller.xdot_full.tolist())

        if self.publish_Ax:
            Ax_names = [f'Ax/{entry_name}' for entry_name in filtered_bA_names]
            msg.name.extend(Ax_names)
            pure_xdot = qp_controller.xdot_full.copy()
            pure_xdot[-num_constr:] = 0
            Ax_without_slack = qp_controller.np_A_filtered.dot(pure_xdot)
            Ax_without_slack[-num_constr:] /= sample_period
            msg.position.extend(Ax_without_slack.tolist())
        return msg

    @profile
    def update(self):
        qp_controller: QPProblemBuilder = self.god_map.get_data(identifier.qp_controller)
        msg = self.create_msg(qp_controller)
        self.publisher.publish(msg)
        return Status.RUNNING
