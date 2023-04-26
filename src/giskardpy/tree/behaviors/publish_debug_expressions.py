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
from giskardpy.utils.decorators import record_time


class PublishDebugExpressions(GiskardBehavior):
    @profile
    def __init__(self, name, publish_lb: bool = False, publish_ub: bool = False, publish_xdot: bool = False,
                 publish_lbA: bool = False, publish_ubA: bool = False, publish_Ax: bool = False,
                 publish_Ex: bool = False, publish_bE: bool = False,
                 publish_weights: bool = False, publish_g: bool = False, publish_debug: bool = False, **kwargs):
        super().__init__(name)
        self.publish_lb = publish_lb
        self.publish_ub = publish_ub
        self.publish_lbA = publish_lbA
        self.publish_ubA = publish_ubA
        self.publish_bE = publish_bE
        self.publish_weights = publish_weights
        self.publish_g = publish_g
        self.publish_Ax = publish_Ax
        self.publish_Ex = publish_Ex
        self.publish_xdot = publish_xdot
        self.publish_debug = publish_debug

    @profile
    def setup(self, timeout):
        self.publisher = rospy.Publisher('~qp_data', JointState, queue_size=1)
        return super().setup(timeout)

    @profile
    def create_msg(self, qp_controller: QPProblemBuilder):
        msg = JointState()
        msg.header.stamp = rospy.get_rostime()

        weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter = qp_controller.qp_solver.get_problem_data()
        free_variable_names = qp_controller.free_variable_bounds.names[weight_filter]
        equality_constr_names = qp_controller.equality_bounds.names[bE_filter]
        inequality_constr_names = qp_controller.inequality_bounds.names[bA_filter]

        if self.publish_debug:
            for name, value in qp_controller.evaluated_debug_expressions.items():
                if isinstance(value, np.ndarray):
                    if len(value) > 1:
                        if len(value.shape) == 2:
                            for x in range(value.shape[0]):
                                for y in range(value.shape[1]):
                                    tmp_name = f'{name}|{x}_{y}'
                                    msg.name.append(tmp_name)
                                    msg.position.append(value[x, y])
                        else:
                            for x in range(value.shape[0]):
                                    tmp_name = f'{name}|{x}'
                                    msg.name.append(tmp_name)
                                    msg.position.append(value[x])
                    else:
                        msg.name.append(name)
                        msg.position.append(value.flatten())
                else:
                    msg.name.append(name)
                    msg.position.append(value)

        if self.publish_lb:
            names = [f'lb/{entry_name}' for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(lb.tolist())

        if self.publish_ub:
            names = [f'ub/{entry_name}' for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(ub.tolist())

        if self.publish_lbA:
            names = [f'lbA/{entry_name}' for entry_name in inequality_constr_names]
            msg.name.extend(names)
            msg.position.extend(lbA.tolist())

        if self.publish_ubA:
            names = [f'ubA/{entry_name}' for entry_name in inequality_constr_names]
            msg.name.extend(names)
            msg.position.extend(ubA.tolist())

        if self.publish_bE:
            names = [f'bE/{entry_name}' for entry_name in equality_constr_names]
            msg.name.extend(names)
            msg.position.extend(bE.tolist())

        if self.publish_weights:
            names = [f'weights/{entry_name}' for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(weights.tolist())

        if self.publish_g:
            names = [f'g/{entry_name}' for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(g.tolist())

        if self.publish_xdot:
            names = [f'xdot/{entry_name}' for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(qp_controller.xdot_full.tolist())

        if self.publish_Ax or self.publish_Ex:
            sample_period = self.god_map.get_data(identifier.sample_period)
            num_vel_constr = len(qp_controller.derivative_constraints) * (qp_controller.prediction_horizon - 2)
            num_neq_constr = len(qp_controller.inequality_constraints)
            num_eq_constr = len(qp_controller.equality_constraints)
            num_constr = num_vel_constr + num_neq_constr + num_eq_constr

            pure_xdot = qp_controller.xdot_full.copy()
            pure_xdot[-num_constr:] = 0

            if self.publish_Ax:
                names = [f'Ax/{entry_name}' for entry_name in inequality_constr_names]
                msg.name.extend(names)
                Ax = np.dot(A, pure_xdot)
                # Ax[-num_constr:] /= sample_period
                msg.position.extend(Ax.tolist())
            if self.publish_Ex:
                names = [f'Ex/{entry_name}' for entry_name in equality_constr_names]
                msg.name.extend(names)
                Ex = np.dot(E, pure_xdot)
                # Ex[-num_constr:] /= sample_period
                msg.position.extend(Ex.tolist())

        return msg

    @record_time
    @profile
    def update(self):
        qp_controller: QPProblemBuilder = self.god_map.get_data(identifier.qp_controller)
        msg = self.create_msg(qp_controller)
        self.publisher.publish(msg)
        return Status.RUNNING
