import numpy as np
import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

from giskardpy import identifier
from giskardpy.data_types import JointStates
from giskardpy.model.trajectory import Trajectory
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


    def split_traj(self, traj) -> Trajectory:
        new_traj = Trajectory()
        for time, js in traj.items():
            new_js = JointStates()
            for name, js_ in js.items():
                # name = name.replace('/', '|')
                # traj_name = ''.join(name.split('/')[:-1])
                # name = name.split('/')[-1]
                if isinstance(js_.position, np.ndarray):
                    for x in range(js_.position.shape[0]):
                        for y in range(js_.position.shape[1]):
                            tmp_name = f'{name}|{x}_{y}'
                            # tmp_name = re.escape(tmp_name)
                            # tmp_name = tmp_name.replace('/', '|')
                            # tmp_name = tmp_name.replace('/', '\/')
                            new_js[tmp_name].position = js_.position[x, y]
                            new_js[tmp_name].velocity = js_.velocity[x, y]
                else:
                    new_js[name] = js_
                new_traj.set(time, new_js)

        return new_traj

    @profile
    def update(self):
        # print('hi')
        debug_pandas = self.god_map.get_data(identifier.debug_expressions_evaluated)
        qp_controller: QPController = self.god_map.get_data(identifier.qp_controller)
        qp_controller._create_debug_pandas()
        msg = JointState()
        msg.header.stamp = rospy.get_rostime()
        for debug_name, debug_value in debug_pandas.items():
            if isinstance(debug_value, float):
                msg.name.append(debug_name)
                msg.position.append(debug_value)
            elif isinstance(debug_value, np.ndarray):
                for x in range(debug_value.shape[0]):
                    for y in range(debug_value.shape[1]):
                        msg.name.append(f'{debug_name}|{x}_{y}')
                        msg.position.append(debug_value[x, y])
        for name, thing in zip(['lbA', 'ubA', 'lb', 'ub', 'weights', 'xdot', 'Ax no slack'],
                         [qp_controller.p_lbA, qp_controller.p_ubA, qp_controller.p_lb, qp_controller.p_ub,
                          qp_controller.p_weights, qp_controller.p_xdot, qp_controller.p_Ax_without_slack]):
            msg.name.extend([f'{name}/{x}' for x in thing.index])
            msg.position.extend(list(thing.values.T[0]))

        self.publisher.publish(msg)
        return Status.RUNNING
