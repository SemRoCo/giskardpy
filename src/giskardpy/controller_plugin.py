from collections import OrderedDict
import pylab as plt
import rospy
from control_msgs.msg import FollowJointTrajectoryGoal
import numpy as np

from trajectory_msgs.msg import JointTrajectoryPoint

from giskardpy.plugin import IOPlugin
from giskardpy.symengine_controller import JointController


class ControllerPlugin(IOPlugin):
    def __init__(self):
        self._joint_states_identifier = 'js'
        self._solution_identifier = 'solution'
        self._solution = None
        self._goal_identifier = 'goal'
        self._controller = None
        super(ControllerPlugin, self).__init__()

    def get_readings(self):
        updates = {self._solution_identifier: self._solution,
                   self._goal_identifier: None}
        self._solution = None
        return updates

    def update(self):
        if self.databus.get_data(self._goal_identifier) is not None:
            self._solution = self.trajectory_rollout()

    def start(self, databus):
        super(ControllerPlugin, self).start(databus)

    def stop(self):
        pass

    def trajectory_rollout(self, time_limit=10, frequency=100, precision=0.0025):
        # TODO sanity checks
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self._controller.robot.get_joint_names()
        simulated_js = self.databus.get_data(self._joint_states_identifier)

        step_size = 1. / frequency
        for k in range(int(time_limit / step_size)):
            p = JointTrajectoryPoint()
            p.time_from_start = rospy.Duration((k + 1) * step_size)
            if k != 0:
                cmd_dict = self._controller.get_cmd(self.databus.get_expr_values())
            for i, j in enumerate(goal.trajectory.joint_names):
                j2 = str(self._controller.robot.get_joint_to_symbols()[j])
                if k > 0 and j2 in cmd_dict:
                    simulated_js[j].position += cmd_dict[j2] * step_size
                    p.velocities.append(cmd_dict[j2])
                else:
                    p.velocities.append(0)
                    pass
                p.positions.append(simulated_js[j].position)
            goal.trajectory.points.append(p)
            if k > 0 and np.abs(cmd_dict.values()).max() < precision:
                print('done')
                break
        # self.plot_trajectory(goal.trajectory)
        return goal

    def plot_trajectory(self, tj):
        positions = []
        velocities = []
        for point in tj.points:
            positions.append(point.positions)
            velocities.append(point.velocities)
        positions = np.array(positions)
        velocities = np.array(velocities)
        plt.plot(positions - positions.mean(axis=0))
        plt.show()
        plt.plot(velocities)
        plt.show()
        pass


class JointControllerPlugin(ControllerPlugin):
    def start(self, databus):
        super(JointControllerPlugin, self).start(databus)
        urdf = rospy.get_param('robot_description')
        self._controller = JointController(urdf)
        self.goal_symbol_map = {}
        self.joint_symbol_map = {}
        for joint_name in self._controller.robot.get_joint_names():
            self.goal_symbol_map[joint_name] = self.databus.get_expr('{}/{}/position'.format(self._goal_identifier,
                                                                                             joint_name))
            self.joint_symbol_map[joint_name] = self.databus.get_expr(
                '{}/{}/position'.format(self._joint_states_identifier,
                                        joint_name))

        self._controller.init(self.joint_symbol_map, self.goal_symbol_map)
