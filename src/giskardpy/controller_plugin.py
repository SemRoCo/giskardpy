from collections import OrderedDict

import rospy
from control_msgs.msg import FollowJointTrajectoryGoal
import numpy as np

from trajectory_msgs.msg import JointTrajectoryPoint

from giskardpy.plugin import IOPlugin
from giskardpy.symengine_controller import JointController


class JointControllerPlugin(IOPlugin):
    def trajectory_rollout(self, time_limit=10, frequency=100, precision=0.0025):
        # TODO sanity checks
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.jc.robot.get_joint_names()
        simulated_js = self.databus.get_data(self.joint_states_identifier)

        step_size = 1. / frequency
        for k in range(int(time_limit / step_size)):
            p = JointTrajectoryPoint()
            p.time_from_start = rospy.Duration((k + 1) * step_size)
            if k != 0:
                cmd_dict = self.jc.get_cmd(self.databus.get_expr_values())
            for i, j in enumerate(goal.trajectory.joint_names):
                if k > 0 and j in cmd_dict:
                    simulated_js[j] += cmd_dict[j] * step_size
                    p.velocities.append(cmd_dict[j])
                else:
                    p.velocities.append(0)
                    pass
                p.positions.append(simulated_js[j])
            goal.trajectory.points.append(p)
            if k > 0 and np.abs(cmd_dict.values()).max() < precision:
                print('done')
                break

        return goal

    def get_readings(self):
        updates = {self.solution_identifier: self.solution}
        self.solution = None
        return updates

    def update(self):
        self.solution = self.trajectory_rollout()

    def start(self, databus):
        super(JointControllerPlugin, self).start(databus)

        urdf = rospy.get_param('robot_description')
        self.jc = JointController(urdf)
        self.goal_symbol_map = {}
        for joint_name in self.jc.robot.get_joint_names():
            self.goal_symbol_map[joint_name] = self.databus.get_expr('{}/{}/position'.format(self.joint_goal_identifier,
                                                                                   joint_name))
        self.jc.init(self.goal_symbol_map)

    def stop(self):
        pass

    def __init__(self):
        super(JointControllerPlugin, self).__init__()
        self.joint_states_identifier = 'js'
        self.solution_identifier = 'solution'
        self.solution = None
        self.joint_goal_identifier = 'joint_goal'
