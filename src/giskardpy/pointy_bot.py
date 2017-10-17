from giskardpy.robot import Robot
import numpy as np


class PointyBot(Robot):
    def __init__(self, weight, urdf='pointy.urdf'):
        super(PointyBot, self).__init__()
        self.load_from_urdf_path(urdf, 'base_link', ['eef'])
        for joint_name in self.joint_constraints:
            self.set_joint_weight(joint_name, weight)

    def set_joint_state(self, joint_state):
        self.joint_state = joint_state

    def update_observables(self):
        return self.joint_state

    def __str__(self):
        return 'x: {:.3f}, y:{:.3f}, z: {:.3f}'.format(*self.joint_state.values())
