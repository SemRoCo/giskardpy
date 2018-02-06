from giskardpy.robot import Robot
import operator
from collections import namedtuple


Gripper = namedtuple('Gripper', ['frame', 'opening', 'height'])

class Fetch(Robot):
    def __init__(self, urdf='fetch.urdf'):
        super(Fetch, self).__init__()
        self.load_from_urdf_path(urdf, 'base_link', ['gripper_link', 'head_camera_link'])
        for joint_name in self.joint_constraints:
            self.set_joint_weight(joint_name, 0.01)

        self.set_joint_weight('torso_lift_joint', 0.05)

        self.gripper = Gripper(frame=self.frames['gripper_link'], opening=0.1, height=0.03)
        self.eef     = self.gripper.frame
        self.camera  = self.frames['head_camera_link']

    def set_joint_state(self, joint_state):
        for i, joint_name in enumerate(joint_state.name):
            if joint_name in self._state:
                self._state[joint_name] = joint_state.position[i]

    def __str__(self):
        return 'Fetch\'s state:\n' + reduce(operator.add, map(lambda t: t[0] + ': {:.3f}\n'.format(t[1]), self.joint_state.iteritems()), '')
