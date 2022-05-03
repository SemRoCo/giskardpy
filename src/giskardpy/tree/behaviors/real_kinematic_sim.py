import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class RealKinSimPlugin(GiskardBehavior):
    last_time: rospy.Time()

    def __init__(self, name):
        self.last_time = None
        super().__init__(name)

    @profile
    def update(self):
        if self.last_time is None:
            self.last_time = rospy.get_rostime()
        next_cmds = self.god_map.get_data(identifier.qp_solver_solution)
        joints = self.world.joints
        next_time = rospy.get_rostime()
        dt = (next_time - self.last_time).to_sec()
        # print(f'dt: {dt}')
        for joint_name in self.world.controlled_joints:
            joints[joint_name].update_state(next_cmds, dt)
        self.last_time = next_time
        self.world.notify_state_change()
        return Status.RUNNING
