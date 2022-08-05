import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class RealKinSimPlugin(GiskardBehavior):
    last_time: float

    # def initialise(self):
    #     self.last_time = None
        # self.start_time = self.god_map.get_data(identifier.tracking_start_time)

    @profile
    def update(self):
        next_time = self.god_map.get_data(identifier.time)
        if next_time <= 0.0 or not hasattr(self, 'last_time'):
            self.last_time = next_time
            return Status.RUNNING
        # if self.last_time is None:
        next_cmds = self.god_map.get_data(identifier.qp_solver_solution)
        joints = self.world.joints
        # next_time = rospy.get_rostime()
        dt = next_time - self.last_time
        # print(f'dt: {dt}')
        for joint_name in self.world.controlled_joints:
            joints[joint_name].update_state(next_cmds, dt)
        self.last_time = next_time
        self.world.notify_state_change()
        return Status.RUNNING
