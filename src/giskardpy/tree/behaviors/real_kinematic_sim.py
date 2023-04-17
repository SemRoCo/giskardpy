from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class RealKinSimPlugin(GiskardBehavior):
    last_time: float

    # def initialise(self):
    #     self.last_time = None
        # self.start_time = self.god_map.get_data(identifier.tracking_start_time)

    @record_time
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
        self.world.update_state(next_cmds, dt)
        self.last_time = next_time
        self.world.notify_state_change()
        return Status.RUNNING
