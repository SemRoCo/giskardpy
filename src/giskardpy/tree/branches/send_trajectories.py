from typing import Dict

from py_trees import Sequence

from giskardpy.data_types import PrefixName, Derivatives
from giskardpy.tree.behaviors.send_trajectory import SendFollowJointTrajectory
from giskardpy.tree.branches.control_loop import ControlLoop
from giskardpy.tree.branches.prepare_control_loop import PrepareBaseTrajControlLoop
from giskardpy.tree.composites.better_parallel import Parallel, ParallelPolicy


class ExecuteTraj(Sequence):
    base_closed_loop: ControlLoop
    prepare_base_control: PrepareBaseTrajControlLoop
    move_robots: Parallel

    def __init__(self, name: str = 'execute traj'):
        super().__init__(name)
        self.move_robots = Parallel(name='move robot', policy=ParallelPolicy.SuccessOnAll(synchronise=True))
        self.add_child(self.move_robots)
        self.prepare_base_control = PrepareBaseTrajControlLoop()
        self.insert_child(self.prepare_base_control, 0)

        self.base_closed_loop = ControlLoop(log_traj=False)
        self.base_closed_loop.add_closed_loop_behaviors()
        self.move_robots.add_child(self.base_closed_loop)

    def add_follow_joint_traj_action_server(self, namespace: str, group_name: str,
                                            fill_velocity_values: bool,
                                            path_tolerance: Dict[Derivatives, float] = None):
        behavior = SendFollowJointTrajectory(namespace=namespace, group_name=group_name,
                                             fill_velocity_values=fill_velocity_values, path_tolerance=path_tolerance)
        self.move_robots.add_child(behavior)

    def add_base_traj_action_server(self, cmd_vel_topic: str, track_only_velocity: bool = False,
                                    joint_name: PrefixName = None):
        self.base_closed_loop.send_controls.add_send_cmd_velocity(cmd_vel_topic=cmd_vel_topic,
                                                                  joint_name=joint_name)
