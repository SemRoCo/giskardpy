from __future__ import division

from typing import Optional

from giskardpy.data_types.data_types import PrefixName
from giskardpy.goals.cartesian_goals import CartesianPose
from giskardpy.goals.goal import Goal
from giskardpy.motion_graph.tasks.cartesian_tasks import CartesianPoseAsTask
from giskardpy.motion_graph.tasks.joint_tasks import JointPositionList
from giskardpy.motion_graph.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
from giskardpy.god_map import god_map
import giskardpy.casadi_wrapper as cas


class GraspSequence(Goal):
    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 gripper_joint: str,
                 goal_pose: cas.TransMatrix,
                 max_velocity: float = 100,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None):
        """
        Open a container in an environment.
        Only works with the environment was added as urdf.
        Assumes that a handle has already been grasped.
        Can only handle containers with 1 dof, e.g. drawers or doors.
        :param tip_link: end effector that is grasping the handle
        :param environment_link: name of the handle that was grasped
        :param goal_joint_state: goal state for the container. default is maximum joint state.
        :param weight:
        """
        self.weight = weight
        self.tip_link = tip_link
        self.root_link = root_link
        super().__init__(name=name)

        open_state = {gripper_joint: 1.23}
        close_state = {gripper_joint: 0}
        gripper_open = JointPositionList(goal_state=open_state,
                                         name=f'{self.name}/open',
                                         weight=self.weight)
        self.add_task(gripper_open)
        gripper_closed = JointPositionList(goal_state=close_state,
                                           name=f'{self.name}/close',
                                           weight=self.weight)
        self.add_task(gripper_closed)

        grasp = CartesianPoseAsTask(root_link=self.root_link,
                                    tip_link=self.tip_link,
                                    name=f'{self.name}/grasp',
                                    goal_pose=goal_pose,
                                    weight=self.weight)
        self.add_task(grasp)

        lift_pose = god_map.world.transform(god_map.world.root_link_name, goal_pose)
        lift_pose.z += 0.1

        lift = CartesianPoseAsTask(root_link=self.root_link,
                                   tip_link=self.tip_link,
                                   name=f'{self.name}/lift',
                                   goal_pose=lift_pose,
                                   weight=self.weight)
        self.add_task(lift)
        self.arrange_in_sequence([gripper_open, grasp, gripper_closed, lift])
        self.expression = lift.get_observation_state_expression()
