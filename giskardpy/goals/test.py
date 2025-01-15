from __future__ import division

from typing import Optional

from giskardpy.data_types.data_types import PrefixName
from giskardpy.goals.cartesian_goals import CartesianPose
from giskardpy.goals.goal import Goal
from giskardpy.motion_statechart.monitors.monitors import TrueMonitor, CancelMotion
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPoseAsTask
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
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


class Cutting(Goal):
    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 depth: float,
                 right_shift: float,
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

        schnibble_down_pose = god_map.world.compute_fk(root_link=self.tip_link, tip_link=self.tip_link)
        schnibble_down_pose.x = -depth
        cut_down = CartesianPoseAsTask(root_link=self.root_link,
                                       name=f'{self.name}/Down',
                                       goal_pose=schnibble_down_pose,
                                       tip_link=self.tip_link,
                                       absolute=False)
        self.add_task(cut_down)

        made_contact = TrueMonitor(name=f'{self.name}/Made Contact?')
        self.add_monitor(made_contact)
        made_contact.start_condition = cut_down.get_observation_state_expression()
        made_contact.end_condition = made_contact.get_observation_state_expression()

        cancel = CancelMotion(name=f'{self.name}/CancelMotion', exception=Exception('no contact'))
        self.add_monitor(cancel)
        cancel.start_condition = cas.logic_not(made_contact.get_observation_state_expression())


        schnibble_up_pose = god_map.world.compute_fk(root_link=self.tip_link, tip_link=self.tip_link)
        schnibble_up_pose.x = depth
        cut_up = CartesianPoseAsTask(root_link=self.root_link,
                                     name=f'{self.name}/Up',
                                     goal_pose=schnibble_up_pose,
                                     tip_link=self.tip_link,
                                     absolute=False)
        self.add_task(cut_up)

        schnibble_right_pose = god_map.world.compute_fk(root_link=self.tip_link, tip_link=self.tip_link)
        schnibble_right_pose.y = right_shift
        move_right = CartesianPoseAsTask(root_link=self.root_link,
                                         name=f'{self.name}/Move Right',
                                         goal_pose=schnibble_right_pose,
                                         tip_link=self.tip_link,
                                         absolute=False)
        self.add_task(move_right)

        self.arrange_in_sequence([cut_down, cut_up, move_right])
        self.expression = cas.if_else(cas.is_true3(move_right.get_observation_state_expression()),
                                      cas.TrinaryTrue,
                                      cas.TrinaryFalse)
