from __future__ import division

from typing import Optional

from giskardpy.data_types.data_types import PrefixName
from giskardpy.goals.cartesian_goals import CartesianPose
from giskardpy.goals.goal import Goal
from giskardpy.motion_graph.tasks.joint_tasks import JointPositionList
from giskardpy.motion_graph.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
from giskardpy.god_map import god_map
import giskardpy.casadi_wrapper as cas


class Open(Goal):
    def __init__(self,
                 tip_link: PrefixName,
                 environment_link: PrefixName,
                 goal_joint_state: Optional[float] = None,
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
        self.handle_link = environment_link
        self.joint_name = god_map.world.get_movable_parent_joint(self.handle_link)
        if name is None:
            name = f'{self.__class__.__name__}'
        super().__init__(name=name)

        _, max_position = god_map.world.get_joint_position_limits(self.joint_name)
        if goal_joint_state is None:
            goal_joint_state = max_position
        else:
            goal_joint_state = min(max_position, goal_joint_state)

        goal_state = {self.joint_name.short_name: goal_joint_state}
        hinge_goal = JointPositionList(goal_state=goal_state,
                                       name=f'{self.name}/hinge',
                                       weight=self.weight)
        self.add_task(hinge_goal)

        hold_handle = CartesianPose(root_link=self.handle_link,
                                    tip_link=self.tip_link,
                                    name=f'{self.name}/hold handle',
                                    goal_pose=god_map.world.compute_fk(self.handle_link, self.tip_link),
                                    weight=self.weight)
        self.add_goal(hold_handle)
        self.expression = cas.logic_and(hinge_goal.expression, hold_handle.expression)


class Close(Goal):
    def __init__(self,
                 tip_link: PrefixName,
                 environment_link: PrefixName,
                 goal_joint_state: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.tip_link = tip_link
        self.environment_link = environment_link
        if name is None:
            name = f'{self.__class__.__name__}'
        super().__init__(name=name)
        joint_name = god_map.world.get_movable_parent_joint(self.environment_link)
        min_position, _ = god_map.world.get_joint_position_limits(joint_name)
        if goal_joint_state is None:
            goal_joint_state = min_position
        else:
            goal_joint_state = max(min_position, goal_joint_state)
        self.add_constraints_of_goal(Open(tip_link=tip_link,
                                          environment_link=environment_link,
                                          goal_joint_state=goal_joint_state,
                                          weight=weight))
