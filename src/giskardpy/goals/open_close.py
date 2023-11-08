from __future__ import division

from typing import Optional, List

from giskardpy.goals.cartesian_goals import CartesianPose
from giskardpy.goals.goal import Goal
from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.joint_goals import JointPositionList
from giskardpy.god_map import god_map


class Open(Goal):
    def __init__(self,
                 tip_link: str,
                 environment_link: str,
                 tip_group: Optional[str] = None,
                 environment_group: Optional[str] = None,
                 goal_joint_state: Optional[float] = None,
                 max_velocity: float = 100,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 to_start: Optional[List[Monitor]] = None,
                 to_hold: Optional[List[Monitor]] = None,
                 to_end: Optional[List[Monitor]] = None
                 ):
        """
        Open a container in an environment.
        Only works with the environment was added as urdf.
        Assumes that a handle has already been grasped.
        Can only handle containers with 1 dof, e.g. drawers or doors.
        :param tip_link: end effector that is grasping the handle
        :param environment_link: name of the handle that was grasped
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param environment_group: if environment_link is not unique, search in this group for matches
        :param goal_joint_state: goal state for the container. default is maximum joint state.
        :param weight:
        """
        self.weight = weight
        self.tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        self.handle_link = god_map.world.search_for_link_name(environment_link, environment_group)
        self.joint_name = god_map.world.get_movable_parent_joint(self.handle_link)
        self.joint_group = god_map.world.get_group_of_joint(self.joint_name)
        self.handle_T_tip = god_map.world.compute_fk_pose(self.handle_link, self.tip_link)
        if name is None:
            name = f'{self.__class__.__name__}/{self.tip_link}/{self.handle_link}'
        super().__init__(name)

        _, max_position = god_map.world.get_joint_position_limits(self.joint_name)
        if goal_joint_state is None:
            goal_joint_state = max_position
        else:
            goal_joint_state = min(max_position, goal_joint_state)

        self.add_constraints_of_goal(CartesianPose(root_link=environment_link,
                                                   root_group=environment_group,
                                                   tip_link=tip_link,
                                                   tip_group=tip_group,
                                                   goal_pose=self.handle_T_tip,
                                                   weight=self.weight,
                                                   to_start=to_start,
                                                   to_hold=to_hold,
                                                   to_end=to_end))
        goal_state = {self.joint_name.short_name: goal_joint_state}
        self.add_constraints_of_goal(JointPositionList(goal_state=goal_state,
                                                       group_name=self.joint_group.name,
                                                       max_velocity=max_velocity,
                                                       weight=WEIGHT_BELOW_CA,
                                                       to_start=to_start,
                                                       to_hold=to_hold,
                                                       to_end=to_end))


class Close(Goal):
    def __init__(self,
                 tip_link: str,
                 environment_link: str,
                 tip_group: Optional[str] = None,
                 environment_group: Optional[str] = None,
                 goal_joint_state: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 to_start: Optional[List[Monitor]] = None,
                 to_hold: Optional[List[Monitor]] = None,
                 to_end: Optional[List[Monitor]] = None
                 ):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.tip_link = tip_link
        self.environment_link = environment_link
        if name is None:
            name = f'{self.__class__.__name__}/{self.tip_link}/{self.environment_link}'
        super().__init__(name)
        handle_link = god_map.world.search_for_link_name(environment_link, environment_group)
        joint_name = god_map.world.get_movable_parent_joint(handle_link)
        min_position, _ = god_map.world.get_joint_position_limits(joint_name)
        if goal_joint_state is None:
            goal_joint_state = min_position
        else:
            goal_joint_state = max(min_position, goal_joint_state)
        self.add_constraints_of_goal(Open(tip_link=tip_link,
                                          tip_group=tip_group,
                                          environment_link=environment_link,
                                          environment_group=environment_group,
                                          goal_joint_state=goal_joint_state,
                                          weight=weight,
                                          to_start=to_start,
                                          to_hold=to_hold,
                                          to_end=to_end))
