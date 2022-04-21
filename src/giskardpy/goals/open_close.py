from __future__ import division

from giskardpy.data_types import PrefixName
from giskardpy.goals.cartesian_goals import CartesianPose
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
from giskardpy.goals.joint_goals import JointPosition


class Open(Goal):
    def __init__(self, tip_link, tip_group, environment_link, environment_group, goal_joint_state=None,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super(Open, self).__init__(**kwargs)
        self.weight = weight
        environment_prefix = self.world.groups[environment_group].get_link_short_name_match(environment_link).prefix
        tip_prefix = self.world.groups[tip_group].get_link_short_name_match(tip_link).prefix
        self.tip_link = PrefixName(tip_link, tip_prefix)
        self.handle_link = PrefixName(environment_link, environment_prefix)
        self.joint_name = self.world.get_movable_parent_joint(environment_link)
        self.joint_group = self.world.get_group_of_joint(self.joint_name)
        self.handle_T_tip = self.world.compute_fk_pose(self.handle_link, self.tip_link)

        _, max_position = self.world.get_joint_position_limits(self.joint_name)
        if goal_joint_state is None:
            goal_joint_state = max_position
        else:
            goal_joint_state = min(max_position, goal_joint_state)

        self.add_constraints_of_goal(CartesianPose(root_link=environment_link,
                                                   root_group=environment_group,
                                                   tip_link=tip_link,
                                                   tip_group=tip_group,
                                                   goal_pose=self.handle_T_tip,
                                                   weight=self.weight, **kwargs))
        self.add_constraints_of_goal(JointPosition(joint_name=self.joint_name.short_name,
                                                   group_name=self.joint_group.name,
                                                   goal=goal_joint_state,
                                                   weight=weight,
                                                   **kwargs))

    def __str__(self):
        return '{}/{}'.format(super(Open, self).__str__(), self.tip_link, self.handle_link)


class Close(Goal):
    def __init__(self, tip_link, tip_group, environment_link, environment_group, weight=WEIGHT_ABOVE_CA, **kwargs):
        super(Close, self).__init__(**kwargs)
        joint_name = self.world.get_movable_parent_joint(environment_link)
        goal_joint_state, _ = self.world.get_joint_position_limits(joint_name)
        self.add_constraints_of_goal(Open(tip_link=tip_link,
                                          tip_group=tip_group,
                                          environment_link=environment_link,
                                          environment_group=environment_group,
                                          goal_joint_state=goal_joint_state,
                                          weight=weight,
                                          **kwargs))
