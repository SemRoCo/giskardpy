from __future__ import division

from giskardpy.goals.cartesian_goals import CartesianPose
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
from giskardpy.goals.joint_goals import JointPosition


class Open(Goal):
    def __init__(self, tip_link, environment_link, goal_joint_state=None, weight=WEIGHT_ABOVE_CA, **kwargs):
        super(Open, self).__init__(**kwargs)
        self.weight = weight
        self.tip_link = tip_link
        self.handle_link = environment_link
        self.joint_name = self.world.get_movable_parent_joint(environment_link)
        self.handle_T_tip = self.world.compute_fk_pose(self.handle_link, self.tip_link)

        _, max_position = self.world.get_joint_position_limits(self.joint_name)
        if goal_joint_state is None:
            goal_joint_state = max_position
        else:
            goal_joint_state = min(max_position, goal_joint_state)

        self.add_constraints_of_goal(CartesianPose(root_link=self.handle_link,
                                                   tip_link=self.tip_link,
                                                   goal=self.handle_T_tip,
                                                   weight=self.weight, **kwargs))
        self.add_constraints_of_goal(JointPosition(joint_name=self.joint_name,
                                                   goal=goal_joint_state,
                                                   weight=weight,
                                                   **kwargs))

    def __str__(self):
        return '{}/{}'.format(super(Open, self).__str__(), self.tip_link, self.handle_link)


class Close(Goal):
    def __init__(self, tip_link, environment_link, weight=WEIGHT_ABOVE_CA, **kwargs):
        super(Close, self).__init__(**kwargs)
        joint_name = self.world.get_movable_parent_joint(environment_link)
        goal_joint_state, _ = self.world.get_joint_position_limits(joint_name)
        self.add_constraints_of_goal(Open(tip_link=tip_link,
                                          environment_link=environment_link,
                                          goal_joint_state=goal_joint_state,
                                          weight=weight,
                                          **kwargs))
