from __future__ import division

from typing import Optional

from giskardpy.data_types.data_types import PrefixName
from giskardpy.goals.cartesian_goals import CartesianPosition, CartesianOrientation
from giskardpy.goals.goal import Goal
from giskardpy.motion_graph.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
from giskardpy.goals.joint_goals import JointPositionList
from giskardpy.god_map import god_map
import giskardpy.casadi_wrapper as cas


class Open(Goal):
    def __init__(self,
                 tip_link: PrefixName,
                 environment_link: PrefixName,
                 goal_joint_state: Optional[float] = None,
                 max_velocity: float = 100,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
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
        self.handle_T_tip = god_map.world.compute_fk(self.handle_link, self.tip_link)
        if name is None:
            name = f'{self.__class__.__name__}'
        super().__init__(name)

        _, max_position = god_map.world.get_joint_position_limits(self.joint_name)
        if goal_joint_state is None:
            goal_joint_state = max_position
        else:
            goal_joint_state = min(max_position, goal_joint_state)

        if not cas.is_true(start_condition):
            handle_T_tip = god_map.world.compose_fk_expression(self.handle_link, self.tip_link)
            handle_T_tip = god_map.monitor_manager.register_expression_updater(handle_T_tip,
                                                                               start_condition)
        else:
            handle_T_tip = cas.TransMatrix(god_map.world.compute_fk(self.handle_link, self.tip_link))

        # %% position
        r_P_c = god_map.world.compose_fk_expression(self.handle_link, self.tip_link).to_position()
        task = self.create_and_add_task('position')
        task.add_point_goal_constraints(frame_P_goal=handle_T_tip.to_position(),
                                        frame_P_current=r_P_c,
                                        reference_velocity=CartesianPosition.default_reference_velocity,
                                        weight=self.weight)

        # %% orientation
        r_R_c = god_map.world.compose_fk_expression(self.handle_link, self.tip_link).to_rotation()
        c_R_r_eval = god_map.world.compose_fk_evaluated_expression(self.tip_link, self.handle_link).to_rotation()

        task = self.create_and_add_task('orientation')
        task.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=handle_T_tip.to_rotation(),
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=CartesianOrientation.default_reference_velocity,
                                           weight=self.weight)

        self.connect_monitors_to_all_tasks(start_condition=start_condition,
                                           hold_condition=hold_condition,
                                           end_condition=end_condition)

        goal_state = {self.joint_name.short_name: goal_joint_state}
        self.add_constraints_of_goal(JointPositionList(goal_state=goal_state,
                                                       max_velocity=max_velocity,
                                                       weight=WEIGHT_BELOW_CA,
                                                       start_condition=start_condition,
                                                       hold_condition=hold_condition,
                                                       end_condition=end_condition,
                                                       name=f'{self.name}/{self.joint_name.short_name}'))


class Close(Goal):
    def __init__(self,
                 tip_link: PrefixName,
                 environment_link: PrefixName,
                 goal_joint_state: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.tip_link = tip_link
        self.environment_link = environment_link
        if name is None:
            name = f'{self.__class__.__name__}'
        super().__init__(name)
        joint_name = god_map.world.get_movable_parent_joint(self.environment_link)
        min_position, _ = god_map.world.get_joint_position_limits(joint_name)
        if goal_joint_state is None:
            goal_joint_state = min_position
        else:
            goal_joint_state = max(min_position, goal_joint_state)
        self.add_constraints_of_goal(Open(tip_link=tip_link,
                                          environment_link=environment_link,
                                          goal_joint_state=goal_joint_state,
                                          weight=weight,
                                          start_condition=start_condition,
                                          hold_condition=hold_condition,
                                          end_condition=end_condition))
