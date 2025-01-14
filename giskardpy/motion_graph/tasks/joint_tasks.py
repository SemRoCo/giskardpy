from typing import Optional, Dict, List, Tuple

from giskardpy import casadi_wrapper as cas
from giskardpy.data_types.data_types import Derivatives, PrefixName
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.god_map import god_map
from giskardpy.model.joints import OneDofJoint, JustinTorso
from giskardpy.motion_graph.monitors.joint_monitors import JointGoalReached
from giskardpy.motion_graph.tasks.task import Task, WEIGHT_BELOW_CA
from giskardpy.qp.pos_in_vel_limits import b_profile
from giskardpy.utils.math import find_best_jerk_limit


class JointPositionList(Task):
    def __init__(self, *,
                 goal_state: Dict[str, float],
                 group_name: Optional[str] = None,
                 threshold: float = 0.01,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 name: Optional[str] = None,
                 plot: bool = True):
        super().__init__(name=name, plot=plot)

        self.current_positions = []
        self.goal_positions = []
        self.velocity_limits = []
        self.joint_names = []
        self.max_velocity = max_velocity
        self.weight = weight
        if len(goal_state) == 0:
            raise GoalInitalizationException(f'Can\'t initialize {self} with no joints.')

        for joint_name, goal_position in goal_state.items():
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            self.joint_names.append(joint_name)

            ll_pos, ul_pos = god_map.world.compute_joint_limits(joint_name, Derivatives.position)
            # if ll_pos is not None:
            #     goal_position = cas.limit(goal_position, ll_pos, ul_pos)

            ll_vel, ul_vel = god_map.world.compute_joint_limits(joint_name, Derivatives.velocity)
            velocity_limit = cas.limit(max_velocity, ll_vel, ul_vel)

            joint: OneDofJoint = god_map.world.joints[joint_name]
            self.current_positions.append(joint.free_variable.get_symbol(Derivatives.position))
            self.goal_positions.append(goal_position)
            self.velocity_limits.append(velocity_limit)

        for name, current, goal, velocity_limit in zip(self.joint_names, self.current_positions,
                                                       self.goal_positions, self.velocity_limits):
            if god_map.world.is_joint_continuous(name):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current

            self.add_equality_constraint(name=f'{self.name}/{name}',
                                         reference_velocity=velocity_limit,
                                         equality_bound=error,
                                         weight=self.weight,
                                         task_expression=current)
            ll_pos, ul_pos = god_map.world.compute_joint_limits(name, Derivatives.position)
            # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/target', goal,
            #                                                       derivatives_to_plot=[
            #                                                           Derivatives.position,
            #                                                           # Derivatives.velocity
            #                                                       ])
            # cap = self.max_velocity*god_map.qp_controller.mpc_dt * (god_map.qp_controller.prediction_horizon-2)
            # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/upper_cap', goal + cap,
            #                                                       derivatives_to_plot=[
            #                                                           Derivatives.position,
            #                                                           # Derivatives.velocity
            #                                                       ])
            # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/lower_cap', goal - cap,
            #                                                       derivatives_to_plot=[
            #                                                           Derivatives.position,
            #                                                           # Derivatives.velocity
            #                                                       ])
            # god_map.debug_expression_manager.add_debug_expression(f'{name}/lower_limit', cas.Expression(ll_pos),
            #                                                       derivatives_to_plot=[
            #                                                           Derivatives.position,
            #                                                           # Derivatives.velocity
            #                                                       ])
            # if god_map.qp_controller.qp_formulation.is_mpc() and ul_pos is not None:
            #     god_map.debug_expression_manager.add_debug_expression(f'{name}/joint_bounds', cas.Expression(ul_pos),
            #                                                           derivatives_to_plot=[
            #                                                               Derivatives.position,
            #                                                               # Derivatives.velocity
            #                                                           ])
            #     current_vel = god_map.world.joints[name].free_variable.get_symbol(Derivatives.velocity)
            #     current_acc = god_map.world.joints[name].free_variable.get_symbol(Derivatives.acceleration)
            #     jerk_limit = find_best_jerk_limit(god_map.qp_controller.prediction_horizon, god_map.qp_controller.mpc_dt, god_map.world.compute_joint_limits(name, Derivatives.velocity)[1])
            #     lb, ub = b_profile(current_pos=current,
            #                        current_vel=current_vel,
            #                        current_acc=current_acc,
            #                        pos_limits=(ll_pos, ul_pos),
            #                        vel_limits=god_map.world.compute_joint_limits(name, Derivatives.velocity),
            #                        acc_limits=god_map.world.compute_joint_limits(name, Derivatives.acceleration),
            #                        jerk_limits=(-jerk_limit, jerk_limit),
            #                        dt=god_map.qp_controller.mpc_dt,
            #                        ph=god_map.qp_controller.prediction_horizon)
            #     god_map.debug_expression_manager.add_debug_expression(f'{name}/upper_vel',
            #                                                           ub[0],
            #                                                           derivative=Derivatives.velocity,
            #                                                           color='r--',
            #                                                           derivatives_to_plot=[Derivatives.velocity])
            #     god_map.debug_expression_manager.add_debug_expression(f'{name}/lower_vel',
            #                                                           lb[0],
            #                                                           derivative=Derivatives.velocity,
            #                                                           color='r--',
            #                                                           derivatives_to_plot=[Derivatives.velocity])
            #     god_map.debug_expression_manager.add_debug_expression(f'{name}/upper_jerk',
            #                                                           ub[god_map.qp_controller.prediction_horizon*2],
            #                                                           derivative=Derivatives.jerk,
            #                                                           color='r--',
            #                                                           derivatives_to_plot=[Derivatives.jerk])
            #     god_map.debug_expression_manager.add_debug_expression(f'{name}/lower_jerk',
            #                                                           lb[god_map.qp_controller.prediction_horizon*2],
            #                                                           derivative=Derivatives.jerk,
            #                                                           color='r--',
            #                                                           derivatives_to_plot=[Derivatives.jerk])
            # god_map.debug_expression_manager.add_debug_expression(f'{name}/weight',
            #                                                       weight,
            #                                                       derivative=Derivatives.position,
            #                                                       color='r--',
            #                                                       derivatives_to_plot=[Derivatives.position])
            # for d in Derivatives.range(Derivatives.position, Derivatives.jerk):
            #     if d == Derivatives.position:
            #         variable_name = f'{name}/current'
            #     else:
            #         variable_name = f'{name}/current/{d}'
            #     god_map.debug_expression_manager.add_debug_expression(variable_name,
            #                                                           god_map.world.joints[name].get_symbol(d),
            #                                                           derivative=d,
            #                                                           color='r--',
            #                                                           derivatives_to_plot=[d])
        joint_monitor = JointGoalReached(goal_state=goal_state,
                                         threshold=threshold)
        self.expression = joint_monitor.expression


class JointPositionLimitList(Task):
    def __init__(self,
                 lower_upper_limits: Dict[str, Tuple[float, float]],
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.BinaryTrue,
                 pause_condition: cas.Expression = cas.BinaryFalse,
                 end_condition: cas.Expression = cas.BinaryFalse):
        """
        Calls JointPosition for a list of joints.
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints, you should group joint types, e.g., prismatic joints
        :param hard: turns this into a hard constraint.
        """
        self.current_positions = []
        self.lower_limits = []
        self.upper_limits = []
        self.velocity_limits = []
        self.names = []
        self.joint_names = list(sorted(lower_upper_limits.keys()))
        if name is None:
            name = f'{self.__class__.__name__} {self.joint_names}'
        super().__init__(name=name)
        self.max_velocity = max_velocity
        self.weight = weight
        if len(lower_upper_limits) == 0:
            raise GoalInitalizationException(f'Can\'t initialize {self} with no joints.')
        for joint_name, (lower_limit, upper_limit) in lower_upper_limits.items():
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)

            ll_pos, ul_pos = god_map.world.compute_joint_limits(joint_name, Derivatives.position)
            if ll_pos is not None:
                lower_limit = min(ul_pos, max(ll_pos, lower_limit))
                upper_limit = min(ul_pos, max(ll_pos, upper_limit))

            ll_vel, ul_vel = god_map.world.compute_joint_limits(joint_name, Derivatives.velocity)
            velocity_limit = min(ul_vel, max(ll_vel, max_velocity))

            joint: OneDofJoint = god_map.world.joints[joint_name]
            self.names.append(str(joint_name))
            self.current_positions.append(joint.get_symbol(Derivatives.position))
            self.lower_limits.append(lower_limit)
            self.upper_limits.append(upper_limit)
            self.velocity_limits.append(velocity_limit)

        for name, current, lower_limit, upper_limit, velocity_limit in zip(self.names, self.current_positions,
                                                       self.lower_limits, self.upper_limits, self.velocity_limits):
            if god_map.world.is_joint_continuous(name):
                lower_error = cas.shortest_angular_distance(current, lower_limit)
                upper_error = cas.shortest_angular_distance(current, upper_limit)
            else:
                lower_error = lower_limit - current
                upper_error = upper_limit - current

            self.add_inequality_constraint(name=name,
                                           reference_velocity=velocity_limit,
                                           lower_error=lower_error,
                                           upper_error=upper_error,
                                           weight=self.weight,
                                           task_expression=current)

class JustinTorsoLimit(Task):
    def __init__(self,
                 joint_name: PrefixName,
                 lower_limit: Optional[float] = None,
                 upper_limit: Optional[float] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        joint: JustinTorso = god_map.world.joints[joint_name]
        self.max_velocity = max_velocity
        self.weight = weight

        current = joint.q3

        if god_map.world.is_joint_continuous(joint_name):
            lower_error = cas.shortest_angular_distance(current, lower_limit)
            upper_error = cas.shortest_angular_distance(current, upper_limit)
        else:
            lower_error = lower_limit - current
            upper_error = upper_limit - current

        god_map.debug_expression_manager.add_debug_expression('torso 4 joint', current)
        god_map.debug_expression_manager.add_debug_expression('torso 2 joint', joint.q1.get_symbol(Derivatives.position))
        god_map.debug_expression_manager.add_debug_expression('torso 3 joint', joint.q2.get_symbol(Derivatives.position))
        god_map.debug_expression_manager.add_debug_expression('lower_limit', lower_limit)
        god_map.debug_expression_manager.add_debug_expression('upper_limit', upper_limit)

        self.add_inequality_constraint(name=name,
                                       reference_velocity=1,
                                       lower_error=lower_error,
                                       upper_error=upper_error,
                                       weight=self.weight,
                                       task_expression=current)



class JointVelocityLimit(Task):
    def __init__(self,
                 joint_names: List[str],
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 hard: bool = False,
                 name: Optional[str] = None):
        """
        Limits the joint velocity of a revolute joint.
        :param joint_name:
        :param group_name: if joint_name is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        self.weight = weight
        self.max_velocity = max_velocity
        self.hard = hard
        self.joint_names = joint_names
        if name is None:
            name = f'{self.__class__.__name__}/{self.joint_names}'
        super().__init__(name=name)

        for joint_name in self.joint_names:
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            joint: OneDofJoint = god_map.world.joints[joint_name]
            current_joint = joint.get_symbol(Derivatives.position)
            try:
                limit_expr = joint.get_limit_expressions(Derivatives.velocity)[1]
                max_velocity = cas.min(self.max_velocity, limit_expr)
            except IndexError:
                max_velocity = self.max_velocity
            if self.hard:
                self.add_velocity_constraint(lower_velocity_limit=-max_velocity,
                                             upper_velocity_limit=max_velocity,
                                             weight=self.weight,
                                             task_expression=current_joint,
                                             velocity_limit=max_velocity,
                                             lower_slack_limit=0,
                                             upper_slack_limit=0)
            else:
                self.add_velocity_constraint(lower_velocity_limit=-max_velocity,
                                             upper_velocity_limit=max_velocity,
                                             weight=self.weight,
                                             task_expression=current_joint,
                                             velocity_limit=max_velocity,
                                             name=joint_name)


class JointVelocity(Task):
    def __init__(self,
                 joint_names: List[str],
                 vel_goal: float,
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 hard: bool = False,
                 name: Optional[str] = None):
        """
        Limits the joint velocity of a revolute joint.
        :param joint_name:
        :param group_name: if joint_name is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        self.weight = weight
        self.vel_goal = vel_goal
        self.max_velocity = max_velocity
        self.hard = hard
        self.joint_names = joint_names
        if name is None:
            name = f'{self.__class__.__name__}/{self.joint_names}'
        super().__init__(name=name)

        for joint_name in self.joint_names:
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            joint: OneDofJoint = god_map.world.joints[joint_name]
            current_joint = joint.get_symbol(Derivatives.position)
            try:
                limit_expr = joint.get_limit_expressions(Derivatives.velocity)[1]
                max_velocity = cas.min(self.max_velocity, limit_expr)
            except IndexError:
                max_velocity = self.max_velocity
            self.add_velocity_eq_constraint(velocity_goal=self.vel_goal,
                                            weight=self.weight,
                                            task_expression=current_joint,
                                            velocity_limit=max_velocity,
                                            name=joint_name)
