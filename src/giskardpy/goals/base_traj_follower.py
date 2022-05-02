from __future__ import division

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA
from giskardpy.model.joints import OmniDrive
from giskardpy.my_types import my_string, expr_symbol


class BaseTrajFollower(Goal):
    def __init__(self, joint_name: my_string, weight: float = WEIGHT_BELOW_CA, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to achieve a goal position for tip link.
        :param root_link: root link of kinematic chain
        :param tip_link: tip link of kinematic chain
        :param goal: the goal, orientation part will be ignored
        :param max_velocity: m/s
        :param reference_velocity: m/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        super().__init__(**kwargs)
        self.weight = weight
        self.joint_name = joint_name
        self.joint: OmniDrive = self.world.joints[joint_name]
        self.trajectory = self.god_map.get_data(identifier.trajectory)
        self.trajectory_length = len(self.trajectory.items())

    def x_symbol(self, t: int, free_variable_name: str) -> expr_symbol:
        return self.god_map.to_symbol(identifier.trajectory + ['get_exact', (t,), free_variable_name, 'position'])

    def current_traj_point(self, free_variable_name: str) -> expr_symbol:
        time = self.god_map.to_expr(identifier.time)
        return w.if_less_eq_cases(time,
                                  [(t * self.get_sampling_period_symbol() + 0.5, self.x_symbol(t, free_variable_name))
                                   for t in range(self.trajectory_length)],
                                  self.x_symbol(self.trajectory_length-1, free_variable_name))

    def make_constraints(self):
        odom_link = self.joint.parent_link_name
        base_footprint_link = self.joint.child_link_name
        x = self.current_traj_point(self.joint.x_name)
        y = self.current_traj_point(self.joint.y_name)
        rot = self.current_traj_point(self.joint.rot_name)
        odom_T_base_footprint_goal = w.frame_from_x_y_rot(x, y, rot)
        map_T_odom = self.get_fk_evaluated(self.world.root_link_name, odom_link)
        map_T_base_footprint_goal = w.dot(map_T_odom, odom_T_base_footprint_goal)
        map_T_base_footprint_current = self.get_fk(self.world.root_link_name, base_footprint_link)
        # self.add_debug_expr('x goal', map_T_base_footprint_goal[0,3])
        # self.add_debug_expr('y goal', map_T_base_footprint_goal[1,3])
        # self.add_debug_expr('rot', rot)
        # self.add_debug_expr('x current', map_T_base_footprint_current[0,3])
        # self.add_debug_expr('y current', map_T_base_footprint_current[1,3])

        self.add_point_goal_constraints(frame_P_goal=w.position_of(map_T_base_footprint_goal),
                                        frame_P_current=w.position_of(map_T_base_footprint_current),
                                        reference_velocity=1,
                                        weight=WEIGHT_BELOW_CA)

        _, rotation_goal = w.axis_angle_from_matrix(map_T_base_footprint_goal)
        axis_current, rotation_current = w.axis_angle_from_matrix(map_T_base_footprint_current)

        error = w.shortest_angular_distance(rotation_current, rotation_goal)
        self.add_constraint(reference_velocity=0.5,
                            lower_error=error,
                            upper_error=error,
                            weight=self.weight,
                            expression=rotation_current)
        self.add_debug_expr('rot goal', rotation_goal)
        self.add_debug_expr('error', error)
        self.add_debug_expr('rot current', rotation_current)
        self.add_debug_vector('axis_current', axis_current)


    def __str__(self):
        return f'{super().__str__()}/{self.joint_name}'
