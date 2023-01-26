from __future__ import division

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.model.joints import OmniDrive
from giskardpy.my_types import my_string, Derivatives, PrefixName


class BaseTrajFollower(Goal):
    def __init__(self, joint_name: my_string, track_only_velocity: bool = False, weight: float = WEIGHT_ABOVE_CA):
        super().__init__()
        self.weight = weight
        self.joint_name = joint_name
        self.joint: OmniDrive = self.world.joints[joint_name]
        self.odom_link = self.joint.parent_link_name
        self.base_footprint_link = self.joint.child_link_name
        self.track_only_velocity = track_only_velocity
        # self.control_horizon = 1

    @profile
    def x_symbol(self, t: int, free_variable_name: PrefixName, derivative: Derivatives = Derivatives.position) \
            -> w.Symbol:
        return self.god_map.to_symbol(identifier.trajectory + ['get_exact', (t,), free_variable_name, derivative])

    @profile
    def current_traj_point(self, free_variable_name: PrefixName, start_t: float,
                           derivative: Derivatives = Derivatives.position) \
            -> w.Expression:
        time = self.god_map.to_expr(identifier.time)
        # self.add_debug_expr('time', time)
        b_result_cases = []
        for t in range(self.trajectory_length):
            b = t * self.sample_period
            eq_result = self.x_symbol(t, free_variable_name, derivative)
            b_result_cases.append((b, eq_result))
        return w.if_less_eq_cases(a=time + start_t,
                                  b_result_cases=b_result_cases,
                                  else_result=self.x_symbol(self.trajectory_length - 1, free_variable_name, derivative))

    @profile
    def make_odom_T_base_footprint_goal(self, t: float, derivative: Derivatives = Derivatives.position):
        x = self.current_traj_point(self.joint.x.name, t, derivative)
        if isinstance(self.joint, OmniDrive) or derivative == 0:
            y = self.current_traj_point(self.joint.y.name, t, derivative)
        else:
            y = 0
        rot = self.current_traj_point(self.joint.yaw.name, t, derivative)
        odom_T_base_footprint_goal = w.TransMatrix.from_xyz_rpy(x=x, y=y, yaw=rot)
        # self.add_debug_expr('x goal in odom', x)
        return odom_T_base_footprint_goal

    @profile
    def trans_error_at(self, t: float):
        odom_T_base_footprint_goal = self.make_odom_T_base_footprint_goal(t)
        map_T_odom = self.get_fk_evaluated(self.world.root_link_name, self.odom_link)
        map_T_base_footprint_goal = w.dot(map_T_odom, odom_T_base_footprint_goal)
        map_T_base_footprint_current = self.get_fk(self.world.root_link_name, self.base_footprint_link)
        # self.add_debug_expr('x goal', map_T_base_footprint_goal[0,3])
        # self.add_debug_expr('y goal', map_T_base_footprint_goal[1,3])
        # self.add_debug_expr('rot', rot)
        # self.add_debug_expr('x current', map_T_base_footprint_current[0,3])
        # self.add_debug_expr('y current', map_T_base_footprint_current[1,3])
        # self.add_debug_expr('y norm', w.norm(map_T_base_footprint_current[:,3]))

        frame_P_goal = map_T_base_footprint_goal.to_position()
        frame_P_current = map_T_base_footprint_current.to_position()
        error = (frame_P_goal[:3] - frame_P_current[:3]) / self.sample_period
        # error /= self.get_sampling_period_symbol()
        # self.add_debug_expr('error x', error[0])
        # self.add_debug_expr('error y', error[1])
        # self.add_debug_expr('|error|', w.norm(error))
        return error[0], error[1]
        # self.add_constraint_vector(reference_velocities=[reference_velocity] * 3,
        #                            lower_errors=error[:3],
        #                            upper_errors=error[:3],
        #                            weights=[weight] * 3,
        #                            expressions=frame_P_current[:3],
        #                            name_suffixes=['{}/x'.format(name_suffix),
        #                                           '{}/y'.format(name_suffix),
        #                                           '{}/z'.format(name_suffix)])

    @profile
    def add_trans_constraints(self):
        tube_size = 0.0
        errors_x = []
        errors_y = []
        map_T_base_footprint = self.get_fk(self.world.root_link_name, self.base_footprint_link)
        for t in range(self.prediction_horizon):
            # if t == 0:
            #     errors_x.append(self.trans_error_at(t)[0])
            #     errors_y.append(self.trans_error_at(t)[1])
            # else:
            odom_T_base_footprint_goal = self.make_odom_T_base_footprint_goal(t * self.sample_period)
            base_footprint_T_odom = self.get_fk(self.base_footprint_link, self.odom_link)
            base_footprint_T_base_footprint_goal = w.dot(base_footprint_T_odom, odom_T_base_footprint_goal)
            # errors_x.append(w.position_of(base_footprint_T_base_footprint_goal)[0])
            # errors_y.append(w.position_of(base_footprint_T_base_footprint_goal)[1])
            x = self.current_traj_point(self.joint.x_vel.name, t * self.sample_period, Derivatives.velocity)
            if isinstance(self.joint, OmniDrive):
                y = self.current_traj_point(self.joint.y_vel.name, t * self.sample_period, Derivatives.velocity)
            else:
                y = 0
            base_footprint_P_vel = w.Vector3((x, y, 0))
            map_P_vel = w.dot(map_T_base_footprint, base_footprint_P_vel)
            if t == 0 and not self.track_only_velocity:
                actual_error_x, actual_error_y = self.trans_error_at(0)
                errors_x.append(map_P_vel[0] + actual_error_x)
                errors_y.append(map_P_vel[1] + actual_error_y)
            else:
                errors_x.append(map_P_vel[0])
                errors_y.append(map_P_vel[1])
            # if t == 0:
            #     asdf = w.norm(map_P_vel[:2])
            # self.add_debug_expr('lower tube x', asdf-tube_size)
            # self.add_debug_expr('upper tube x', asdf+tube_size)
            # errors_y.append(self.current_traj_point(self.joint.y_vel_name, t * self.get_sampling_period_symbol(), 1))

        # lower_slack_limits = [0] + [-1e4] * (self.control_horizon - 1)
        # upper_slack_limits = [0] + [1e4] * (self.control_horizon - 1)
        weight_e = WEIGHT_ABOVE_CA
        weight_vel = WEIGHT_ABOVE_CA
        lba_x = [x + (-tube_size if i > 0 else 0) for i, x in enumerate(errors_x)]
        uba_x = [x + (tube_size if i > 0 else 0) for i, x in enumerate(errors_x)]
        lba_y = [y + (-tube_size if i > 0 else 0) for i, y in enumerate(errors_y)]
        uba_y = [y + (tube_size if i > 0 else 0) for i, y in enumerate(errors_y)]

        def horizon_weight_function(weight, t):
            weight = weight + weight * -0.1 * t
            return weight

        self.add_velocity_constraint(lower_velocity_limit=lba_x,
                                     upper_velocity_limit=uba_x,
                                     weight=weight_vel,
                                     # expression=self.joint.x_vel.get_symbol(0),
                                     task_expression=map_T_base_footprint.to_position().x,
                                     velocity_limit=0.5,
                                     # lower_slack_limit=lower_slack_limits,
                                     # upper_slack_limit=upper_slack_limits,
                                     name_suffix='/vel x')
        if isinstance(self.joint, OmniDrive):
            self.add_velocity_constraint(lower_velocity_limit=lba_y,
                                         upper_velocity_limit=uba_y,
                                         weight=weight_vel,
                                         # expression=self.joint.y_vel.get_symbol(0),
                                         task_expression=map_T_base_footprint.to_position().y,
                                         velocity_limit=0.5,
                                         # lower_slack_limit=lower_slack_limits,
                                         # upper_slack_limit=upper_slack_limits,
                                         name_suffix='/vel y')
        # actual_error_x, actual_error_y = self.trans_error_at(0)
        # self.add_constraint(reference_velocity=0.5,
        #                     lower_error=actual_error_x,
        #                     upper_error=actual_error_x,
        #                     weight=weight_e,
        #                     expression=w.position_of(map_T_base_footprint)[0],
        #                     name_suffix='/error x',
        #                     control_horizon=1)
        # self.add_constraint(reference_velocity=0.5,
        #                     lower_error=actual_error_y,
        #                     upper_error=actual_error_y,
        #                     weight=weight_e,
        #                     expression=w.position_of(map_T_base_footprint)[1],
        #                     name_suffix='/error y',
        #                     control_horizon=1)

        # self.add_debug_expr('x error', actual_error_x)
        # self.add_debug_expr('y error', actual_error_y)
        # self.add_debug_expr('ref x traj', self.current_traj_point(self.joint.x_vel_name, 0, 0))
        # self.add_debug_expr('ref x vel traj/0', self.current_traj_point(self.joint.x_vel_name, 0, 1))
        # self.add_debug_expr('ref y traj', self.current_traj_point(self.joint.y_vel_name, 0, 0))
        # self.add_debug_expr('ref y vel traj/0', self.current_traj_point(self.joint.y_vel_name, 0, 1))
        # self.add_debug_expr('error sum', errors_x[0] + errors_y[0])

    @profile
    def rot_error_at(self, t: int):
        # odom_link = self.joint.parent_link_name
        # base_footprint_link = self.joint.child_link_name
        # x = self.current_traj_point(self.joint.x_name, t)
        # y = self.current_traj_point(self.joint.y_name, t)
        # rot = self.current_traj_point(self.joint.rot_name, t)
        # odom_T_base_footprint_goal = w.frame_from_x_y_rot(x, y, rot)
        # map_T_odom = self.get_fk_evaluated(self.world.root_link_name, odom_link)
        # map_T_base_footprint_goal = w.dot(map_T_odom, odom_T_base_footprint_goal)
        # map_T_base_footprint_current = self.get_fk(self.world.root_link_name, base_footprint_link)
        # _, rotation_goal = w.axis_angle_from_matrix(map_T_base_footprint_goal)
        # axis_current, rotation_current = w.axis_angle_from_matrix(map_T_base_footprint_current)

        rotation_goal = self.current_traj_point(self.joint.yaw.name, t)
        rotation_current = self.joint.yaw.get_symbol(0)
        error = w.shortest_angular_distance(rotation_current, rotation_goal) / self.sample_period
        # self.add_debug_expr('rot goal', rotation_goal)
        # self.add_debug_expr('rot current', rotation_current)
        return error

    @profile
    def add_rot_constraints(self):
        errors = []
        for t in range(self.prediction_horizon):
            errors.append(self.current_traj_point(self.joint.yaw.name, t * self.sample_period, Derivatives.velocity))
            if t == 0 and not self.track_only_velocity:
                errors[-1] += self.rot_error_at(t)
        self.add_velocity_constraint(lower_velocity_limit=errors,
                                     upper_velocity_limit=errors,
                                     weight=WEIGHT_BELOW_CA,
                                     task_expression=self.joint.yaw_vel.get_symbol(0),
                                     velocity_limit=0.5,
                                     name_suffix='/rot')
        # self.add_constraint(reference_velocity=0.5,
        #                     lower_error=error,
        #                     upper_error=error,
        #                     weight=self.weight,
        #                     expression=b)
        # self.add_debug_expr('rot error', self.current_traj_point(self.joint.rot_name, 0) - self.joint.rot.get_symbol(0))
        # self.add_debug_expr('rot goal', self.current_traj_point(self.joint.rot_name, 0))
        # self.add_debug_expr('rot current', self.joint.rot.get_symbol(0))
        # self.add_debug_expr('rot vel', self.joint.rot_vel.get_symbol(1))
        # self.add_debug_expr('ref traj', self.current_traj_point(self.joint.rot_name, 0, 0))
        # self.add_debug_expr('ref vel traj/0', self.current_traj_point(self.joint.rot_name, 0, 1))
        # self.add_debug_expr('ref vel traj/1',
        #                     self.current_traj_point(self.joint.rot_name, 1 * self.get_sampling_period_symbol(), 1))
        # self.add_debug_expr('ref vel traj/2', self.current_traj_point(self.joint.rot_name, 2*self.get_sampling_period_symbol(), 1))
        # self.add_debug_expr('ref vel traj/3', self.current_traj_point(self.joint.rot_name, 3*self.get_sampling_period_symbol(), 1))
        # self.add_debug_expr('time', self.god_map.to_expr(identifier.time))
        # self.add_debug_expr('ref acc traj', self.current_traj_point(self.joint.rot_name, 0, 2))
        # self.add_debug_expr('ref jerk traj', self.current_traj_point(self.joint.rot_name, 0, 3))
        # self.add_debug_vector('axis_current', axis_current)

    @profile
    def make_constraints(self):
        trajectory = self.god_map.get_data(identifier.trajectory)
        self.trajectory_length = len(trajectory.items())
        self.add_trans_constraints()
        self.add_rot_constraints()

    def __str__(self):
        return f'{super().__str__()}/{self.joint_name}'
