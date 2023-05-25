from __future__ import division

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.model.joints import OmniDrive, OmniDrivePR22
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

    @profile
    def x_symbol(self, t: int, free_variable_name: PrefixName, derivative: Derivatives = Derivatives.position) \
            -> w.Symbol:
        return self.god_map.to_symbol(identifier.trajectory + ['get_exact', (t,), free_variable_name, derivative])

    @profile
    def current_traj_point(self, free_variable_name: PrefixName, start_t: float,
                           derivative: Derivatives = Derivatives.position) \
            -> w.Expression:
        time = self.god_map.to_expr(identifier.time)
        b_result_cases = []
        for t in range(self.trajectory_length):
            b = t * self.sample_period
            eq_result = self.x_symbol(t, free_variable_name, derivative)
            b_result_cases.append((b, eq_result))
            #FIXME if less eq cases behavior changed
        return w.if_less_eq_cases(a=time + start_t,
                                  b_result_cases=b_result_cases,
                                  else_result=self.x_symbol(self.trajectory_length - 1, free_variable_name, derivative))

    @profile
    def make_odom_T_base_footprint_goal(self, t_in_s: float, derivative: Derivatives = Derivatives.position):
        x = self.current_traj_point(self.joint.x.name, t_in_s, derivative)
        if isinstance(self.joint, OmniDrive) or derivative == 0:
            y = self.current_traj_point(self.joint.y.name, t_in_s, derivative)
        else:
            y = 0
        rot = self.current_traj_point(self.joint.yaw.name, t_in_s, derivative)
        odom_T_base_footprint_goal = w.TransMatrix.from_xyz_rpy(x=x, y=y, yaw=rot)
        return odom_T_base_footprint_goal

    @profile
    def make_map_T_base_footprint_goal(self, t_in_s: float, derivative: Derivatives = Derivatives.position):
        odom_T_base_footprint_goal = self.make_odom_T_base_footprint_goal(t_in_s, derivative)
        map_T_odom = self.get_fk_evaluated(self.world.root_link_name, self.odom_link)
        return w.dot(map_T_odom, odom_T_base_footprint_goal)

    @profile
    def trans_error_at(self, t_in_s: float):
        odom_T_base_footprint_goal = self.make_odom_T_base_footprint_goal(t_in_s)
        map_T_odom = self.get_fk_evaluated(self.world.root_link_name, self.odom_link)
        map_T_base_footprint_goal = w.dot(map_T_odom, odom_T_base_footprint_goal)
        map_T_base_footprint_current = self.get_fk(self.world.root_link_name, self.base_footprint_link)

        frame_P_goal = map_T_base_footprint_goal.to_position()
        frame_P_current = map_T_base_footprint_current.to_position()
        error = (frame_P_goal - frame_P_current) / self.sample_period
        return error[0], error[1]

    @profile
    def add_trans_constraints(self):
        errors_x = []
        errors_y = []
        map_T_base_footprint = self.get_fk(self.world.root_link_name, self.base_footprint_link)
        for t in range(self.prediction_horizon):
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
        weight_vel = WEIGHT_ABOVE_CA
        lba_x = errors_x
        uba_x = errors_x
        lba_y = errors_y
        uba_y = errors_y

        self.add_velocity_constraint(lower_velocity_limit=lba_x,
                                     upper_velocity_limit=uba_x,
                                     weight=weight_vel,
                                     task_expression=map_T_base_footprint.to_position().x,
                                     velocity_limit=0.5,
                                     name_suffix='/vel x')
        if isinstance(self.joint, OmniDrive):
            self.add_velocity_constraint(lower_velocity_limit=lba_y,
                                         upper_velocity_limit=uba_y,
                                         weight=weight_vel,
                                         task_expression=map_T_base_footprint.to_position().y,
                                         velocity_limit=0.5,
                                         name_suffix='/vel y')

    @profile
    def rot_error_at(self, t_in_s: int):
        rotation_goal = self.current_traj_point(self.joint.yaw.name, t_in_s)
        rotation_current = self.joint.yaw.get_symbol(Derivatives.position)
        error = w.shortest_angular_distance(rotation_current, rotation_goal) / self.sample_period
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
                                     task_expression=self.joint.yaw.get_symbol(Derivatives.position),
                                     velocity_limit=0.5,
                                     name_suffix='/rot')

    @profile
    def make_constraints(self):
        trajectory = self.god_map.get_data(identifier.trajectory)
        self.trajectory_length = len(trajectory.items())
        self.add_trans_constraints()
        self.add_rot_constraints()

    def __str__(self):
        return f'{super().__str__()}/{self.joint_name}'


class BaseTrajFollowerPR2(BaseTrajFollower):
    joint: OmniDrivePR22

    def make_constraints(self):
        constraints = super().make_constraints()
        return constraints

    @profile
    def add_trans_constraints(self):
        lb_yaw1 = []
        lb_forward = []
        self.world.state[self.joint.yaw1_vel.name].position = 0
        map_T_current = self.get_fk(self.world.root_link_name, self.base_footprint_link)
        map_P_current = map_T_current.to_position()
        self.add_debug_expr(f'map_P_current.x', map_P_current.x)
        self.add_debug_expr('time', self.god_map.to_expr(identifier.time))
        for t in range(self.prediction_horizon-2):
            trajectory_time_in_s = t * self.sample_period
            map_P_goal = self.make_map_T_base_footprint_goal(trajectory_time_in_s).to_position()
            map_V_error = (map_P_goal - map_P_current)
            self.add_debug_expr(f'map_P_goal.x/{t}', map_P_goal.x)
            self.add_debug_expr(f'map_V_error.x/{t}', map_V_error.x)
            self.add_debug_expr(f'map_V_error.y/{t}', map_V_error.y)
            weight = self.weight
            if t < 100:
                self.add_constraint(reference_velocity=self.joint.translation_limits[Derivatives.velocity],
                                    lower_error=map_V_error.x,
                                    upper_error=map_V_error.x,
                                    weight=weight,
                                    task_expression=map_P_current.x,
                                    name=f'base/x/{t:02d}',
                                    control_horizon=t+1)
                self.add_constraint(reference_velocity=self.joint.translation_limits[Derivatives.velocity],
                                    lower_error=map_V_error.y,
                                    upper_error=map_V_error.y,
                                    weight=weight,
                                    task_expression=map_P_current.y,
                                    name=f'base/y/{t:02d}',
                                    control_horizon=t+1)
            yaw1 = self.current_traj_point(self.joint.yaw1_vel.name, trajectory_time_in_s, Derivatives.velocity)
            lb_yaw1.append(yaw1)
            # if t == 0 and not self.track_only_velocity:
            #     lb_yaw1[-1] += self.rot_error_at(t)
            #     yaw1_goal_position = self.current_traj_point(self.joint.yaw1_vel.name, trajectory_time_in_s,
            #                                                  Derivatives.position)
            forward = self.current_traj_point(self.joint.forward_vel.name, t * self.sample_period,
                                              Derivatives.velocity)*1.1
            lb_forward.append(forward)
        weight_vel = WEIGHT_ABOVE_CA
        lba_yaw = lb_yaw1
        uba_yaw = lb_yaw1
        lba_forward = lb_forward
        uba_forward = lb_forward

        yaw1 = self.joint.yaw1_vel.get_symbol(Derivatives.position)
        yaw2 = self.joint.yaw.get_symbol(Derivatives.position)
        bf_yaw = yaw1 - yaw2
        x = w.cos(bf_yaw)
        y = w.sin(bf_yaw)
        v = w.Vector3([x, y, 0])
        v.vis_frame = 'pr2/base_footprint'
        v.reference_frame = 'pr2/base_footprint'
        self.add_debug_expr('v', v)

        # self.add_velocity_constraint(lower_velocity_limit=lba_yaw,
        #                              upper_velocity_limit=uba_yaw,
        #                              weight=weight_vel,
        #                              task_expression=self.joint.yaw1_vel.get_symbol(Derivatives.position),
        #                              velocity_limit=100,
        #                              name_suffix='/yaw1')
        self.add_velocity_constraint(lower_velocity_limit=lba_forward,
                                     upper_velocity_limit=uba_forward,
                                     weight=weight_vel,
                                     task_expression=self.joint.forward_vel.get_symbol(Derivatives.position),
                                     velocity_limit=self.joint.translation_limits[Derivatives.velocity],
                                     name_suffix='/forward')
