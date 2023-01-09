from __future__ import division

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.model.joints import OmniDrive, OmniDrivePR22
from giskardpy.my_types import my_string, Derivatives


class BaseTrajFollower(Goal):
    def __init__(self, joint_name: my_string, track_only_velocity: bool = False, weight: float = WEIGHT_ABOVE_CA):
        super().__init__()
        self.weight = weight
        self.joint_name = joint_name
        self.joint: OmniDrive = self.world._joints[joint_name]
        self.odom_link = self.joint.parent_link_name
        self.base_footprint_link = self.joint.child_link_name
        self.track_only_velocity = track_only_velocity

    @profile
    def x_symbol(self, t: int, free_variable_name: str, derivative: Derivatives = Derivatives.position) \
            -> w.Symbol:
        return self.god_map.to_symbol(identifier.trajectory + ['get_exact', (t,), free_variable_name, derivative])

    @profile
    def current_traj_point(self, free_variable_name: str, start_t: float,
                           derivative: Derivatives = Derivatives.position) \
            -> w.Expression:
        time = self.god_map.to_expr(identifier.time)
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
        x = self.current_traj_point(self.joint.x_name, t, derivative)
        if isinstance(self.joint, OmniDrive) or derivative == 0:
            y = self.current_traj_point(self.joint.y_name, t, derivative)
        else:
            y = 0
        rot = self.current_traj_point(self.joint.yaw_name, t, derivative)
        odom_T_base_footprint_goal = w.TransMatrix.from_xyz_rpy(x=x, y=y, yaw=rot)
        return odom_T_base_footprint_goal

    @profile
    def trans_error_at(self, t: float):
        odom_T_base_footprint_goal = self.make_odom_T_base_footprint_goal(t)
        map_T_odom = self.get_fk_evaluated(self.world.root_link_name, self.odom_link)
        map_T_base_footprint_goal = w.dot(map_T_odom, odom_T_base_footprint_goal)
        map_T_base_footprint_current = self.get_fk(self.world.root_link_name, self.base_footprint_link)

        frame_P_goal = map_T_base_footprint_goal.to_position()
        frame_P_current = map_T_base_footprint_current.to_position()
        error = (frame_P_goal[:3] - frame_P_current[:3]) / self.sample_period
        return error[0], error[1]

    @profile
    def add_trans_constraints(self):
        errors_x = []
        errors_y = []
        map_T_base_footprint = self.get_fk(self.world.root_link_name, self.base_footprint_link)
        for t in range(self.prediction_horizon):
            x = self.current_traj_point(self.joint.x_vel_name, t * self.sample_period, Derivatives.velocity)
            if isinstance(self.joint, OmniDrive):
                y = self.current_traj_point(self.joint.y_vel_name, t * self.sample_period, Derivatives.velocity)
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
    def rot_error_at(self, t: int):
        rotation_goal = self.current_traj_point(self.joint.yaw_name, t)
        rotation_current = self.joint.yaw.get_symbol(0)
        error = w.shortest_angular_distance(rotation_current, rotation_goal) / self.sample_period
        return error

    @profile
    def add_rot_constraints(self):
        errors = []
        for t in range(self.prediction_horizon):
            errors.append(self.current_traj_point(self.joint.yaw_name, t * self.sample_period, Derivatives.velocity))
            if t == 0 and not self.track_only_velocity:
                errors[-1] += self.rot_error_at(t)
        self.add_velocity_constraint(lower_velocity_limit=errors,
                                     upper_velocity_limit=errors,
                                     weight=WEIGHT_BELOW_CA,
                                     task_expression=self.joint.yaw_vel.get_symbol(0),
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

    @profile
    def add_trans_constraints(self):
        lb_yaw1 = []
        lb_forward = []
        self.world.state[self.joint.caster_yaw1_name].position = 0
        for t in range(self.prediction_horizon):
            yaw1 = self.current_traj_point(self.joint.caster_yaw1_name, t * self.sample_period, Derivatives.velocity)
            forward = self.current_traj_point(self.joint.caster_forward_name, t * self.sample_period,
                                              Derivatives.velocity)
            lb_yaw1.append(yaw1)
            lb_forward.append(forward)
        weight_vel = WEIGHT_ABOVE_CA
        lba_x = lb_yaw1
        uba_x = lb_yaw1
        lba_y = lb_forward
        uba_y = lb_forward

        self.add_velocity_constraint(lower_velocity_limit=lba_x,
                                     upper_velocity_limit=uba_x,
                                     weight=weight_vel,
                                     task_expression=self.joint.caster_yaw1.get_symbol(Derivatives.position),
                                     velocity_limit=10,
                                     name_suffix='/yaw1')
        self.add_velocity_constraint(lower_velocity_limit=lba_y,
                                     upper_velocity_limit=uba_y,
                                     weight=weight_vel,
                                     task_expression=self.joint.caster_forward.get_symbol(Derivatives.position),
                                     velocity_limit=0.5,
                                     name_suffix='/forward')
