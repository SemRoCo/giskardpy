from __future__ import division

from copy import deepcopy
from typing import Optional

import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import rospy
from geometry_msgs.msg import PointStamped, Vector3Stamped, Vector3, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.model.joints import OmniDrive, OmniDrivePR22
from giskardpy.my_types import my_string, Derivatives, PrefixName
from giskardpy.utils.decorators import memoize_with_counter, clear_memo
from giskardpy.utils.tfwrapper import point_to_np


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
            # FIXME if less eq cases behavior changed
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


class CarryMyBullshit(Goal):
    trajectory: np.ndarray
    trajectory_data: list

    def __init__(self,
                 patrick_topic_name: str,
                 laser_topic_name: str = 'laser',
                 root_link: Optional[str] = None,
                 tip_link: str = 'base_footprint',
                 last_distance_threshold: float = 1,
                 laser_range: float = np.pi / 4,
                 max_rotation_velocity: float = 0.5,
                 max_translation_velocity: float = 0.5,
                 next_point_radius: float = 0.4,
                 max_temporal_distance_between_closest_and_next: float = 1):
        super().__init__()
        self.sub = rospy.Subscriber(patrick_topic_name, PointStamped, self.target_cb, queue_size=10)
        self.laser_sub = rospy.Subscriber(laser_topic_name, LaserScan, self.laser_cb, queue_size=10)
        self.pub = rospy.Publisher('~visualization_marker_array', MarkerArray)
        if root_link is None:
            self.root = self.world.root_link_name
        else:
            self.root = self.world.search_for_link_name(root_link)
        self.tip = self.world.search_for_link_name(tip_link)
        self.tip_V_pointing_axis = Vector3()
        self.tip_V_pointing_axis.x = 1
        self.max_rotation_velocity = max_rotation_velocity
        self.max_translation_velocity = max_translation_velocity
        self.weight = WEIGHT_ABOVE_CA
        self.trajectory = np.array([0, 0], ndmin=2)
        self.trajectory_data = [(0, 0, rospy.get_rostime().to_sec())]
        self.distance_to_target = last_distance_threshold
        self.radius = next_point_radius
        self.step_dt = 0.01
        self.interpolation_step_size = 0.1
        self.smoothing_factor = 0.6
        self.max_temp_distance = max_temporal_distance_between_closest_and_next
        self.max_temp_distance = int(self.max_temp_distance / self.step_dt)
        self.closest_laser_reading = 100
        self.laser_range = laser_range
        self.human_point = Point()
        # self.init_fake_path()
        while self.trajectory.shape[0] < 5 and not rospy.is_shutdown():
            print(f'waiting for at least 5 traj points, current length {len(self.trajectory)}')
            rospy.sleep(0.5)
        # self.publish_trajectory()

    def laser_cb(self, scan: LaserScan):
        center_id = int(len(scan.ranges) / 2)
        range_ = self.laser_range / scan.angle_increment
        min_id = int(center_id - range_)
        max_id = int(center_id + range_)
        segment = scan.ranges[min_id:max_id]
        self.closest_laser_reading = min(segment)

    def init_fake_path(self):
        rng = np.random.default_rng()
        self.traj_length = 5
        t = np.linspace(0, self.traj_length, 50)
        x = 2 * (-np.cos(2 * -t) + 0.1 * rng.standard_normal(50) + 2)
        y = 2 * (np.sin(2 * -t) + 0.1 * rng.standard_normal(50) + 1)

        spl_x = UnivariateSpline(t, x)
        spl_y = UnivariateSpline(t, y)
        ts = np.linspace(0, self.traj_length, int(self.traj_length / self.step_dt + 1))
        self.trajectory = np.vstack((spl_x(ts), spl_y(ts))).T

    @memoize_with_counter(6)
    def get_current_target(self):
        traj = self.trajectory.copy()
        root_T_tip = self.world.compute_fk_np(self.root, self.tip)
        x = root_T_tip[0, 3]
        y = root_T_tip[1, 3]
        current_point = np.array([x, y])
        error = traj - current_point
        distances = np.linalg.norm(error, axis=1)
        # cut off old points
        in_radius = np.where(distances < self.radius)[0]
        if len(in_radius) > 0:
            next_idx = max(in_radius)
            offset = max(0, next_idx - self.max_temp_distance)
            closest_idx = np.argmin(distances[offset:]) + offset
        else:
            next_idx = closest_idx = np.argmin(distances)

        if closest_idx <= 1:
            tangent = traj[next_idx] - current_point
        elif closest_idx >= len(traj) - 1:
            tangent = traj[-1] - traj[- 2]
        else:
            tangent = traj[closest_idx + 1] - traj[closest_idx - 1]
        result = {
            'next_x': traj[next_idx, 0],
            'next_y': traj[next_idx, 1],
            'closest_x': traj[closest_idx, 0],
            'closest_y': traj[closest_idx, 1],
            'tangent_x': tangent[0],
            'tangent_y': tangent[1],
        }
        return result

    def publish_trajectory(self):
        ms = MarkerArray()
        m_line = Marker()
        m_line.action = m_line.ADD
        m_line.ns = 'debug'
        m_line.id = 1
        m_line.type = m_line.LINE_STRIP
        m_line.header.frame_id = str(self.world.root_link_name)
        for item in self.trajectory:
            p = Point()
            p.x = item[0]
            p.y = item[1]
            m_line.points.append(p)
        m_line.scale.x = 0.05
        m_line.color.a = 1
        m_line.color.r = 1
        ms.markers.append(m_line)
        m_point = deepcopy(m_line)
        m_point.id = 2
        m_point.type = m_line.POINTS
        m_point.color.b = 1
        m_point.scale.y = m_point.scale.x
        m_point.points = []
        for item in self.trajectory_data:
            p = Point()
            p.x = item[0]
            p.y = item[1]
            m_point.points.append(p)
        ms.markers.append(m_point)

        self.pub.publish(ms)

    def project_point_to_floor(self, point: PointStamped):
        map_P = self.world.transform_point(self.world.root_link_name, point)
        map_P.point.z = 0
        return map_P

    def target_cb(self, point: PointStamped):
        now = point.header.stamp.to_sec()
        last_time = self.trajectory_data[-1][-1]
        current_point = np.array([point.point.x, point.point.y])
        last_point = np.array([self.trajectory_data[-1][0], self.trajectory_data[-1][1]])
        error_vector = current_point - last_point
        distance = np.linalg.norm(error_vector)
        error_vector /= distance
        if distance > self.interpolation_step_size * 2:
            ranges = np.arange(self.interpolation_step_size, distance, self.interpolation_step_size)
            dt = (now - last_time) / len(ranges)
            for i, interpolated_distance in enumerate(ranges):
                interpolated_point = last_point + error_vector * interpolated_distance
                self.trajectory_data.append((interpolated_point[0], interpolated_point[1], last_time + dt * (i + 1)))

        self.trajectory_data.append((current_point[0], current_point[1], now))
        traj_start = self.trajectory_data[0][2]
        traj_end = self.trajectory_data[-1][2]
        self.traj_length = traj_end - traj_start
        data = np.array(self.trajectory_data).T
        x = data[0]
        y = data[1]
        t = data[2]
        if len(x) < 5:
            return

        spl_x = UnivariateSpline(t, x)
        spl_x.set_smoothing_factor(self.smoothing_factor)
        spl_y = UnivariateSpline(t, y)
        spl_y.set_smoothing_factor(self.smoothing_factor)
        ts = np.linspace(traj_start, traj_end, int(self.traj_length / self.step_dt + 1))
        self.trajectory = np.vstack((spl_x(ts), spl_y(ts))).T
        self.human_point = point.point
        self.publish_trajectory()

    def make_constraints(self):
        root_T_tip = self.get_fk(self.root, self.tip)
        root_P_tip = root_T_tip.to_position()
        laser_center_reading = self.get_parameter_as_symbolic_expression('closest_laser_reading')
        map_P_human = w.Point3(self.get_parameter_as_symbolic_expression('human_point'))
        next_x = self.god_map.to_expr(self._get_identifier() + ['get_current_target', tuple(), 'next_x'])
        next_y = self.god_map.to_expr(self._get_identifier() + ['get_current_target', tuple(), 'next_y'])
        closest_x = self.god_map.to_expr(self._get_identifier() + ['get_current_target', tuple(), 'closest_x'])
        closest_y = self.god_map.to_expr(self._get_identifier() + ['get_current_target', tuple(), 'closest_y'])
        tangent_x = self.god_map.to_expr(self._get_identifier() + ['get_current_target', tuple(), 'tangent_x'])
        tangent_y = self.god_map.to_expr(self._get_identifier() + ['get_current_target', tuple(), 'tangent_y'])
        clear_memo(self.get_current_target)
        root_P_goal_point = w.Point3([next_x, next_y, 0])
        root_P_closest_point = w.Point3([closest_x, closest_y, 0])
        root_V_goal_axis = w.Vector3([tangent_x, tangent_y, 0])
        tip_V_pointing_axis = w.Vector3(self.tip_V_pointing_axis)

        # root_V_goal_axis = root_P_goal_point - root_P_tip
        root_V_goal_axis = map_P_human - root_P_tip
        distance_to_human = w.norm(root_V_goal_axis)
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip
        root_V_goal_axis.vis_frame = self.tip
        self.add_debug_expr('goal_point', root_P_goal_point)
        self.add_debug_expr('root_P_closest_point', root_P_closest_point)
        self.add_debug_expr('laser_center_reading', laser_center_reading)
        self.add_debug_expr('root_V_pointing_axis', root_V_pointing_axis)
        self.add_debug_expr('root_V_goal_axis', root_V_goal_axis)
        self.add_debug_expr('distance_to_human', distance_to_human)
        self.add_vector_goal_constraints(frame_V_current=root_V_pointing_axis,
                                         frame_V_goal=root_V_goal_axis,
                                         reference_velocity=self.max_rotation_velocity,
                                         weight=self.weight)

        # position_weight = self.weight
        position_weight = w.if_else(w.logic_or(w.less_equal(laser_center_reading, self.distance_to_target),
                                               w.less_equal(distance_to_human, self.distance_to_target)),
                                    0,
                                    self.weight)

        self.add_point_goal_constraints(frame_P_current=root_P_tip,
                                        frame_P_goal=root_P_goal_point,
                                        reference_velocity=self.max_translation_velocity,
                                        weight=position_weight,
                                        name='next')

        distance, _ = w.distance_point_to_line_segment(frame_P_current=root_P_tip,
                                                       frame_P_line_start=root_P_closest_point - root_V_goal_axis * 0.1,
                                                       frame_P_line_end=root_P_closest_point + root_V_goal_axis * 0.1)
        self.add_position_constraint(expr_current=distance,
                                     expr_goal=0,
                                     reference_velocity=self.max_translation_velocity,
                                     weight=position_weight * 10)

    def __str__(self) -> str:
        return super().__str__()


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
        for t in range(self.prediction_horizon - 2):
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
                                    control_horizon=t + 1)
                self.add_constraint(reference_velocity=self.joint.translation_limits[Derivatives.velocity],
                                    lower_error=map_V_error.y,
                                    upper_error=map_V_error.y,
                                    weight=weight,
                                    task_expression=map_P_current.y,
                                    name=f'base/y/{t:02d}',
                                    control_horizon=t + 1)
            yaw1 = self.current_traj_point(self.joint.yaw1_vel.name, trajectory_time_in_s, Derivatives.velocity)
            lb_yaw1.append(yaw1)
            # if t == 0 and not self.track_only_velocity:
            #     lb_yaw1[-1] += self.rot_error_at(t)
            #     yaw1_goal_position = self.current_traj_point(self.joint.yaw1_vel.name, trajectory_time_in_s,
            #                                                  Derivatives.position)
            forward = self.current_traj_point(self.joint.forward_vel.name, t * self.sample_period,
                                              Derivatives.velocity) * 1.1
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
