from __future__ import division, annotations

from typing import Optional, List, Dict, Tuple

import numpy as np
# import matplotlib.pyplot as plt
import rospy
from geometry_msgs.msg import PointStamped, Vector3, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker

import giskardpy.casadi_wrapper as cas
from giskardpy.exceptions import GiskardException, GoalInitalizationException, ExecutionException
from giskardpy.goals.goal import Goal
from giskardpy.monitors.monitors import ExpressionMonitor, EndMotion
from giskardpy.tasks.task import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE, Task
from giskardpy.god_map import god_map
from giskardpy.model.joints import OmniDrive, OmniDrivePR22
from giskardpy.data_types import my_string, Derivatives, PrefixName
from giskardpy.utils import logging
from giskardpy.utils.decorators import memoize_with_counter, clear_memo
from giskardpy.utils.utils import raise_to_blackboard
from giskardpy.symbol_manager import symbol_manager


class BaseTrajFollower(Goal):
    def __init__(self,
                 joint_name: my_string,
                 track_only_velocity: bool = False,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        self.weight = weight
        self.joint_name = joint_name
        super().__init__(name=f'{self.__class__.__name__}/{self.joint_name}')
        self.joint: OmniDrive = god_map.world.joints[joint_name]
        self.odom_link = self.joint.parent_link_name
        self.base_footprint_link = self.joint.child_link_name
        self.track_only_velocity = track_only_velocity
        self.task = self.create_and_add_task()
        trajectory = god_map.trajectory
        self.trajectory_length = len(trajectory.items())
        self.add_trans_constraints()
        self.add_rot_constraints()
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)

    @profile
    def x_symbol(self, t: int, free_variable_name: PrefixName, derivative: Derivatives = Derivatives.position) \
            -> cas.Symbol:
        expr = f'god_map.trajectory.get_exact({t})[\'{free_variable_name}\'][{derivative}]'
        return symbol_manager.get_symbol(expr)

    @profile
    def current_traj_point(self, free_variable_name: PrefixName, start_t: float,
                           derivative: Derivatives = Derivatives.position) \
            -> cas.Expression:
        time = symbol_manager.time
        b_result_cases = []
        for t in range(self.trajectory_length):
            b = t * god_map.qp_controller_config.sample_period
            eq_result = self.x_symbol(t, free_variable_name, derivative)
            b_result_cases.append((b, eq_result))
            # FIXME if less eq cases behavior changed
        return cas.if_less_eq_cases(a=time + start_t,
                                    b_result_cases=b_result_cases,
                                    else_result=self.x_symbol(self.trajectory_length - 1, free_variable_name,
                                                              derivative))

    @profile
    def make_odom_T_base_footprint_goal(self, t_in_s: float, derivative: Derivatives = Derivatives.position):
        x = self.current_traj_point(self.joint.x.name, t_in_s, derivative)
        if isinstance(self.joint, OmniDrive) or derivative == 0:
            y = self.current_traj_point(self.joint.y.name, t_in_s, derivative)
        else:
            y = 0
        rot = self.current_traj_point(self.joint.yaw.name, t_in_s, derivative)
        odom_T_base_footprint_goal = cas.TransMatrix.from_xyz_rpy(x=x, y=y, yaw=rot)
        return odom_T_base_footprint_goal

    @profile
    def make_map_T_base_footprint_goal(self, t_in_s: float, derivative: Derivatives = Derivatives.position):
        odom_T_base_footprint_goal = self.make_odom_T_base_footprint_goal(t_in_s, derivative)
        map_T_odom = god_map.world.compose_fk_evaluated_expression(god_map.world.root_link_name, self.odom_link)
        return cas.dot(map_T_odom, odom_T_base_footprint_goal)

    @profile
    def trans_error_at(self, t_in_s: float):
        odom_T_base_footprint_goal = self.make_odom_T_base_footprint_goal(t_in_s)
        map_T_odom = god_map.world.compose_fk_evaluated_expression(god_map.world.root_link_name, self.odom_link)
        map_T_base_footprint_goal = cas.dot(map_T_odom, odom_T_base_footprint_goal)
        map_T_base_footprint_current = god_map.world.compose_fk_expression(god_map.world.root_link_name,
                                                                           self.base_footprint_link)

        frame_P_goal = map_T_base_footprint_goal.to_position()
        frame_P_current = map_T_base_footprint_current.to_position()
        error = (frame_P_goal - frame_P_current) / god_map.qp_controller_config.sample_period
        return error[0], error[1]

    @profile
    def add_trans_constraints(self):
        errors_x = []
        errors_y = []
        map_T_base_footprint = god_map.world.compose_fk_expression(god_map.world.root_link_name,
                                                                   self.base_footprint_link)
        for t in range(god_map.qp_controller_config.prediction_horizon):
            x = self.current_traj_point(self.joint.x_vel.name, t * god_map.qp_controller_config.sample_period,
                                        Derivatives.velocity)
            if isinstance(self.joint, OmniDrive):
                y = self.current_traj_point(self.joint.y_vel.name, t * god_map.qp_controller_config.sample_period,
                                            Derivatives.velocity)
            else:
                y = 0
            base_footprint_P_vel = cas.Vector3((x, y, 0))
            map_P_vel = cas.dot(map_T_base_footprint, base_footprint_P_vel)
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

        self.task.add_velocity_constraint(lower_velocity_limit=lba_x,
                                          upper_velocity_limit=uba_x,
                                          weight=weight_vel,
                                          task_expression=map_T_base_footprint.to_position().x,
                                          velocity_limit=0.5,
                                          name='/vel x')
        if isinstance(self.joint, OmniDrive):
            self.task.add_velocity_constraint(lower_velocity_limit=lba_y,
                                              upper_velocity_limit=uba_y,
                                              weight=weight_vel,
                                              task_expression=map_T_base_footprint.to_position().y,
                                              velocity_limit=0.5,
                                              name='/vel y')

    @profile
    def rot_error_at(self, t_in_s: int):
        rotation_goal = self.current_traj_point(self.joint.yaw.name, t_in_s)
        rotation_current = self.joint.yaw.get_symbol(Derivatives.position)
        error = cas.shortest_angular_distance(rotation_current,
                                              rotation_goal) / god_map.qp_controller_config.sample_period
        return error

    @profile
    def add_rot_constraints(self):
        errors = []
        for t in range(god_map.qp_controller_config.prediction_horizon):
            errors.append(self.current_traj_point(self.joint.yaw.name, t * god_map.qp_controller_config.sample_period,
                                                  Derivatives.velocity))
            if t == 0 and not self.track_only_velocity:
                errors[-1] += self.rot_error_at(t)
        self.task.add_velocity_constraint(lower_velocity_limit=errors,
                                          upper_velocity_limit=errors,
                                          weight=WEIGHT_BELOW_CA,
                                          task_expression=self.joint.yaw.get_symbol(Derivatives.position),
                                          velocity_limit=0.5,
                                          name='/rot')


class FollowNavPath(Goal):
    trajectory: np.ndarray
    traj_data: List[np.ndarray]
    laser_thresholds: Dict[int, np.ndarray]
    pub: rospy.Publisher = None
    laser_subs: List[rospy.Subscriber]
    last_scan: Dict[int, LaserScan]
    odom_joint: OmniDrive

    def __init__(self,
                 name: str,
                 path: Path,
                 laser_topics: Tuple[str] = ('/hsrb/base_scan',),
                 odom_joint_name: Optional[str] = None,
                 root_link: Optional[str] = None,
                 camera_link: str = 'head_rgbd_sensor_link',
                 laser_scan_age_threshold: float = 2,
                 laser_distance_threshold: float = 0.5,
                 laser_distance_threshold_width: float = 0.8,
                 laser_avoidance_angle_cutout: float = np.pi / 4,
                 laser_avoidance_sideways_buffer: float = 0.04,
                 base_orientation_threshold: float = np.pi / 16,
                 max_rotation_velocity: float = 0.5,
                 max_rotation_velocity_head: float = 1,
                 max_translation_velocity: float = 0.38,
                 traj_tracking_radius: float = 0.5,
                 height_for_camera_target: float = 1,
                 laser_frame_id: str = 'base_range_sensor_link',
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name)
        self.end_of_traj_reached = False
        self.laser_thresholds = {}
        self.last_scan = {}
        self.enable_laser_avoidance = len(laser_topics) > 0
        if FollowNavPath.pub is None:
            FollowNavPath.pub = rospy.Publisher('~visualization_marker_array', MarkerArray)
        self.laser_topics = list(laser_topics)
        self.laser_distance_threshold_width = laser_distance_threshold_width / 2
        self.closest_laser_left = [self.laser_distance_threshold_width] * len(self.laser_topics)
        self.closest_laser_right = [-self.laser_distance_threshold_width] * len(self.laser_topics)
        self.closest_laser_reading = [0] * len(self.laser_topics)
        self.laser_frame = laser_frame_id
        self.laser_scan_age_threshold = laser_scan_age_threshold
        self.laser_avoidance_angle_cutout = laser_avoidance_angle_cutout
        self.laser_avoidance_sideways_buffer = laser_avoidance_sideways_buffer
        self.base_orientation_threshold = base_orientation_threshold
        if odom_joint_name is None:
            self.odom_joint = god_map.world.search_for_joint_of_type(OmniDrive)[0]
            self.odom_joint_name = self.odom_joint.name
        else:
            self.odom_joint_name = god_map.world.search_for_joint_name(odom_joint_name)
            self.odom_joint = god_map.world.get_joint(self.odom_joint_name)
        if root_link is None:
            self.root = god_map.world.root_link_name
        else:
            self.root = god_map.world.search_for_link_name(root_link)
        self.camera_link = god_map.world.search_for_link_name(camera_link)
        self.tip_V_camera_axis = Vector3()
        self.tip_V_camera_axis.z = 1
        self.tip = self.odom_joint.child_link_name
        self.odom = self.odom_joint.parent_link_name
        self.tip_V_pointing_axis = Vector3()
        self.tip_V_pointing_axis.x = 1
        self.max_rotation_velocity = max_rotation_velocity
        self.max_rotation_velocity_head = max_rotation_velocity_head
        self.max_translation_velocity = max_translation_velocity
        self.weight = WEIGHT_BELOW_CA
        self.laser_distance_threshold = laser_distance_threshold
        self.traj_tracking_radius = traj_tracking_radius
        self.interpolation_step_size = 0.05
        self.max_temp_distance = int(self.traj_tracking_radius / self.interpolation_step_size)
        self.human_point = PointStamped()
        self.height_for_camera_target = height_for_camera_target
        self.laser_subs = []
        for i, laser_topic in enumerate(self.laser_topics):
            cb = lambda scan: self.laser_cb(scan, i)
            self.laser_subs.append(rospy.Subscriber(laser_topic, LaserScan, cb, queue_size=10))
        self.publish_tracking_radius()
        self.path_to_trajectory(path=path)

        # %% real shit
        root_T_bf = god_map.world.compose_fk_expression(self.root, self.tip)
        root_T_odom = god_map.world.compose_fk_expression(self.root, self.odom)
        root_T_camera = god_map.world.compose_fk_expression(self.root, self.camera_link)
        root_P_bf = root_T_bf.to_position()

        if self.enable_laser_avoidance:
            closest_laser_left = symbol_manager.get_symbol(self + f'.closest_laser_left[0]')
            closest_laser_right = symbol_manager.get_symbol(self + f'.closest_laser_right[0]')
            closest_laser_reading = symbol_manager.get_symbol(self + f'.closest_laser_reading[0]')
            for laser_id in range(1, len(self.laser_subs)):
                next_min_left_violation = symbol_manager.get_symbol(self + f'.closest_laser_left[{laser_id}]')
                next_min_right_violation = symbol_manager.get_symbol(self + f'.closest_laser_right[{laser_id}]')
                next_closest_laser_reading = symbol_manager.get_symbol(self + f'.closest_laser_reading[{laser_id}]')
                closest_laser_left = cas.min(closest_laser_left, next_min_left_violation)
                closest_laser_right = cas.max(closest_laser_right, next_min_right_violation)
                closest_laser_reading = cas.min(closest_laser_reading, next_closest_laser_reading)

        next_x = symbol_manager.get_symbol(self + '.get_current_target()[\'next_x\']')
        next_y = symbol_manager.get_symbol(self + '.get_current_target()[\'next_y\']')
        closest_x = symbol_manager.get_symbol(self + '.get_current_target()[\'closest_x\']')
        closest_y = symbol_manager.get_symbol(self + '.get_current_target()[\'closest_y\']')
        clear_memo(self.get_current_target)
        root_P_goal_point = cas.Point3([next_x, next_y, 0])
        root_P_closest_point = cas.Point3([closest_x, closest_y, 0])
        tip_V_pointing_axis = cas.Vector3(self.tip_V_pointing_axis)

        map_P_human = root_P_goal_point

        # %% orient to goal
        orient_to_goal = self.create_and_add_task('orient to goal')
        _, _, map_odom_angle = root_T_odom.to_rotation().to_rpy()
        odom_current_angle = self.odom_joint.yaw.get_symbol(Derivatives.position)
        map_current_angle = map_odom_angle + odom_current_angle
        root_V_tip_to_closest = root_P_bf - root_P_closest_point
        root_P_between_tip_and_closest = root_P_closest_point + root_V_tip_to_closest / 2
        root_V_goal_axis = root_P_goal_point - root_P_between_tip_and_closest
        root_V_goal_axis2 = cas.Vector3(root_V_goal_axis)
        root_V_goal_axis2.scale(1)
        map_P_human = map_P_human + root_V_goal_axis2 * 1.5
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_bf.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip
        root_V_goal_axis.vis_frame = self.tip
        map_goal_angle = cas.angle_between_vector(cas.Vector3([1, 0, 0]), root_V_goal_axis)
        map_goal_angle = cas.if_greater(root_V_goal_axis.y, 0, map_goal_angle, -map_goal_angle)
        map_angle_error = cas.shortest_angular_distance(map_current_angle, map_goal_angle)
        buffer = 0
        ll = map_angle_error - buffer
        ul = map_angle_error + buffer
        orient_to_goal.add_inequality_constraint(reference_velocity=self.max_rotation_velocity,
                                                 lower_error=ll,
                                                 upper_error=ul,
                                                 weight=self.weight,
                                                 task_expression=map_current_angle,
                                                 name='rot')

        # %% look at goal
        camera_V_camera_axis = cas.Vector3(self.tip_V_camera_axis)
        root_V_camera_axis = root_T_camera.dot(camera_V_camera_axis)
        root_P_camera = root_T_camera.to_position()
        map_P_human.z = self.height_for_camera_target
        root_V_camera_goal_axis = map_P_human - root_P_camera
        root_V_camera_goal_axis.scale(1)
        look_at_target = self.create_and_add_task('look at target')
        look_at_target.add_vector_goal_constraints(frame_V_current=root_V_camera_axis,
                                                   frame_V_goal=root_V_camera_goal_axis,
                                                   reference_velocity=self.max_rotation_velocity_head,
                                                   weight=self.weight,
                                                   name='move camera')

        # %% follow next point
        follow_next_point = self.create_and_add_task('follow next')
        root_V_camera_axis.vis_frame = self.camera_link
        root_V_camera_goal_axis.vis_frame = self.camera_link

        oriented_towards_next = ExpressionMonitor(name='oriented towards next')
        self.add_monitor(oriented_towards_next)
        oriented_towards_next.expression = cas.abs(map_angle_error) > self.base_orientation_threshold

        follow_next_point.hold_condition = oriented_towards_next.get_state_expression()
        if self.enable_laser_avoidance:
            laser_violated = ExpressionMonitor(name='laser violated')
            self.add_monitor(laser_violated)
            laser_violated.expression = cas.less(closest_laser_reading, 0)
            follow_next_point.hold_condition |= laser_violated.get_state_expression()
        follow_next_point.add_point_goal_constraints(frame_P_current=root_P_bf,
                                                     frame_P_goal=root_P_goal_point,
                                                     reference_velocity=self.max_translation_velocity,
                                                     weight=self.weight,
                                                     name='min dist to next')
        god_map.debug_expression_manager.add_debug_expression('root_P_goal_point', root_P_goal_point)

        # %% keep the closest point in footprint radius
        stay_in_circle = self.create_and_add_task('in circle')
        buffer = self.traj_tracking_radius
        hack = cas.Vector3([0, 0, 0.0001])
        distance_to_closest_point = cas.norm(root_P_closest_point + hack - root_P_bf)
        stay_in_circle.add_inequality_constraint(task_expression=distance_to_closest_point,
                                                 lower_error=-distance_to_closest_point - buffer,
                                                 upper_error=-distance_to_closest_point + buffer,
                                                 reference_velocity=self.max_translation_velocity,
                                                 weight=self.weight,
                                                 name='stay in circle')

        # %% laser avoidance
        if self.enable_laser_avoidance:
            laser_avoidance_task = self.create_and_add_task('laser avoidance')
            sideways_vel = (closest_laser_left + closest_laser_right)
            bf_V_laser_avoidance_direction = cas.Vector3([0, sideways_vel, 0])
            map_V_laser_avoidance_direction = root_T_bf.dot(bf_V_laser_avoidance_direction)
            map_V_laser_avoidance_direction.vis_frame = god_map.world.search_for_link_name(self.laser_frame)
            god_map.debug_expression_manager.add_debug_expression('base_V_laser_avoidance_direction',
                                                                  map_V_laser_avoidance_direction)
            odom_y_vel = self.odom_joint.y_vel.get_symbol(Derivatives.position)

            active = ExpressionMonitor(name='too far from path')
            self.add_monitor(active)
            active.expression = cas.greater(distance_to_closest_point, self.traj_tracking_radius)

            buffer = self.laser_avoidance_sideways_buffer / 2

            laser_avoidance_task.hold_condition = active.get_state_expression()
            laser_avoidance_task.add_inequality_constraint(reference_velocity=self.max_translation_velocity,
                                                           lower_error=sideways_vel - buffer,
                                                           upper_error=sideways_vel + buffer,
                                                           weight=WEIGHT_COLLISION_AVOIDANCE,
                                                           task_expression=odom_y_vel,
                                                           name='move sideways')

        last_point = cas.Point3([self.trajectory[-1][0], self.trajectory[-1][1], 0])
        goal_reached = ExpressionMonitor(name='goal reached?')
        self.add_monitor(goal_reached)
        goal_reached.expression = cas.euclidean_distance(last_point, root_P_bf) < 0.03
        self.connect_end_condition_to_all_tasks(goal_reached.get_state_expression())

        final_orientation = self.create_and_add_task('final orientation')
        frame_R_current = root_T_bf.to_rotation()
        current_R_frame_eval = god_map.world.compose_fk_evaluated_expression(self.tip, self.root).to_rotation()
        frame_R_goal = cas.TransMatrix(path.poses[-1]).to_rotation()
        final_orientation.add_rotation_goal_constraints(frame_R_current=frame_R_current,
                                                        frame_R_goal=frame_R_goal,
                                                        current_R_frame_eval=current_R_frame_eval,
                                                        reference_velocity=self.max_rotation_velocity,
                                                        weight=self.weight)
        final_orientation.start_condition = goal_reached.get_state_expression()

        orientation_reached = ExpressionMonitor(name='final orientation reached',
                                                start_condition=goal_reached.get_state_expression())
        self.add_monitor(orientation_reached)
        rotation_error = cas.rotational_error(frame_R_current, frame_R_goal)
        orientation_reached.expression = cas.less(cas.abs(rotation_error), 0.01)

        end = EndMotion(name='done')
        end.start_condition = orientation_reached.get_state_expression()
        self.add_monitor(end)
        self.connect_start_condition_to_all_tasks(start_condition)
        self.connect_hold_condition_to_all_tasks(hold_condition)
        self.connect_end_condition_to_all_tasks(end_condition)

    def path_to_trajectory(self, path: Path):
        self.traj_data = [self.get_current_point()]
        for p in path.poses:
            self.append_point_to_traj(p.pose.position.x, p.pose.position.y)
        self.trajectory = np.array(self.traj_data)
        self.publish_trajectory()

    def append_point_to_traj(self, x: float, y: float):
        current_point = np.array([x, y], dtype=np.float64)
        last_point = self.traj_data[-1]
        error_vector = current_point - last_point
        distance = np.linalg.norm(error_vector)
        if distance < self.interpolation_step_size * 2:
            self.traj_data[-1] = 0.5 * self.traj_data[-1] + 0.5 * current_point
        else:
            error_vector /= distance
            ranges = np.arange(self.interpolation_step_size, distance, self.interpolation_step_size)
            interpolated_distance = distance / len(ranges)
            for i, dt in enumerate(ranges):
                interpolated_point = last_point + error_vector * interpolated_distance * (i + 1)
                self.traj_data.append(interpolated_point)
            self.traj_data.append(current_point)

    def clean_up(self):
        for sub in self.laser_subs:
            sub.unregister()

    def init_laser_stuff(self, laser_scan: LaserScan):
        thresholds = []
        if len(laser_scan.ranges) % 2 == 0:
            print('laser range is even')
            angles = np.arange(laser_scan.angle_min,
                               laser_scan.angle_max,
                               laser_scan.angle_increment)[:-1]
        else:
            angles = np.arange(laser_scan.angle_min,
                               laser_scan.angle_max,
                               laser_scan.angle_increment)
        for angle in angles:
            if angle < 0:
                y = -self.laser_distance_threshold_width
                length = y / np.sin((angle))
                x = np.cos(angle) * length
                thresholds.append((x, y, length, angle))
            else:
                y = self.laser_distance_threshold_width
                length = y / np.sin((angle))
                x = np.cos(angle) * length
                thresholds.append((x, y, length, angle))
            if length > self.laser_distance_threshold:
                length = self.laser_distance_threshold
                x = np.cos(angle) * length
                y = np.sin(angle) * length
                thresholds[-1] = (x, y, length, angle)
        thresholds = np.array(thresholds)
        assert len(thresholds) == len(laser_scan.ranges)
        return thresholds

    def muddle_laser_scan(self, scan: LaserScan, thresholds: np.ndarray):
        data = np.array(scan.ranges)
        xs = np.cos(thresholds[:, 3]) * data
        ys = np.sin(thresholds[:, 3]) * data
        violations = data < thresholds[:, 2]
        xs_error = xs - thresholds[:, 0]
        half = int(data.shape[0] / 2)
        # x_positive = np.where(thresholds[:, 0] > 0)[0]
        x_below_laser_avoidance_threshold1 = thresholds[:, -1] > self.laser_avoidance_angle_cutout
        x_below_laser_avoidance_threshold2 = thresholds[:, -1] < -self.laser_avoidance_angle_cutout
        x_below_laser_avoidance_threshold = x_below_laser_avoidance_threshold1 | x_below_laser_avoidance_threshold2
        y_filter = x_below_laser_avoidance_threshold & violations
        closest_laser_right = ys[:half][y_filter[:half]]
        closest_laser_left = ys[half:][y_filter[half:]]

        x_positive = np.where(np.invert(x_below_laser_avoidance_threshold))[0]
        x_start = x_positive[0]
        x_end = x_positive[-1]

        front_violation = xs_error[x_start:x_end][violations[x_start:x_end]]
        if len(closest_laser_left) > 0:
            closest_laser_left = min(closest_laser_left)
        else:
            closest_laser_left = self.laser_distance_threshold_width
        if len(closest_laser_right) > 0:
            closest_laser_right = max(closest_laser_right)
        else:
            closest_laser_right = -self.laser_distance_threshold_width
        if len(front_violation) > 0:
            closest_laser_reading = min(front_violation)
        else:
            closest_laser_reading = 0
        return closest_laser_reading, closest_laser_left, closest_laser_right

    def laser_cb(self, scan: LaserScan, id_: int):
        self.last_scan[id_] = scan
        if id_ not in self.laser_thresholds:
            self.laser_thresholds[id_] = self.init_laser_stuff(scan)
            self.publish_laser_thresholds()
        closest, left, right = self.muddle_laser_scan(scan, self.laser_thresholds[id_])
        self.closest_laser_reading[id_] = closest
        self.closest_laser_left[id_] = left
        self.closest_laser_right[id_] = right

    def get_current_point(self) -> np.ndarray:
        root_T_tip = god_map.world.compute_fk_np(self.root, self.tip)
        x = root_T_tip[0, 3]
        y = root_T_tip[1, 3]
        return np.array([x, y])

    @memoize_with_counter(4)
    def get_current_target(self) -> Dict[str, float]:
        if self.enable_laser_avoidance:
            self.check_laser_scan_age()
        traj = self.trajectory.copy()
        current_point = self.get_current_point()
        error = traj - current_point
        distances = np.linalg.norm(error, axis=1)
        # cut off old points
        in_radius = np.where(distances < self.traj_tracking_radius)[0]
        if len(in_radius) > 0:
            next_idx = max(in_radius)
            offset = max(0, next_idx - self.max_temp_distance)
            closest_idx = np.argmin(distances[offset:]) + offset
        else:
            next_idx = closest_idx = np.argmin(distances)
        # CarryMyBullshit.traj_data = CarryMyBullshit.traj_data[closest_idx:]
        if closest_idx == self.trajectory.shape[0] - 1:
            self.end_of_traj_reached = True
        result = {
            'next_x': traj[next_idx, 0],
            'next_y': traj[next_idx, 1],
            'closest_x': traj[closest_idx, 0],
            'closest_y': traj[closest_idx, 1],
        }
        return result

    def check_laser_scan_age(self):
        current_time = rospy.get_rostime().to_sec()
        for id_, scan in self.last_scan.items():
            base_laser_age = current_time - scan.header.stamp.to_sec()
            if base_laser_age > self.laser_scan_age_threshold:
                logging.logwarn(f'last base laser scan is too old: {base_laser_age}')
                self.closest_laser_left[id_] = self.laser_distance_threshold_width
                self.closest_laser_right[id_] = -self.laser_distance_threshold_width
                self.closest_laser_reading[id_] = 0

    def publish_trajectory(self):
        ms = MarkerArray()
        m_line = Marker()
        m_line.action = m_line.ADD
        m_line.ns = 'traj'
        m_line.id = 1
        m_line.type = m_line.LINE_STRIP
        m_line.header.frame_id = str(god_map.world.root_link_name)
        m_line.scale.x = 0.05
        m_line.color.a = 1
        m_line.color.r = 1
        try:
            for item in self.trajectory:
                p = Point()
                p.x = item[0]
                p.y = item[1]
                m_line.points.append(p)
            ms.markers.append(m_line)
        except Exception as e:
            logging.logwarn('failed to create traj marker')
        self.pub.publish(ms)

    def publish_laser_thresholds(self):
        ms = MarkerArray()
        m_line = Marker()
        m_line.action = m_line.ADD
        m_line.ns = 'laser_thresholds'
        m_line.id = 1332
        m_line.type = m_line.LINE_STRIP
        m_line.header.frame_id = self.laser_frame
        m_line.scale.x = 0.05
        m_line.color.a = 1
        m_line.color.r = 0.5
        m_line.color.b = 1
        m_line.frame_locked = True
        for item in self.laser_thresholds[0]:
            p = Point()
            p.x = item[0]
            p.y = item[1]
            m_line.points.append(p)
        ms.markers.append(m_line)
        square = Marker()
        square.action = m_line.ADD
        square.ns = 'laser_avoidance_angle_cutout'
        square.id = 1333
        square.type = m_line.LINE_STRIP
        square.header.frame_id = self.laser_frame
        # p = Point()
        # p.x = self.thresholds[0, 0]
        # p.y = self.thresholds[0, 1]
        # square.points.append(p)
        p = Point()
        idx = np.where(self.laser_thresholds[0][:, -1] < -self.laser_avoidance_angle_cutout)[0][-1]
        p.x = self.laser_thresholds[0][idx, 0]
        p.y = self.laser_thresholds[0][idx, 1]
        square.points.append(p)
        p = Point()
        square.points.append(p)
        p = Point()
        idx = np.where(self.laser_thresholds[0][:, -1] > self.laser_avoidance_angle_cutout)[0][0]
        p.x = self.laser_thresholds[0][idx, 0]
        p.y = self.laser_thresholds[0][idx, 1]
        square.points.append(p)
        # p = Point()
        # p.x = self.thresholds[-1, 0]
        # p.y = self.thresholds[-1, 1]
        # square.points.append(p)
        square.scale.x = 0.05
        square.color.a = 1
        square.color.r = 0.5
        square.color.b = 1
        square.frame_locked = True
        ms.markers.append(square)
        self.pub.publish(ms)

    def publish_tracking_radius(self):
        ms = MarkerArray()
        m_line = Marker()
        m_line.action = m_line.ADD
        m_line.ns = 'traj_tracking_radius'
        m_line.id = 1332
        m_line.type = m_line.CYLINDER
        m_line.header.frame_id = str(self.tip.short_name)
        m_line.scale.x = self.traj_tracking_radius * 2
        m_line.scale.y = self.traj_tracking_radius * 2
        m_line.scale.z = 0.05
        m_line.color.a = 1
        m_line.color.b = 1
        m_line.pose.orientation.w = 1
        m_line.frame_locked = True
        ms.markers.append(m_line)
        FollowNavPath.pub.publish(ms)


class CarryMyBullshit(Goal):
    trajectory: np.ndarray = np.array([])
    traj_data: List[np.ndarray] = None
    thresholds: np.ndarray = None
    thresholds_pc: np.ndarray = None
    human_point: PointStamped = None
    pub: rospy.Publisher = None
    laser_sub: rospy.Subscriber = None
    point_cloud_laser_sub: rospy.Subscriber = None
    target_sub: rospy.Subscriber = None
    traj_flipped: bool = False
    last_scan: LaserScan = None
    last_scan_pc: LaserScan = None

    def __init__(self,
                 name: str,
                 patrick_topic_name: str = '/robokudovanessa/human_position',
                 laser_topic_name: str = '/hsrb/base_scan',
                 point_cloud_laser_topic_name: Optional[str] = '/scan',
                 odom_joint_name: str = 'brumbrum',
                 root_link: Optional[str] = None,
                 camera_link: str = 'head_rgbd_sensor_link',
                 distance_to_target_stop_threshold: float = 1,
                 laser_scan_age_threshold: float = 2,
                 laser_distance_threshold: float = 0.5,
                 laser_distance_threshold_width: float = 0.8,
                 laser_avoidance_angle_cutout: float = np.pi / 4,
                 laser_avoidance_sideways_buffer: float = 0.04,
                 base_orientation_threshold: float = np.pi / 16,
                 wait_for_patrick_timeout: int = 30,
                 max_rotation_velocity: float = 0.5,
                 max_rotation_velocity_head: float = 1,
                 max_translation_velocity: float = 0.38,
                 traj_tracking_radius: float = 0.4,
                 height_for_camera_target: float = 1,
                 laser_frame_id: str = 'base_range_sensor_link',
                 target_age_threshold: float = 2,
                 target_age_exception_threshold: float = 5,
                 clear_path: bool = False,
                 drive_back: bool = False,
                 enable_laser_avoidance: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name)
        if drive_back:
            logging.loginfo('driving back')
        self.end_of_traj_reached = False
        self.enable_laser_avoidance = enable_laser_avoidance
        if CarryMyBullshit.pub is None:
            CarryMyBullshit.pub = rospy.Publisher('~visualization_marker_array', MarkerArray)
        self.laser_topic_name = laser_topic_name
        if point_cloud_laser_topic_name == '':
            self.point_cloud_laser_topic_name = None
        else:
            self.point_cloud_laser_topic_name = point_cloud_laser_topic_name
        self.laser_distance_threshold_width = laser_distance_threshold_width / 2
        self.last_target_age = 0
        self.closest_laser_left = self.laser_distance_threshold_width
        self.closest_laser_right = -self.laser_distance_threshold_width
        self.closest_laser_reading = 0
        self.closest_laser_left_pc = self.laser_distance_threshold_width
        self.closest_laser_right_pc = -self.laser_distance_threshold_width
        self.closest_laser_reading_pc = 0
        self.laser_frame = laser_frame_id
        self.last_scan = LaserScan()
        self.last_scan.header.stamp = rospy.get_rostime()
        self.last_scan_pc = LaserScan()
        self.last_scan_pc.header.stamp = rospy.get_rostime()
        self.laser_scan_age_threshold = laser_scan_age_threshold
        self.laser_avoidance_angle_cutout = laser_avoidance_angle_cutout
        self.laser_avoidance_sideways_buffer = laser_avoidance_sideways_buffer
        self.base_orientation_threshold = base_orientation_threshold
        self.odom_joint_name = god_map.world.search_for_joint_name(odom_joint_name)
        self.odom_joint: OmniDrive = god_map.world.get_joint(self.odom_joint_name)
        self.target_age_threshold = target_age_threshold
        self.target_age_exception_threshold = target_age_exception_threshold
        if root_link is None:
            self.root = god_map.world.root_link_name
        else:
            self.root = god_map.world.search_for_link_name(root_link)
        self.camera_link = god_map.world.search_for_link_name(camera_link)
        self.tip_V_camera_axis = Vector3()
        self.tip_V_camera_axis.z = 1
        self.tip = self.odom_joint.child_link_name
        self.odom = self.odom_joint.parent_link_name
        self.tip_V_pointing_axis = Vector3()
        self.tip_V_pointing_axis.x = 1
        self.max_rotation_velocity = max_rotation_velocity
        self.max_rotation_velocity_head = max_rotation_velocity_head
        self.max_translation_velocity = max_translation_velocity
        self.weight = WEIGHT_BELOW_CA
        self.min_distance_to_target = distance_to_target_stop_threshold
        self.laser_distance_threshold = laser_distance_threshold
        self.traj_tracking_radius = traj_tracking_radius
        self.interpolation_step_size = 0.05
        self.max_temp_distance = int(self.traj_tracking_radius / self.interpolation_step_size)
        self.human_point = PointStamped()
        self.height_for_camera_target = height_for_camera_target
        self.drive_back = drive_back
        if clear_path or (not self.drive_back and CarryMyBullshit.trajectory is None):
            CarryMyBullshit.trajectory = np.array(self.get_current_point(), ndmin=2)
        if clear_path or CarryMyBullshit.traj_data is None:
            CarryMyBullshit.traj_data = [self.get_current_point()]
        if clear_path:
            CarryMyBullshit.traj_flipped = False
            logging.loginfo('cleared old path')
        if CarryMyBullshit.laser_sub is None:
            CarryMyBullshit.laser_sub = rospy.Subscriber(self.laser_topic_name, LaserScan, self.laser_cb, queue_size=10)
        if CarryMyBullshit.point_cloud_laser_sub is None and self.point_cloud_laser_topic_name is not None:
            CarryMyBullshit.point_cloud_laser_sub = rospy.Subscriber(self.point_cloud_laser_topic_name,
                                                                     LaserScan, self.point_cloud_laser_cb,
                                                                     queue_size=10)
        self.publish_tracking_radius()
        self.publish_distance_to_target()
        if not self.drive_back:
            if CarryMyBullshit.target_sub is None:
                CarryMyBullshit.target_sub = rospy.Subscriber(patrick_topic_name, PointStamped, self.target_cb,
                                                              queue_size=10)
            rospy.sleep(0.5)
            for i in range(int(wait_for_patrick_timeout)):
                if CarryMyBullshit.trajectory.shape[0] > 5:
                    break
                print(f'waiting for at least 5 traj points, current length {len(CarryMyBullshit.trajectory)}')
                rospy.sleep(1)
            else:
                raise GoalInitalizationException(
                    f'didn\'t receive enough points after {wait_for_patrick_timeout}s')
            logging.loginfo(f'waiting for one more target point for {wait_for_patrick_timeout}s')
            rospy.wait_for_message(patrick_topic_name, PointStamped, rospy.Duration(wait_for_patrick_timeout))
            logging.loginfo('received target point.')

        else:
            if not CarryMyBullshit.traj_flipped:
                CarryMyBullshit.trajectory = np.flip(CarryMyBullshit.trajectory, axis=0)
                CarryMyBullshit.traj_flipped = True
            self.publish_trajectory()

        # %% real shit
        root_T_bf = god_map.world.compose_fk_expression(self.root, self.tip)
        root_T_odom = god_map.world.compose_fk_expression(self.root, self.odom)
        root_T_camera = god_map.world.compose_fk_expression(self.root, self.camera_link)
        root_P_bf = root_T_bf.to_position()

        min_left_violation1 = symbol_manager.get_symbol(self + '.closest_laser_left')
        min_right_violation1 = symbol_manager.get_symbol(self + '.closest_laser_right')
        closest_laser_reading1 = symbol_manager.get_symbol(self + '.closest_laser_reading')
        min_left_violation2 = symbol_manager.get_symbol(self + '.closest_laser_left_pc')
        min_right_violation2 = symbol_manager.get_symbol(self + '.closest_laser_right_pc')
        closest_laser_reading2 = symbol_manager.get_symbol(self + '.closest_laser_reading_pc')

        closest_laser_left = cas.min(min_left_violation1, min_left_violation2)
        closest_laser_right = cas.max(min_right_violation1, min_right_violation2)
        closest_laser_reading = cas.min(closest_laser_reading1, closest_laser_reading2)

        if not self.drive_back:
            last_target_age = symbol_manager.get_symbol(self + '.last_target_age')
            target_lost = ExpressionMonitor(name='target out of sight')
            self.add_monitor(target_lost)
            target_lost.expression = cas.greater_equal(last_target_age, self.target_age_threshold)

        next_x = symbol_manager.get_symbol(self + '.get_current_target()[\'next_x\']')
        next_y = symbol_manager.get_symbol(self + '.get_current_target()[\'next_y\']')
        closest_x = symbol_manager.get_symbol(self + '.get_current_target()[\'closest_x\']')
        closest_y = symbol_manager.get_symbol(self + '.get_current_target()[\'closest_y\']')
        # tangent_x = god_map.to_expr(self._get_identifier() + ['get_current_target', tuple(), 'tangent_x'])
        # tangent_y = god_map.to_expr(self._get_identifier() + ['get_current_target', tuple(), 'tangent_y'])
        clear_memo(self.get_current_target)
        root_P_goal_point = cas.Point3([next_x, next_y, 0])
        root_P_closest_point = cas.Point3([closest_x, closest_y, 0])
        # tangent = root_P_goal_point - root_P_closest_point
        # root_V_tangent = cas.Vector3([tangent.x, tangent.y, 0])
        tip_V_pointing_axis = cas.Vector3(self.tip_V_pointing_axis)

        if self.drive_back:
            map_P_human = root_P_goal_point
        else:
            map_P_human = cas.Point3(symbol_manager.get_expr(self + '.human_point', input_type_hint=PointStamped))
            map_P_human_projected = cas.Point3(map_P_human)
            map_P_human_projected.z = 0

        # %% orient to goal
        orient_to_goal = self.create_and_add_task('orient to goal')
        _, _, map_odom_angle = root_T_odom.to_rotation().to_rpy()
        odom_current_angle = self.odom_joint.yaw.get_symbol(Derivatives.position)
        map_current_angle = map_odom_angle + odom_current_angle
        if self.drive_back:
            root_V_tip_to_closest = root_P_bf - root_P_closest_point
            root_P_between_tip_and_closest = root_P_closest_point + root_V_tip_to_closest / 2
            root_V_goal_axis = root_P_goal_point - root_P_between_tip_and_closest
            root_V_goal_axis2 = cas.Vector3(root_V_goal_axis)
            root_V_goal_axis2.scale(1)
            map_P_human = map_P_human + root_V_goal_axis2 * 1.5
        else:
            root_V_goal_axis = map_P_human_projected - root_P_bf
        distance_to_human = cas.norm(root_V_goal_axis)
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_bf.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip
        root_V_goal_axis.vis_frame = self.tip
        map_goal_angle = cas.angle_between_vector(cas.Vector3([1, 0, 0]), root_V_goal_axis)
        map_goal_angle = cas.if_greater(root_V_goal_axis.y, 0, map_goal_angle, -map_goal_angle)
        map_angle_error = cas.shortest_angular_distance(map_current_angle, map_goal_angle)
        if self.drive_back:
            buffer = 0
        else:
            buffer = self.base_orientation_threshold
        ll = map_angle_error - buffer
        ul = map_angle_error + buffer
        orient_to_goal.add_inequality_constraint(reference_velocity=self.max_rotation_velocity,
                                                 lower_error=ll,
                                                 upper_error=ul,
                                                 weight=self.weight,
                                                 task_expression=map_current_angle,
                                                 name='rot')

        # %% look at goal
        camera_V_camera_axis = cas.Vector3(self.tip_V_camera_axis)
        root_V_camera_axis = root_T_camera.dot(camera_V_camera_axis)
        root_P_camera = root_T_camera.to_position()
        map_P_human.z = self.height_for_camera_target
        root_V_camera_goal_axis = map_P_human - root_P_camera
        root_V_camera_goal_axis.scale(1)
        look_at_target = self.create_and_add_task('look at target')
        if not self.drive_back:
            look_at_target.hold_condition = target_lost.get_state_expression()
        look_at_target.add_vector_goal_constraints(frame_V_current=root_V_camera_axis,
                                                   frame_V_goal=root_V_camera_goal_axis,
                                                   reference_velocity=self.max_rotation_velocity_head,
                                                   weight=self.weight,
                                                   name='move camera')

        # %% follow next point
        follow_next_point = self.create_and_add_task('follow next')
        root_V_camera_axis.vis_frame = self.camera_link
        root_V_camera_goal_axis.vis_frame = self.camera_link

        laser_violated = ExpressionMonitor(name='laser violated')
        self.add_monitor(laser_violated)
        laser_violated.expression = cas.less(closest_laser_reading, 0)
        if self.drive_back:
            oriented_towards_next = ExpressionMonitor(name='oriented towards next')
            oriented_towards_next.expression = cas.abs(map_angle_error) > self.base_orientation_threshold
            self.add_monitor(oriented_towards_next)

            follow_next_point.hold_condition = (laser_violated.get_state_expression()
                                                | oriented_towards_next.get_state_expression())
        else:
            target_too_close = ExpressionMonitor(name='target close')
            self.add_monitor(target_too_close)
            target_too_close.expression = cas.less_equal(distance_to_human, self.min_distance_to_target)

            follow_next_point.hold_condition = (laser_violated.get_state_expression() |
                                                (~target_lost.get_state_expression()
                                                 & target_too_close.get_state_expression()))
        follow_next_point.add_point_goal_constraints(frame_P_current=root_P_bf,
                                                     frame_P_goal=root_P_goal_point,
                                                     reference_velocity=self.max_translation_velocity,
                                                     weight=self.weight,
                                                     name='min dist to next')

        # %% keep the closest point in footprint radius
        stay_in_circle = self.create_and_add_task('in circle')
        buffer = self.traj_tracking_radius
        distance_to_closest_point = cas.norm(root_P_closest_point - root_P_bf)
        stay_in_circle.add_inequality_constraint(task_expression=distance_to_closest_point,
                                                 lower_error=-distance_to_closest_point - buffer,
                                                 upper_error=-distance_to_closest_point + buffer,
                                                 reference_velocity=self.max_translation_velocity,
                                                 weight=self.weight,
                                                 name='stay in circle')

        # %% laser avoidance
        if self.enable_laser_avoidance:
            laser_avoidance_task = self.create_and_add_task('laser avoidance')
            sideways_vel = (closest_laser_left + closest_laser_right)
            bf_V_laser_avoidance_direction = cas.Vector3([0, sideways_vel, 0])
            map_V_laser_avoidance_direction = root_T_bf.dot(bf_V_laser_avoidance_direction)
            map_V_laser_avoidance_direction.vis_frame = god_map.world.search_for_link_name(self.laser_frame)
            god_map.debug_expression_manager.add_debug_expression('base_V_laser_avoidance_direction',
                                                                  map_V_laser_avoidance_direction)
            odom_y_vel = self.odom_joint.y_vel.get_symbol(Derivatives.position)

            active = ExpressionMonitor(name='too far from path')
            self.add_monitor(active)
            active.expression = cas.greater(distance_to_closest_point, self.traj_tracking_radius)

            buffer = self.laser_avoidance_sideways_buffer / 2

            laser_avoidance_task.hold_condition = active.get_state_expression()
            laser_avoidance_task.add_inequality_constraint(reference_velocity=self.max_translation_velocity,
                                                           lower_error=sideways_vel - buffer,
                                                           upper_error=sideways_vel + buffer,
                                                           weight=WEIGHT_COLLISION_AVOIDANCE,
                                                           task_expression=odom_y_vel,
                                                           name='move sideways')

        if self.drive_back:
            first_traj_x = symbol_manager.get_symbol(self + '.get_first_traj_point()[0]')
            first_traj_y = symbol_manager.get_symbol(self + '.get_first_traj_point()[1]')
            first_point = cas.Point3([first_traj_x, first_traj_y, 0])
            goal_reached = ExpressionMonitor(name='goal reached?')
            self.add_monitor(goal_reached)
            goal_reached.expression = cas.euclidean_distance(first_point, root_P_bf) < self.traj_tracking_radius
            self.connect_end_condition_to_all_tasks(goal_reached.get_state_expression())
            end = EndMotion(name='done')
            end.start_condition = goal_reached.get_state_expression()
            self.add_monitor(end)
        self.connect_start_condition_to_all_tasks(start_condition)
        self.connect_hold_condition_to_all_tasks(hold_condition)
        self.connect_end_condition_to_all_tasks(end_condition)

    def clean_up(self):
        if CarryMyBullshit.target_sub is not None:
            CarryMyBullshit.target_sub.unregister()
            CarryMyBullshit.target_sub = None
        if CarryMyBullshit.laser_sub is not None:
            CarryMyBullshit.laser_sub.unregister()
            CarryMyBullshit.laser_sub = None
        if CarryMyBullshit.point_cloud_laser_sub is not None:
            CarryMyBullshit.point_cloud_laser_sub.unregister()
            CarryMyBullshit.point_cloud_laser_sub = None

    def init_laser_stuff(self, laser_scan: LaserScan):
        thresholds = []
        if len(laser_scan.ranges) % 2 == 0:
            print('laser range is even')
            angles = np.arange(laser_scan.angle_min,
                               laser_scan.angle_max,
                               laser_scan.angle_increment)[:-1]
        else:
            angles = np.arange(laser_scan.angle_min,
                               laser_scan.angle_max,
                               laser_scan.angle_increment)
        for angle in angles:
            if angle < 0:
                y = -self.laser_distance_threshold_width
                length = y / np.sin((angle))
                x = np.cos(angle) * length
                thresholds.append((x, y, length, angle))
            else:
                y = self.laser_distance_threshold_width
                length = y / np.sin((angle))
                x = np.cos(angle) * length
                thresholds.append((x, y, length, angle))
            if length > self.laser_distance_threshold:
                length = self.laser_distance_threshold
                x = np.cos(angle) * length
                y = np.sin(angle) * length
                thresholds[-1] = (x, y, length, angle)
        thresholds = np.array(thresholds)
        assert len(thresholds) == len(laser_scan.ranges)
        return thresholds

    def muddle_laser_scan(self, scan: LaserScan, thresholds: np.ndarray):
        data = np.array(scan.ranges)
        xs = np.cos(thresholds[:, 3]) * data
        ys = np.sin(thresholds[:, 3]) * data
        violations = data < thresholds[:, 2]
        xs_error = xs - thresholds[:, 0]
        half = int(data.shape[0] / 2)
        # x_positive = np.where(thresholds[:, 0] > 0)[0]
        x_below_laser_avoidance_threshold1 = thresholds[:, -1] > self.laser_avoidance_angle_cutout
        x_below_laser_avoidance_threshold2 = thresholds[:, -1] < -self.laser_avoidance_angle_cutout
        x_below_laser_avoidance_threshold = x_below_laser_avoidance_threshold1 | x_below_laser_avoidance_threshold2
        y_filter = x_below_laser_avoidance_threshold & violations
        closest_laser_right = ys[:half][y_filter[:half]]
        closest_laser_left = ys[half:][y_filter[half:]]

        x_positive = np.where(np.invert(x_below_laser_avoidance_threshold))[0]
        x_start = x_positive[0]
        x_end = x_positive[-1]

        front_violation = xs_error[x_start:x_end][violations[x_start:x_end]]
        if len(closest_laser_left) > 0:
            closest_laser_left = min(closest_laser_left)
        else:
            closest_laser_left = self.laser_distance_threshold_width
        if len(closest_laser_right) > 0:
            closest_laser_right = max(closest_laser_right)
        else:
            closest_laser_right = -self.laser_distance_threshold_width
        if len(front_violation) > 0:
            closest_laser_reading = min(front_violation)
        else:
            closest_laser_reading = 0
        return closest_laser_reading, closest_laser_left, closest_laser_right

    def laser_cb(self, scan: LaserScan):
        self.last_scan = scan
        if self.thresholds is None:
            self.thresholds = self.init_laser_stuff(scan)
            self.publish_laser_thresholds()
        self.closest_laser_reading, self.closest_laser_left, self.closest_laser_right = self.muddle_laser_scan(scan,
                                                                                                               self.thresholds)

    def point_cloud_laser_cb(self, scan: LaserScan):
        self.last_scan_pc = scan
        if self.thresholds_pc is None:
            self.thresholds_pc = self.init_laser_stuff(scan)
        self.closest_laser_reading_pc, self.closest_laser_left_pc, self.closest_laser_right_pc = self.muddle_laser_scan(
            scan, self.thresholds_pc)

    def get_current_point(self) -> np.ndarray:
        root_T_tip = god_map.world.compute_fk_np(self.root, self.tip)
        x = root_T_tip[0, 3]
        y = root_T_tip[1, 3]
        return np.array([x, y])

    def get_first_traj_point(self) -> np.ndarray:
        return CarryMyBullshit.traj_data[0]

    @memoize_with_counter(4)
    def get_current_target(self) -> Dict[str, float]:
        self.check_laser_scan_age()
        traj = CarryMyBullshit.trajectory.copy()
        current_point = self.get_current_point()
        error = traj - current_point
        distances = np.linalg.norm(error, axis=1)
        # cut off old points
        in_radius = np.where(distances < self.traj_tracking_radius)[0]
        if len(in_radius) > 0:
            next_idx = max(in_radius)
            offset = max(0, next_idx - self.max_temp_distance)
            closest_idx = np.argmin(distances[offset:]) + offset
        else:
            next_idx = closest_idx = np.argmin(distances)
        # CarryMyBullshit.traj_data = CarryMyBullshit.traj_data[closest_idx:]
        if not self.drive_back:
            self.last_target_age = rospy.get_rostime().to_sec() - self.human_point.header.stamp.to_sec()
            if self.last_target_age > self.target_age_exception_threshold:
                raise_to_blackboard(
                    ExecutionException(f'lost target for longer than {self.target_age_exception_threshold}s'))
        else:
            if closest_idx == CarryMyBullshit.trajectory.shape[0] - 1:
                self.end_of_traj_reached = True
        result = {
            'next_x': traj[next_idx, 0],
            'next_y': traj[next_idx, 1],
            'closest_x': traj[closest_idx, 0],
            'closest_y': traj[closest_idx, 1],
        }
        return result

    def check_laser_scan_age(self):
        current_time = rospy.get_rostime().to_sec()
        base_laser_age = current_time - self.last_scan.header.stamp.to_sec()
        if base_laser_age > self.laser_scan_age_threshold:
            logging.logwarn(f'last base laser scan is too old: {base_laser_age}')
            self.closest_laser_left = self.laser_distance_threshold_width
            self.closest_laser_right = -self.laser_distance_threshold_width
            self.closest_laser_reading = 0
        point_cloud_laser_age = current_time - self.last_scan_pc.header.stamp.to_sec()
        if point_cloud_laser_age > self.laser_scan_age_threshold and CarryMyBullshit.point_cloud_laser_sub is not None:
            logging.logwarn(f'last point cloud laser scan is too old: {point_cloud_laser_age}')
            self.closest_laser_left_pc = self.laser_distance_threshold_width
            self.closest_laser_right_pc = -self.laser_distance_threshold_width
            self.closest_laser_reading_pc = 0

    def is_done(self):
        return self.end_of_traj_reached

    def publish_trajectory(self):
        ms = MarkerArray()
        m_line = Marker()
        m_line.action = m_line.ADD
        m_line.ns = 'traj'
        m_line.id = 1
        m_line.type = m_line.LINE_STRIP
        m_line.header.frame_id = str(god_map.world.root_link_name)
        m_line.scale.x = 0.05
        m_line.color.a = 1
        m_line.color.r = 1
        try:
            for item in CarryMyBullshit.trajectory:
                p = Point()
                p.x = item[0]
                p.y = item[1]
                m_line.points.append(p)
            ms.markers.append(m_line)
        except Exception as e:
            logging.logwarn('failed to create traj marker')
        self.pub.publish(ms)

    def publish_laser_thresholds(self):
        ms = MarkerArray()
        m_line = Marker()
        m_line.action = m_line.ADD
        m_line.ns = 'laser_thresholds'
        m_line.id = 1332
        m_line.type = m_line.LINE_STRIP
        m_line.header.frame_id = self.laser_frame
        m_line.scale.x = 0.05
        m_line.color.a = 1
        m_line.color.r = 0.5
        m_line.color.b = 1
        m_line.frame_locked = True
        for item in self.thresholds:
            p = Point()
            p.x = item[0]
            p.y = item[1]
            m_line.points.append(p)
        ms.markers.append(m_line)
        square = Marker()
        square.action = m_line.ADD
        square.ns = 'laser_avoidance_angle_cutout'
        square.id = 1333
        square.type = m_line.LINE_STRIP
        square.header.frame_id = self.laser_frame
        # p = Point()
        # p.x = self.thresholds[0, 0]
        # p.y = self.thresholds[0, 1]
        # square.points.append(p)
        p = Point()
        idx = np.where(self.thresholds[:, -1] < -self.laser_avoidance_angle_cutout)[0][-1]
        p.x = self.thresholds[idx, 0]
        p.y = self.thresholds[idx, 1]
        square.points.append(p)
        p = Point()
        square.points.append(p)
        p = Point()
        idx = np.where(self.thresholds[:, -1] > self.laser_avoidance_angle_cutout)[0][0]
        p.x = self.thresholds[idx, 0]
        p.y = self.thresholds[idx, 1]
        square.points.append(p)
        # p = Point()
        # p.x = self.thresholds[-1, 0]
        # p.y = self.thresholds[-1, 1]
        # square.points.append(p)
        square.scale.x = 0.05
        square.color.a = 1
        square.color.r = 0.5
        square.color.b = 1
        square.frame_locked = True
        ms.markers.append(square)
        self.pub.publish(ms)

    def publish_tracking_radius(self):
        ms = MarkerArray()
        m_line = Marker()
        m_line.action = m_line.ADD
        m_line.ns = 'traj_tracking_radius'
        m_line.id = 1332
        m_line.type = m_line.CYLINDER
        m_line.header.frame_id = str(self.tip.short_name)
        m_line.scale.x = self.traj_tracking_radius * 2
        m_line.scale.y = self.traj_tracking_radius * 2
        m_line.scale.z = 0.05
        m_line.color.a = 1
        m_line.color.b = 1
        m_line.pose.orientation.w = 1
        m_line.frame_locked = True
        ms.markers.append(m_line)
        self.pub.publish(ms)

    def publish_distance_to_target(self):
        ms = MarkerArray()
        m_line = Marker()
        m_line.action = m_line.ADD
        m_line.ns = 'distance_to_target_stop_threshold'
        m_line.id = 1332
        m_line.type = m_line.CYLINDER
        m_line.header.frame_id = str(self.tip.short_name)
        m_line.scale.x = self.min_distance_to_target * 2
        m_line.scale.y = self.min_distance_to_target * 2
        m_line.scale.z = 0.01
        m_line.color.a = 0.5
        m_line.color.g = 1
        m_line.pose.orientation.w = 1
        m_line.frame_locked = True
        ms.markers.append(m_line)
        self.pub.publish(ms)

    def target_cb(self, point: PointStamped):
        try:
            current_point = np.array([point.point.x, point.point.y])
            last_point = CarryMyBullshit.traj_data[-1]
            error_vector = current_point - last_point
            distance = np.linalg.norm(error_vector)
            if distance < self.interpolation_step_size * 2:
                CarryMyBullshit.traj_data[-1] = 0.5 * CarryMyBullshit.traj_data[-1] + 0.5 * current_point
            else:
                error_vector /= distance
                ranges = np.arange(self.interpolation_step_size, distance, self.interpolation_step_size)
                interpolated_distance = distance / len(ranges)
                for i, dt in enumerate(ranges):
                    interpolated_point = last_point + error_vector * interpolated_distance * (i + 1)
                    CarryMyBullshit.traj_data.append(interpolated_point)
                CarryMyBullshit.traj_data.append(current_point)

            CarryMyBullshit.trajectory = np.array(CarryMyBullshit.traj_data)
            self.human_point = point
        except Exception as e:
            logging.logwarn(f'rejected new target because: {e}')
        self.publish_trajectory()


class BaseTrajFollowerPR2(BaseTrajFollower):
    joint: OmniDrivePR22

    def make_constraints(self):
        constraints = super().make_constraints()
        return constraints

    @profile
    def add_trans_constraints(self):
        lb_yaw1 = []
        lb_forward = []
        god_map.world.state[self.joint.yaw1_vel.name].position = 0
        map_T_current = god_map.world.compose_fk_expression(god_map.world.root_link_name, self.base_footprint_link)
        map_P_current = map_T_current.to_position()
        self.add_debug_expr(f'map_P_current.x', map_P_current.x)
        self.add_debug_expr('time', god_map.to_expr(identifier.time))
        for t in range(god_map.qp_controller_config.prediction_horizon - 2):
            trajectory_time_in_s = t * god_map.qp_controller_config.sample_period
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
            forward = self.current_traj_point(self.joint.forward_vel.name,
                                              t * god_map.qp_controller_config.sample_period,
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
        x = cas.cos(bf_yaw)
        y = cas.sin(bf_yaw)
        v = cas.Vector3([x, y, 0])
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
