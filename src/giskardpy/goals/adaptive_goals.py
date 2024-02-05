from __future__ import division

from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from giskardpy import casadi_wrapper as cas
from giskardpy.tasks.task import Task, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE \
    , Monitor
from giskardpy.goals.goal import Goal
import rospy
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.data_types import PrefixName, Derivatives
import math
from typing import Optional, List
import numpy as np
from giskardpy.monitors.monitors import ExpressionMonitor


# Todo: instead of relying on predefined poses model the motion as relations between the objects
# Todo: take velocity constraints more into account
class PouringAdaptiveTilt(Goal):
    def __init__(self, root, tip, pouring_pose: PoseStamped, tilt_angle: float, tilt_axis: Vector3Stamped,
                 use_local_min=False, max_vel=0.3, weight=WEIGHT_COLLISION_AVOIDANCE, pre_tilt=False,
                 name: Optional[str] = None, with_feedback=True,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name)
        self.action_sub = rospy.Subscriber('/reasoner/concluded_behaviors', String, self.callback)
        self.root_link = god_map.world.search_for_link_name(root, None)
        self.tip_link = god_map.world.search_for_link_name(tip, None)
        self.tilt_axis = tilt_axis
        self.tilt_axis.header.frame_id = god_map.world.search_for_link_name(tilt_axis.header.frame_id, None)
        self.tilt_axis = god_map.world.transform_msg(self.tip_link, tilt_axis)
        self.tilt_axis_root = god_map.world.transform_msg(self.root_link, tilt_axis)
        self.pouring_pose = pouring_pose
        self.pouring_pose.header.frame_id = god_map.world.search_for_link_name(pouring_pose.header.frame_id, None)
        self.root = root
        self.tip = tip
        self.max_vel = max_vel
        self.weight = weight
        self.use_local_min = use_local_min

        self.action_string = ''
        self.tilt_angle = tilt_angle
        self.forward = False
        self.backward = False
        self.move_x = False
        self.move_x_back = False
        self.move_y = False
        self.move_y_back = False
        self.stop = False
        self.stop_counter = 0

        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        tip_V_tilt_axis = cas.Vector3(self.tilt_axis.vector)
        root_V_tilt_axis_fk = root_T_tip.to_rotation().dot(tip_V_tilt_axis)

        # Gram-Schmidt process to obtain two vector orthogonal to the tilt axis in root coordinates
        # https://stackoverflow.com/questions/33658620/generating-two-orthogonal-vectors-that-are-orthogonal-to-a-particular-direction
        k = np.array([self.tilt_axis_root.vector.x, self.tilt_axis_root.vector.y, self.tilt_axis_root.vector.z])
        first = np.random.randn(3)
        first -= first.dot(k) * k / np.linalg.norm(k) ** 2
        first /= np.linalg.norm(first)
        second = np.cross(k, first)
        root_V_first = cas.Vector3(first)
        root_V_second = cas.Vector3(second)

        self.pos_task = self.create_and_add_task('prePosition')
        root_T_goal = god_map.world.transform_msg(self.root_link, self.pouring_pose)
        root_P_goal = cas.Point3(root_T_goal.pose.position)
        self.pos_task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(), frame_P_goal=root_P_goal,
                                                 reference_velocity=self.max_vel, weight=self.weight)

        rot_task = self.create_and_add_task('preRotation')
        if pre_tilt:
            tip_R_tip = cas.RotationMatrix().from_axis_angle(tip_V_tilt_axis, self.tilt_angle / 2)
            root_R_tip_desired_pre = cas.RotationMatrix(root_T_goal.pose.orientation).dot(tip_R_tip)
        else:
            root_R_tip_desired_pre = cas.RotationMatrix(root_T_goal.pose.orientation)
        rot_task.add_rotation_goal_constraints(frame_R_current=root_T_tip.to_rotation(),
                                               frame_R_goal=root_R_tip_desired_pre,
                                               current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(
                                                   self.tip_link, self.root_link),
                                               reference_velocity=self.max_vel,
                                               weight=self.weight)

        pos_monitor = ExpressionMonitor(name='isPositioned', stay_true=True)
        self.add_monitor(pos_monitor)
        pos_monitor.expression = (cas.less(cas.euclidean_distance(root_T_tip.to_position(), root_P_goal), 0.005))
        # pos_task.add_to_end_monitor(pos_monitor)
        rot_task.end_condition = pos_monitor

        # Begin nominal rotation goal
        root_R_start = cas.RotationMatrix(root_T_goal.pose.orientation)
        root_R_tip = root_T_tip.to_rotation()

        # Rotate tip about angle around z axis
        tip_R_tip = cas.RotationMatrix().from_axis_angle(tip_V_tilt_axis, self.tilt_angle)
        root_R_tip_desired = root_R_start.dot(tip_R_tip)

        nominal_task = self.create_and_add_task('TiltToPlanned')
        # add constraint that achieves the desired rotation
        nominal_task.add_rotation_goal_constraints(frame_R_current=root_T_tip.to_rotation(),
                                                   frame_R_goal=root_R_tip_desired,
                                                   current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(
                                                       self.tip_link, self.root_link),
                                                   reference_velocity=self.max_vel,
                                                   weight=self.weight)
        # add constraints that keep the rotation axis in place. It has to be orthogonal to two vectors that were
        # created orthogonal to the original tilt axis
        root_V_tip_z = cas.Vector3(root_V_tilt_axis_fk)
        exp = root_V_tip_z.dot(root_V_first)
        nominal_task.add_equality_constraint(reference_velocity=self.max_vel,
                                             equality_bound=0 - exp,
                                             weight=self.weight,
                                             task_expression=exp)
        root_V_tip_y = cas.Vector3(root_V_tilt_axis_fk)
        exp2 = root_V_tip_y.dot(root_V_second)
        nominal_task.add_equality_constraint(reference_velocity=self.max_vel,
                                             equality_bound=0 - exp2,
                                             weight=self.weight,
                                             task_expression=exp2)
        # monitor for the nominal task
        nominal_monitor = ExpressionMonitor(name='isTilted', stay_true=True)
        # monitor to observe the angle and stop the nominal task
        self.add_monitor(nominal_monitor)
        nominal_error = cas.rotational_error(root_R_tip, root_R_tip_desired)
        nominal_monitor.expression = (cas.less(nominal_error, 0.1))
        nominal_task.start_condition = pos_monitor
        nominal_task.end_condition = nominal_monitor
        self.pos_task.end_condition = nominal_monitor

        # add the adaptive part of the goal or not depending on the with_feedback flag
        if not with_feedback:
            return

        adaptive_task = self.create_and_add_task('adaptTiltFeedback')
        is_forward = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].forward')
        is_backward = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].backward')
        angle = cas.rotational_error(root_R_tip, root_R_tip_desired_pre)
        # TODO: how can the speed of a rotation be controlled from the outside?
        if self.tilt_angle < 0:
            angle_a = -0.03 * is_forward + 1 * is_backward
            stop_to_large = cas.logic_any(
                cas.Expression([cas.if_greater(angle, 3, 0, 1), cas.if_greater(angle_a, 0, 1, 0)]))
            stop_to_small = cas.if_less(angle, 0.1, 0, 1)
        else:
            angle_a = 0.03 * is_forward - 1 * is_backward
            stop_to_large = cas.logic_any(
                cas.Expression([cas.if_greater(angle, 3, 0, 1), cas.if_less(angle_a, 0, 1, 0)]))
            stop_to_small = cas.logic_any(
                cas.Expression([cas.if_less(angle, 0.1, 0, 1), cas.if_greater(angle_a, 0, 1, 0)]))
        tip_R_tip_a = cas.RotationMatrix().from_axis_angle(tip_V_tilt_axis, angle_a)
        root_R_tip_desired_a = root_R_tip.dot(tip_R_tip_a)
        # TODO: look into slerp again. Is that necessary here?
        # TODO: Try this with quaternions instead!
        adaptive_task.add_rotation_goal_constraints(frame_R_current=root_R_tip,
                                                    frame_R_goal=root_R_tip_desired_a,
                                                    current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(
                                                        self.tip_link, self.root_link),
                                                    reference_velocity=self.max_vel,
                                                    weight=self.weight * stop_to_large * stop_to_small)
        # adaptive_task.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 9,
        #                                              equality_bounds=[root_R_tip_desired_a[0, 0] - root_R_tip[0, 0],
        #                                                               root_R_tip_desired_a[0, 1] - root_R_tip[0, 1],
        #                                                               root_R_tip_desired_a[0, 2] - root_R_tip[0, 2],
        #                                                               root_R_tip_desired_a[1, 0] - root_R_tip[1, 0],
        #                                                               root_R_tip_desired_a[1, 1] - root_R_tip[1, 1],
        #                                                               root_R_tip_desired_a[1, 2] - root_R_tip[1, 2],
        #                                                               root_R_tip_desired_a[2, 0] - root_R_tip[2, 0],
        #                                                               root_R_tip_desired_a[2, 1] - root_R_tip[2, 1],
        #                                                               root_R_tip_desired_a[2, 2] - root_R_tip[2, 2]
        #                                                               ],
        #                                              weights=[self.weight
        #                                                       * stop_to_large
        #                                                       * stop_to_small] * 9,
        #                                              task_expression=[root_R_tip[0, 0],
        #                                                               root_R_tip[0, 1],
        #                                                               root_R_tip[0, 2],
        #                                                               root_R_tip[1, 0],
        #                                                               root_R_tip[1, 1],
        #                                                               root_R_tip[1, 2],
        #                                                               root_R_tip[2, 0],
        #                                                               root_R_tip[2, 1],
        #                                                               root_R_tip[2, 2]],
        #                                              names=['tipr1a', 'tipr2a', 'tipr3a', 'tipr4a',
        #                                                     'tipr5a',
        #                                                     'tipr6a', 'tipr7a', 'tipr8a', 'tipr9a'
        #                                                     ])
        adaptive_task.start_condition = nominal_monitor
        adaptive_task.add_equality_constraint(reference_velocity=self.max_vel,
                                              equality_bound=0 - exp,
                                              weight=self.weight,
                                              task_expression=exp)
        # TODO: why do i need this second constraint for the pot frame not to rotate around  the z-axis of world?
        #       The rotation matrix constraint above is probably underdefined. test if expanding that constraint helps.
        #       But prob not ideal as the goal is relative to the current rotation, therefore it can shift.
        #       The two dot product exps are good anchor points.
        adaptive_task.add_equality_constraint(reference_velocity=self.max_vel,
                                              equality_bound=0 - exp2,
                                              weight=self.weight,
                                              task_expression=exp2)

        # Todo: make it smooth and nice to look at, tilt back when adapting the position
        adapt_pos_task = self.create_and_add_task('adaptPosition')
        adapt_pos_task.start_condition = nominal_monitor
        is_x = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_x')
        is_x_back = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_x_back')
        is_y = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_y')
        is_y_back = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_y_back')
        root_V_adapt = cas.Vector3([0.02 * is_x - 0.02 * is_x_back,
                                    0.01 * is_y - 0.01 * is_y_back,
                                    0
                                    ])
        adapt_pos_task.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                                      equality_bounds=root_V_adapt[:3],
                                                      weights=[self.weight] * 3,
                                                      task_expression=root_T_tip.to_position()[:3],
                                                      names=['aposx', 'aposy', 'aposz'])
        # adapt_pos_task.add_velocity_constraint(lower_velocity_limit=root_V_adapt[0],
        #                                        upper_velocity_limit=root_V_adapt[0],
        #                                        weight=self.weight,
        #                                        task_expression=root_T_tip.to_position()[0],
        #                                        velocity_limit=root_V_adapt[0],
        #                                        name_suffix='second')
        # adapt_pos_task.add_velocity_constraint(lower_velocity_limit=-0.2,
        #                                        upper_velocity_limit=-0.2,
        #                                        weight=self.weight,
        #                                        task_expression=root_T_tip.to_position()[1],
        #                                        velocity_limit=0.2)

        external_end_monitor = ExpressionMonitor(name='isFnished')
        self.add_monitor(external_end_monitor)
        external_end_monitor.expression = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].stop')
        adaptive_task.end_condition = external_end_monitor
        adapt_pos_task.end_condition = external_end_monitor
        # self.pos_task.add_to_end_monitor(external_end_monitor)

    def callback(self, action_string: String):
        self.action_string = action_string.data
        if 'increase' in action_string.data and 'decrease' in action_string.data:
            self.forward = False
            self.backward = True
        elif 'decrease' in action_string.data:
            self.forward = False
            self.backward = True
        elif 'increase' in action_string.data:
            self.backward = False
            self.forward = True
        else:
            self.forward = False
            self.backward = False

        self.move_x = False
        self.move_x_back = False
        self.move_y = False
        self.move_y_back = False
        if 'moveForward' in action_string.data:
            self.move_x = True
        elif 'moveBackward' in action_string.data:
            self.move_x_back = True
        elif 'moveLeft' in action_string.data:
            self.move_y = True
        if 'moveRight' in action_string.data:
            self.move_y_back = True

        if '{}' in action_string.data:
            self.stop_counter += 1
            if self.stop_counter > 10:
                self.stop = 1
        else:
            self.stop_counter = 0


class PouringAdaptiveTiltScraping(Goal):
    def __init__(self, root, tip, pouring_pose: PoseStamped, tilt_angle: float, tilt_axis: Vector3Stamped,
                 use_local_min=False, max_vel=0.3, weight=WEIGHT_COLLISION_AVOIDANCE, pre_tilt=False,
                 name: Optional[str] = None, with_feedback=True, scrape_tip=None, scrape_pose: PoseStamped = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name)
        self.action_sub = rospy.Subscriber('/reasoner/concluded_behaviors', String, self.callback)
        self.root_link = god_map.world.search_for_link_name(root, None)
        self.tip_link = god_map.world.search_for_link_name(tip, None)
        self.scrape_tip = god_map.world.search_for_link_name(scrape_tip, None)
        self.tilt_axis = tilt_axis
        self.tilt_axis.header.frame_id = god_map.world.search_for_link_name(tilt_axis.header.frame_id, None)
        self.tilt_axis = god_map.world.transform_msg(self.tip_link, tilt_axis)
        self.tilt_axis_root = god_map.world.transform_msg(self.root_link, tilt_axis)
        self.pouring_pose = pouring_pose
        self.pouring_pose.header.frame_id = god_map.world.search_for_link_name(pouring_pose.header.frame_id, None)
        self.root = root
        self.tip = tip
        self.max_vel = max_vel
        self.weight = weight
        self.use_local_min = use_local_min

        self.action_string = ''
        self.tilt_angle = tilt_angle
        self.forward = False
        self.backward = False
        self.move_x = False
        self.move_x_back = False
        self.move_y = False
        self.move_y_back = False
        self.stop = False
        self.stop_counter = 0

        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        tip_V_tilt_axis = cas.Vector3(self.tilt_axis.vector)
        root_V_tilt_axis_fk = root_T_tip.to_rotation().dot(tip_V_tilt_axis)

        # Gram-Schmidt process to obtain two vector orthogonal to the tilt axis in root coordinates
        # https://stackoverflow.com/questions/33658620/generating-two-orthogonal-vectors-that-are-orthogonal-to-a-particular-direction
        k = np.array([self.tilt_axis_root.vector.x, self.tilt_axis_root.vector.y, self.tilt_axis_root.vector.z])
        first = np.random.randn(3)
        first -= first.dot(k) * k / np.linalg.norm(k) ** 2
        first /= np.linalg.norm(first)
        second = np.cross(k, first)
        root_V_first = cas.Vector3(first)
        root_V_second = cas.Vector3(second)

        self.pos_task = self.create_and_add_task('prePosition')
        root_T_goal = god_map.world.transform_msg(self.root_link, self.pouring_pose)
        root_P_goal = cas.Point3(root_T_goal.pose.position)
        self.pos_task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(), frame_P_goal=root_P_goal,
                                                 reference_velocity=self.max_vel, weight=self.weight)
        # position the scrape tip as well
        tip_T_scrape_tip = god_map.world.compose_fk_expression(self.tip_link, self.scrape_tip)
        tip_P_goal = cas.Point3(scrape_pose.pose.position)
        self.pos_task.add_point_goal_constraints(frame_P_current=tip_T_scrape_tip.to_position(),
                                                 frame_P_goal=tip_P_goal, reference_velocity=max_vel, weight=weight,
                                                 name='scraspe')
        tip_R_goal = cas.RotationMatrix(scrape_pose.pose.orientation)
        self.pos_task.add_rotation_goal_constraints(frame_R_current=tip_T_scrape_tip.to_rotation(),
                                                    frame_R_goal=tip_R_goal,
                                                    current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(
                                                        self.scrape_tip, self.tip_link),
                                                    reference_velocity=self.max_vel,
                                                    weight=self.weight, name='scrape')
        rot_task = self.create_and_add_task('preRotation')
        if pre_tilt:
            tip_R_tip = cas.RotationMatrix().from_axis_angle(tip_V_tilt_axis, self.tilt_angle / 2)
            root_R_tip_desired_pre = cas.RotationMatrix(root_T_goal.pose.orientation).dot(tip_R_tip)
        else:
            root_R_tip_desired_pre = cas.RotationMatrix(root_T_goal.pose.orientation)
        rot_task.add_rotation_goal_constraints(frame_R_current=root_T_tip.to_rotation(),
                                               frame_R_goal=root_R_tip_desired_pre,
                                               current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(
                                                   self.tip_link, self.root_link),
                                               reference_velocity=self.max_vel,
                                               weight=self.weight)

        pos_monitor = ExpressionMonitor(name='isPositioned', stay_true=True)
        self.add_monitor(pos_monitor)
        pos_monitor.expression = (cas.less(cas.euclidean_distance(root_T_tip.to_position(), root_P_goal), 0.005))
        rot_task.end_condition = pos_monitor

        # Begin nominal rotation goal
        root_R_start = cas.RotationMatrix(root_T_goal.pose.orientation)
        root_R_tip = root_T_tip.to_rotation()

        # Rotate tip about angle around z axis
        tip_R_tip = cas.RotationMatrix().from_axis_angle(tip_V_tilt_axis, self.tilt_angle)
        root_R_tip_desired = root_R_start.dot(tip_R_tip)

        nominal_task = self.create_and_add_task('TiltToPlanned')
        # add constraint that achieves the desired rotation
        nominal_task.add_rotation_goal_constraints(frame_R_current=root_T_tip.to_rotation(),
                                                   frame_R_goal=root_R_tip_desired,
                                                   current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(
                                                       self.tip_link, self.root_link),
                                                   reference_velocity=self.max_vel,
                                                   weight=self.weight)
        # add constraints that keep the rotation axis in place. It has to be orthogonal to two vectors that were
        # created orthogonal to the original tilt axis
        root_V_tip_z = cas.Vector3(root_V_tilt_axis_fk)
        exp = root_V_tip_z.dot(root_V_first)
        nominal_task.add_equality_constraint(reference_velocity=self.max_vel,
                                             equality_bound=0 - exp,
                                             weight=self.weight,
                                             task_expression=exp)
        root_V_tip_y = cas.Vector3(root_V_tilt_axis_fk)
        exp2 = root_V_tip_y.dot(root_V_second)
        nominal_task.add_equality_constraint(reference_velocity=self.max_vel,
                                             equality_bound=0 - exp2,
                                             weight=self.weight,
                                             task_expression=exp2)
        # monitor for the nominal task
        nominal_monitor = ExpressionMonitor(name='isTilted', stay_true=True)
        # monitor to observe the angle and stop the nominal task
        self.add_monitor(nominal_monitor)
        nominal_error = cas.rotational_error(root_R_tip, root_R_tip_desired)
        nominal_monitor.expression = (cas.less(nominal_error, 0.1))
        nominal_task.start_condition = pos_monitor
        # nominal_task.end_condition = nominal_monitor
        self.pos_task.end_condition = nominal_monitor

        # add the adaptive part of the goal or not depending on the with_feedback flag
        if not with_feedback:
            return
        ###############################################################################
        # TODO: replace tilt with scraping
        adaptive_task = self.create_and_add_task('adaptTiltFeedback')
        is_forward = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].forward')
        is_backward = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].backward')

        adaptive_task.start_condition = nominal_monitor
        tip_T_scrape_tip = god_map.world.compose_fk_expression(self.tip_link, self.scrape_tip)
        tip_P_goal = cas.Point3(scrape_pose.pose.position)
        bounds = - cas.Vector3([is_backward * 0.01 - is_forward * 0.01, 0, 0])
        adaptive_task.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                                     equality_bounds=bounds[:3],
                                                     weights=[self.weight] * 3,
                                                     task_expression=tip_T_scrape_tip.to_position()[:3],
                                                     names=['scrapex', 'scrapey', 'scrapez'])
        tip_R_goal = cas.RotationMatrix(scrape_pose.pose.orientation)
        adaptive_task.add_rotation_goal_constraints(frame_R_current=tip_T_scrape_tip.to_rotation(),
                                                    frame_R_goal=tip_R_goal,
                                                    current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(
                                                        self.scrape_tip, self.tip_link),
                                                    reference_velocity=self.max_vel,
                                                    weight=self.weight)

        ##############################################################################
        adapt_pos_task = self.create_and_add_task('adaptPosition')
        adapt_pos_task.start_condition = nominal_monitor
        is_x = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_x')
        is_x_back = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_x_back')
        is_y = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_y')
        is_y_back = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_y_back')
        root_V_adapt = cas.Vector3([0.02 * is_x - 0.02 * is_x_back,
                                    0.01 * is_y - 0.01 * is_y_back,
                                    0
                                    ])
        adapt_pos_task.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                                      equality_bounds=root_V_adapt[:3],
                                                      weights=[self.weight] * 3,
                                                      task_expression=root_T_tip.to_position()[:3],
                                                      names=['aposx', 'aposy', 'aposz'])

        external_end_monitor = ExpressionMonitor(name='isFnished')
        self.add_monitor(external_end_monitor)
        external_end_monitor.expression = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].stop')
        adaptive_task.end_condition = external_end_monitor
        adapt_pos_task.end_condition = external_end_monitor
        nominal_task.end_condition = external_end_monitor

    def callback(self, action_string: String):
        self.action_string = action_string.data
        if 'increase' in action_string.data and 'decrease' in action_string.data:
            self.forward = False
            self.backward = True
        elif 'decrease' in action_string.data:
            self.forward = False
            self.backward = True
        elif 'increase' in action_string.data:
            self.backward = False
            self.forward = True
        else:
            self.forward = False
            self.backward = False

        self.move_x = False
        self.move_x_back = False
        self.move_y = False
        self.move_y_back = False
        if 'moveForward' in action_string.data:
            self.move_x = True
        elif 'moveBackward' in action_string.data:
            self.move_x_back = True
        elif 'moveLeft' in action_string.data:
            self.move_y = True
        if 'moveRight' in action_string.data:
            self.move_y_back = True

        if '{}' in action_string.data:
            self.stop_counter += 1
            if self.stop_counter > 10:
                self.stop = 1
        else:
            self.stop_counter = 0


class CloseGripper(Goal):
    def __init__(self, effort: int = -180, pub_topic='hsrb4s/hand_motor_joint_velocity_controller/command',
                 joint_state_topic='hsrb4s/joint_states', alibi_joint_name='hand_motor_joint', joint_group=None,
                 velocity_threshold=0.1, effort_threshold=-1, as_open=False, name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name)
        self.pub = rospy.Publisher(pub_topic, Float64, queue_size=1)
        self.effort = 0
        self.velocity_threshold = velocity_threshold
        self.alibi_joint_name = alibi_joint_name
        rospy.Subscriber(joint_state_topic, JointState, self.callback)

        self.msg = Float64()
        self.msg.data = effort

        self.task = self.create_and_add_task(task_name='gripper')
        joint_name = god_map.world.search_for_joint_name(alibi_joint_name, joint_group)
        joint = god_map.world.joints[joint_name]
        self.task.add_equality_constraint(reference_velocity=0.2,
                                          equality_bound=0 - joint.get_symbol(Derivatives.position),
                                          weight=WEIGHT_BELOW_CA,
                                          task_expression=joint.get_symbol(Derivatives.position))
        monitor = ExpressionMonitor(name=f'efforMoniotor{name}')
        effort = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].effort')
        if as_open:
            monitor.expression = cas.if_greater(effort, effort_threshold, 1, 0)
        else:
            monitor.expression = cas.if_less(effort, effort_threshold, 1, 0)
        self.add_monitor(monitor)
        self.task.end_condition = monitor
        # god_map.debug_expression_manager.add_debug_expression('monitor', effort)
        # self.pub.publish(self.msg)

    def callback(self, joints: JointState):
        for name, effort, velocity in zip(joints.name, joints.effort, joints.velocity):
            if self.alibi_joint_name in name:
                if abs(velocity) < self.velocity_threshold:
                    self.effort = effort
                else:
                    self.effort = 0
        #Todo: this still publishes after the goal is finished. Is there a destructor that could be used to stop the subscriber?
        self.pub.publish(self.msg)
