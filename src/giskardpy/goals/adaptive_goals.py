from __future__ import division

from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from giskardpy import casadi_wrapper as w
from giskardpy.goals.tasks.task import Task, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE \
    , Monitor
from giskardpy.goals.goal import Goal
import rospy
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.my_types import PrefixName, Derivatives
import math
from typing import Optional, List


# Todo: instead of relying on predefined poses model the motion as relations between the objects
# Todo: take velocity constraints more into account
class PouringAdaptiveTilt(Goal):
    def __init__(self, root, tip, pouring_pose: PoseStamped, tilt_angle: float, tilt_axis: Vector3Stamped,
                 use_local_min=False, max_vel=0.3, weight=WEIGHT_COLLISION_AVOIDANCE, pre_tilt=False,
                 name: Optional[str] = None,
                 to_start: Optional[List[Monitor]] = None,
                 to_hold: Optional[List[Monitor]] = None,
                 to_end: Optional[List[Monitor]] = None):
        super().__init__(name)
        self.action_sub = rospy.Subscriber('/reasoner/concluded_behaviors', String, self.callback)
        self.root_link = god_map.world.search_for_link_name(root, None)
        self.tip_link = god_map.world.search_for_link_name(tip, None)
        self.tilt_axis = tilt_axis
        self.tilt_axis.header.frame_id = god_map.world.search_for_link_name(tilt_axis.header.frame_id, None)
        self.tilt_axis = god_map.world.transform_msg(self.tip_link, tilt_axis)
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
        tip_V_tilt_axis = w.Vector3(self.tilt_axis.vector)

        self.pos_task = Task(name='prePosition')
        root_T_goal = god_map.world.transform_msg(self.root_link, pouring_pose)
        root_P_goal = w.Point3(root_T_goal.pose.position)
        self.pos_task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(), frame_P_goal=root_P_goal,
                                                 reference_velocity=self.max_vel, weight=self.weight)

        rot_task = Task(name='preRotation')
        if pre_tilt:
            tip_R_tip = w.RotationMatrix().from_axis_angle(tip_V_tilt_axis, self.tilt_angle/2)
            root_R_tip_desired = w.RotationMatrix(root_T_goal.pose.orientation).dot(tip_R_tip)
        else:
            root_R_tip_desired = w.RotationMatrix(root_T_goal.pose.orientation)
        rot_task.add_rotation_goal_constraints(frame_R_current=root_T_tip.to_rotation(),
                                               frame_R_goal=root_R_tip_desired,
                                               current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(
                                                   self.tip_link, self.root_link),
                                               reference_velocity=self.max_vel,
                                               weight=self.weight)

        pos_monitor = Monitor(name='isPositioned', crucial=True, stay_one=True)
        self.add_monitor(pos_monitor)
        pos_monitor.set_expression(w.less(w.euclidean_distance(root_T_tip.to_position(), root_P_goal), 0.005))
        # pos_task.add_to_end_monitor(pos_monitor)
        rot_task.add_to_end_monitor(pos_monitor)
        self.add_task(self.pos_task)
        self.add_task(rot_task)

        # Begin nominal rotation goal
        root_R_start = w.RotationMatrix(root_T_goal.pose.orientation)
        root_R_tip = root_T_tip.to_rotation()

        # Rotate tip about angle around z axis
        tip_R_tip = w.RotationMatrix().from_axis_angle(tip_V_tilt_axis, self.tilt_angle)
        root_R_tip_desired = root_R_start.dot(tip_R_tip)

        nominal_task = Task(name='TiltToPlanned')
        # add constraint that achieves the desired rotation
        nominal_task.add_rotation_goal_constraints(frame_R_current=root_T_tip.to_rotation(),
                                                   frame_R_goal=root_R_tip_desired,
                                                   current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(
                                                       self.tip_link, self.root_link),
                                                   reference_velocity=self.max_vel,
                                                   weight=self.weight)
        # add constraint to keep the z axis orientation at starting position
        # this constraint ensures orthogonality between the z-axis of the world and z-axis of the tip
        root_V_tip_z = root_R_tip[:3, 2]
        root_V_z = w.Vector3([0, 0, 1])
        exp = root_V_tip_z.dot(root_V_z[:3])
        nominal_task.add_equality_constraint(reference_velocity=self.max_vel,
                                             equality_bound=0 - exp,
                                             weight=self.weight,
                                             task_expression=exp)
        # monitor for the nominal task
        nominal_monitor = Monitor(name='isTilted',
                                  crucial=True,
                                  stay_one=True)
        # monitor to observe the angle and stop the nominal task
        self.add_monitor(nominal_monitor)
        nominal_error = w.angle_between_vector(w.Vector3(root_R_tip[:3, 0]), w.Vector3(root_R_tip_desired[:3, 0]))
        nominal_monitor.set_expression(w.less(nominal_error, 0.1))
        nominal_task.add_to_start_monitor(pos_monitor)
        nominal_task.add_to_end_monitor(nominal_monitor)
        self.pos_task.add_to_end_monitor(nominal_monitor)
        self.add_task(nominal_task)
        god_map.debug_expression_manager.add_debug_expression('error', nominal_error)

        # add the adaptive part of the goal
        adaptive_task = Task(name='adaptTiltFeedback')
        is_forward = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].forward')
        is_backward = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].backward')
        if self.tilt_angle < 0:
            angle_a = -0.01 * is_forward + 1 * is_backward
        else:
            angle_a = 0.01 * is_forward - 1 * is_backward
        tip_R_tip_a = w.RotationMatrix().from_axis_angle(tip_V_tilt_axis, angle_a)
        root_R_tip_desired_a = root_R_tip.dot(tip_R_tip_a)
        angle = w.angle_between_vector(w.Vector3(root_R_tip[:, 0]), w.Vector3([0, 0, 1]))
        stop_to_large = w.if_greater(angle, 3, 0, 1)
        stop_to_small = w.if_less(angle, 0.1, 0, 1)
        adaptive_task.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 4,
                                                     equality_bounds=[root_R_tip_desired_a[0, 0] - root_R_tip[0, 0],
                                                                      root_R_tip_desired_a[1, 0] - root_R_tip[1, 0],
                                                                      root_R_tip_desired_a[0, 1] - root_R_tip[0, 1],
                                                                      root_R_tip_desired_a[1, 1] - root_R_tip[1, 1]
                                                                      ],
                                                     weights=[self.weight
                                                              * stop_to_large
                                                              * stop_to_small] * 4,
                                                     task_expression=[root_R_tip[0, 0],
                                                                      root_R_tip[1, 0],
                                                                      root_R_tip[0, 1],
                                                                      root_R_tip[1, 1]],
                                                     names=['tipr1a', 'tipr2a', 'tipr3a', 'tipr4a'])
        adaptive_task.add_to_start_monitor(nominal_monitor)
        adaptive_task.add_equality_constraint(reference_velocity=self.max_vel,
                                              equality_bound=0 - exp,
                                              weight=self.weight,
                                              task_expression=exp)
        self.add_task(adaptive_task)

        # Todo: make it smooth and nice to look at, tilt back when adapting the position
        adapt_pos_task = Task('adaptPosition')
        adapt_pos_task.add_to_start_monitor(nominal_monitor)
        is_x = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_x')
        is_x_back = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_x_back')
        is_y = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_y')
        is_y_back = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].move_y_back')
        root_V_adapt = w.Vector3([0.1 * is_x - 0.1 * is_x_back,
                                  0.1 * is_y - 0.1 * is_y_back,
                                  0
                                  ])
        root_P_tip_eval = god_map.world.compose_fk_evaluated_expression(self.root_link, self.tip_link).to_position()
        adapt_pos_task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                                  frame_P_goal=root_P_goal + root_V_adapt,
                                                  reference_velocity=self.max_vel/3,
                                                  weight=self.weight)
        self.add_task(adapt_pos_task)

        external_end_monitor = Monitor('isFnished', crucial=True)
        self.add_monitor(external_end_monitor)
        external_end_monitor.set_expression(
            symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].stop'))
        adaptive_task.add_to_end_monitor(external_end_monitor)
        adapt_pos_task.add_to_end_monitor(external_end_monitor)
        # self.pos_task.add_to_end_monitor(external_end_monitor)

    def callback(self, action_string: String):
        self.action_string = action_string.data
        if 'increase' in action_string.data:
            self.forward = True
            self.backward = False
        elif 'decrease' in action_string.data:
            self.backward = True
            self.forward = False
        else:
            self.forward = False
            self.backward = False

        if 'moveForward' in action_string.data:
            self.move_x = True
            self.move_x_back = False
            self.move_y = False
            self.move_y_back = False
        elif 'moveBackward' in action_string.data:
            self.move_x = False
            self.move_x_back = True
            self.move_y = False
            self.move_y_back = False
        elif 'moveLeft' in action_string.data:
            self.move_x = False
            self.move_x_back = False
            self.move_y = True
            self.move_y_back = False
        elif 'moveRight' in action_string.data:
            self.move_x = False
            self.move_x_back = False
            self.move_y = False
            self.move_y_back = True
        else:
            self.move_x = False
            self.move_x_back = False
            self.move_y = False
            self.move_y_back = False

        if '{}' in action_string.data:
            self.stop_counter += 1
            if self.stop_counter > 10:
                self.stop = 1
        else:
            self.stop_counter = 0

    def connect_to_end(self, monitor: Monitor):
        if self.use_local_min:
            self.tasks[-1].add_to_end_monitor(monitor)
            self.tasks[-2].add_to_end_monitor(monitor)


class CloseGripper(Goal):
    def __init__(self, effort: int = -180, pub_topic='hsrb4s/hand_motor_joint_velocity_controller/command',
                 joint_state_topic='hsrb4s/joint_states', alibi_joint_name='hand_motor_joint', joint_group=None,
                 velocity_threshold=0.1, effort_threshold=-1, as_open=False, name: Optional[str] = None,
                 to_start: Optional[List[Monitor]] = None,
                 to_hold: Optional[List[Monitor]] = None,
                 to_end: Optional[List[Monitor]] = None):
        super().__init__(name)
        self.pub = rospy.Publisher(pub_topic, Float64, queue_size=1)
        self.effort = 0
        self.velocity_threshold = velocity_threshold
        rospy.Subscriber(joint_state_topic, JointState, self.callback)

        self.msg = Float64()
        self.msg.data = effort

        self.task = Task(name='gripper')
        joint_name = god_map.world.search_for_joint_name(alibi_joint_name, joint_group)
        joint = god_map.world.joints[joint_name]
        self.task.add_equality_constraint(reference_velocity=0.2,
                                          equality_bound=0 - joint.get_symbol(Derivatives.position),
                                          weight=WEIGHT_BELOW_CA,
                                          task_expression=joint.get_symbol(Derivatives.position))
        self.add_task(self.task)
        monitor = Monitor(name='efforMoniotor', crucial=True)
        effort = symbol_manager.get_symbol(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].effort')
        if as_open:
            monitor.set_expression(w.if_greater(effort, effort_threshold, 1, 0))
        else:
            monitor.set_expression(w.if_less(effort, effort_threshold, 1, 0))
        self.add_monitor(monitor)
        self.task.add_to_end_monitor(monitor)
        god_map.debug_expression_manager.add_debug_expression('monitor', effort)

    def callback(self, joints: JointState):
        for name, effort, velocity in zip(joints.name, joints.effort, joints.velocity):
            if 'hand_motor_joint' in name:
                if abs(velocity) < self.velocity_threshold:
                    self.effort = effort
                else:
                    self.effort = 0
        self.pub.publish(self.msg)

    def connect_to_end(self, monitor: Monitor):
        pass
