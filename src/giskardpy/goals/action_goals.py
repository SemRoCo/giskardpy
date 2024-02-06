from __future__ import division

from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.tasks.task import Task, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.cartesian_goals import CartesianOrientation, CartesianPose
import rospy
from std_msgs.msg import String
import math
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.god_map import god_map
from copy import deepcopy
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.expression_definition_utils import transform_msg
from giskardpy.utils.logging import logwarn
from typing import Optional
import giskardpy.utils.tfwrapper as tf
from giskardpy.monitors.payload_monitors import CloseGripper
from giskardpy.monitors.cartesian_monitors import PositionReached


class PouringAction(Goal):
    def __init__(self, tip_link: str, root_link: str,
                 tip_group: str = None, root_group: str = None,
                 max_velocity: float = 0.3, weight: float = WEIGHT_ABOVE_CA,
                 name: str = None,
                 state_topic: str = '/pouringActions',
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        self.root_link = god_map.world.search_for_link_name(root_link, root_group)
        self.tip_link2 = god_map.world.search_for_link_name('hand_camera_frame', tip_group)
        self.tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        self.max_vel = max_velocity / 2
        self.weight = weight
        self.root_group = root_group
        self.tip_group = tip_group
        self.pouring_forward = False
        self.pouring_left = False
        self.pouring_up = False
        self.pouring_backward = False
        self.pouring_right = False
        self.pouring_down = False
        self.pouring_keep_upright = False
        self.pouring_tilt_left = False
        self.pouring_tilt_right = False
        self.pouring_rotate_right = False
        self.pouring_rotate_left = False
        self.map_key_command = {'w': 'forward',
                                's': 'backward',
                                'a': 'left',
                                'd': 'right',
                                'u': 'up',
                                'j': 'down',
                                'y': 'move_to',
                                'g': 'tilt_left',
                                'h': 'tilt_right',
                                'q': 'keep_upright',
                                'z': 'rotate_left',
                                'x': 'rotate_right'}
        self.all_commands = {'forward': 0,
                             'backward': 0,
                             'left': 0,
                             'right': 0,
                             'up': 0,
                             'down': 0,
                             'move_to': 0,
                             'tilt_left': 0,
                             'tilt_right': 0,
                             'keep_upright': 0,
                             'rotate_left': 0,
                             'rotate_right': 0}
        self.all_commands_empty = deepcopy(self.all_commands)
        self.sub = rospy.Subscriber(state_topic, String, self.cb, queue_size=10)

        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        root_P_tip = root_T_tip.to_position()

        is_forward = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'forward\']')
        is_left = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'left\']')
        is_up = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'up\']')
        is_backward = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'backward\']')
        is_right = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'right\']')
        is_down = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'down\']')

        task_movement = self.create_and_add_task('movement')
        is_translation = cas.min(1, is_forward + is_left + is_up + is_backward + is_right + is_down)
        task_movement.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                                     equality_bounds=[
                                                         self.max_vel * is_forward + self.max_vel * -1 * is_backward,
                                                         self.max_vel * is_left + self.max_vel * -1 * is_right,
                                                         self.max_vel * is_up + self.max_vel * -1 * is_down],
                                                     weights=[self.weight * is_translation] * 3,
                                                     task_expression=root_P_tip[:3],
                                                     names=['forward-back', 'left-right', 'up-down'])

        is_uprigth = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'keep_upright\']')

        task_movement.add_vector_goal_constraints(frame_V_current=cas.Vector3(root_T_tip[:3, 0]),
                                                  frame_V_goal=cas.Vector3([0, 0, 1]),
                                                  reference_velocity=self.max_vel,
                                                  weight=self.weight * is_uprigth,
                                                  name='upright')

        is_tilt_left = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'tilt_left\']')
        is_tilt_right = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'tilt_right\']')
        # Todo: I might have to save a reference pose when the pouring starts to define rotation around that
        root_R_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_rotation()
        tip_R_tip = cas.RotationMatrix()
        angle = -0.5 * is_tilt_left + 0.5 * is_tilt_right
        tip_R_tip[0, 0] = cas.cos(angle)
        tip_R_tip[1, 0] = cas.sin(angle)
        tip_R_tip[0, 1] = -cas.sin(angle)
        tip_R_tip[1, 1] = cas.cos(angle)
        tip_R_tip[2, 2] = 1
        root_R_tip_desire = root_R_tip.dot(tip_R_tip)
        task_movement.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 4,
                                                     equality_bounds=[root_R_tip_desire[0, 0] - root_R_tip[0, 0],
                                                                      root_R_tip_desire[1, 0] - root_R_tip[1, 0],
                                                                      root_R_tip_desire[0, 1] - root_R_tip[0, 1],
                                                                      root_R_tip_desire[1, 1] - root_R_tip[1, 1]
                                                                      ],
                                                     weights=[self.weight * cas.max(is_tilt_left, is_tilt_right)] * 4,
                                                     task_expression=[root_R_tip[0, 0],
                                                                      root_R_tip[1, 0],
                                                                      root_R_tip[0, 1],
                                                                      root_R_tip[1, 1]],
                                                     names=['tipr1', 'tipr2', 'tipr3', 'tipr4'])
        task_movement.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                                     equality_bounds=[0] * 3,
                                                     weights=[self.weight * cas.max(is_tilt_left, is_tilt_right)] * 3,
                                                     task_expression=root_P_tip[:3],
                                                     names=['tipp1', 'tipp2', 'tipp3'])
        root_V_tip_z = root_R_tip[:3, 2]
        root_V_z = cas.Vector3([0, 0, 1])
        exp = root_V_tip_z.dot(root_V_z[:3])
        task_movement.add_equality_constraint(reference_velocity=self.max_vel,
                                              equality_bound=0 - exp,
                                              weight=self.weight * cas.max(is_tilt_left, is_tilt_right),
                                              task_expression=exp)

        is_rotate_left = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'rotate_left\']')
        is_rotate_right = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'rotate_right\']')
        base_link = god_map.world.search_for_link_name('base_footprint')
        root_R_base = god_map.world.compose_fk_expression(self.root_link, base_link).to_rotation()
        base_R_base = cas.RotationMatrix()
        angle = 0.5 * is_rotate_left - 0.5 * is_rotate_right
        base_R_base[0, 0] = cas.cos(angle)
        base_R_base[1, 0] = cas.sin(angle)
        base_R_base[0, 1] = -cas.sin(angle)
        base_R_base[1, 1] = cas.cos(angle)
        base_R_base[2, 2] = 1
        root_R_base_desire = root_R_base.dot(base_R_base)
        task_movement.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 4,
                                                     equality_bounds=[root_R_base_desire[0, 0] - root_R_base[0, 0],
                                                                      root_R_base_desire[1, 0] - root_R_base[1, 0],
                                                                      root_R_base_desire[0, 1] - root_R_base[0, 1],
                                                                      root_R_base_desire[1, 1] - root_R_base[1, 1]
                                                                      ],
                                                     weights=[self.weight * cas.max(is_rotate_left,
                                                                                    is_rotate_right)] * 4,
                                                     task_expression=[root_R_base[0, 0],
                                                                      root_R_base[1, 0],
                                                                      root_R_base[0, 1],
                                                                      root_R_base[1, 1]],
                                                     names=['baser1', 'baser2', 'baser3', 'baser4'])

    def cb(self, data: String):
        if data.data == '':
            self.all_commands = deepcopy(self.all_commands_empty)
            return
        keys = data.data.split(';')
        commands = [self.map_key_command[key] for key in keys]
        self.all_commands = deepcopy(self.all_commands_empty)
        for command in commands:
            self.all_commands[command] = 1


class PouringAction2(Goal):
    def __init__(self, tip_link: str, root_link: str,
                 tip_group: str = None, root_group: str = None,
                 max_velocity: float = 0.3, weight: float = WEIGHT_ABOVE_CA,
                 name: str = None,
                 state_topic: str = '/pouringActions',
                 object_name: str = 'free_cup',
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        self.root_link = god_map.world.search_for_link_name(root_link, root_group)
        self.tip_link2 = god_map.world.search_for_link_name('hand_camera_frame', tip_group)
        self.tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        self.max_vel = max_velocity / 2
        self.weight = weight
        self.root_group = root_group
        self.tip_group = tip_group
        self.pouring_forward = False
        self.pouring_left = False
        self.pouring_up = False
        self.pouring_backward = False
        self.pouring_right = False
        self.pouring_down = False
        self.pouring_keep_upright = False
        self.pouring_tilt_left = False
        self.pouring_tilt_right = False
        self.pouring_rotate_right = False
        self.pouring_rotate_left = False
        self.map_key_command = {'w': 'forward',
                                's': 'backward',
                                'a': 'left',
                                'd': 'right',
                                'u': 'up',
                                'j': 'down',
                                'y': 'move_to',
                                'g': 'tilt_left',
                                'h': 'tilt_right',
                                'q': 'keep_upright',
                                'z': 'rotate_left',
                                'x': 'rotate_right',
                                'p': 'pickup',
                                'l': 'putdown',
                                'm': 'align'}
        self.all_commands = {'forward': 0,
                             'backward': 0,
                             'left': 0,
                             'right': 0,
                             'up': 0,
                             'down': 0,
                             'move_to': 0,
                             'tilt_left': 0,
                             'tilt_right': 0,
                             'keep_upright': 0,
                             'rotate_left': 0,
                             'rotate_right': 0,
                             'pickup': 0,
                             'putdown': 0,
                             'align': 0}
        self.all_commands_empty = deepcopy(self.all_commands)
        self.sub = rospy.Subscriber(state_topic, String, self.cb, queue_size=10)

        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        root_P_tip = root_T_tip.to_position()

        is_forward = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'forward\']')
        is_left = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'left\']')
        is_up = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'up\']')
        is_backward = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'backward\']')
        is_right = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'right\']')
        is_down = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'down\']')

        task_movement = self.create_and_add_task('movement')
        is_translation = cas.min(1, is_forward + is_left + is_up + is_backward + is_right + is_down)
        task_movement.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                                     equality_bounds=[
                                                         self.max_vel * is_forward + self.max_vel * -1 * is_backward,
                                                         self.max_vel * is_left + self.max_vel * -1 * is_right,
                                                         self.max_vel * is_up + self.max_vel * -1 * is_down],
                                                     weights=[self.weight * is_translation] * 3,
                                                     task_expression=root_P_tip[:3],
                                                     names=['forward-back', 'left-right', 'up-down'])

        is_uprigth = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'keep_upright\']')

        task_movement.add_vector_goal_constraints(frame_V_current=cas.Vector3(root_T_tip[:3, 0]),
                                                  frame_V_goal=cas.Vector3([0, 0, 1]),
                                                  reference_velocity=self.max_vel,
                                                  weight=self.weight * is_uprigth,
                                                  name='upright')

        is_tilt_left = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'tilt_left\']')
        is_tilt_right = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'tilt_right\']')
        # Todo: I might have to save a reference pose when the pouring starts to define rotation around that
        root_R_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_rotation()
        tip_R_tip = cas.RotationMatrix()
        angle = -0.5 * is_tilt_left + 0.5 * is_tilt_right
        tip_R_tip[0, 0] = cas.cos(angle)
        tip_R_tip[1, 0] = cas.sin(angle)
        tip_R_tip[0, 1] = -cas.sin(angle)
        tip_R_tip[1, 1] = cas.cos(angle)
        tip_R_tip[2, 2] = 1
        root_R_tip_desire = root_R_tip.dot(tip_R_tip)
        task_movement.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 4,
                                                     equality_bounds=[root_R_tip_desire[0, 0] - root_R_tip[0, 0],
                                                                      root_R_tip_desire[1, 0] - root_R_tip[1, 0],
                                                                      root_R_tip_desire[0, 1] - root_R_tip[0, 1],
                                                                      root_R_tip_desire[1, 1] - root_R_tip[1, 1]
                                                                      ],
                                                     weights=[self.weight * cas.max(is_tilt_left, is_tilt_right)] * 4,
                                                     task_expression=[root_R_tip[0, 0],
                                                                      root_R_tip[1, 0],
                                                                      root_R_tip[0, 1],
                                                                      root_R_tip[1, 1]],
                                                     names=['tipr1', 'tipr2', 'tipr3', 'tipr4'])
        task_movement.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                                     equality_bounds=[0] * 3,
                                                     weights=[self.weight * cas.max(is_tilt_left, is_tilt_right)] * 3,
                                                     task_expression=root_P_tip[:3],
                                                     names=['tipp1', 'tipp2', 'tipp3'])
        root_V_tip_z = root_R_tip[:3, 2]
        root_V_z = cas.Vector3([0, 0, 1])
        exp = root_V_tip_z.dot(root_V_z[:3])
        task_movement.add_equality_constraint(reference_velocity=self.max_vel,
                                              equality_bound=0 - exp,
                                              weight=self.weight * cas.max(is_tilt_left, is_tilt_right),
                                              task_expression=exp)

        is_rotate_left = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'rotate_left\']')
        is_rotate_right = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'rotate_right\']')
        base_link = god_map.world.search_for_link_name('base_footprint')
        root_R_base = god_map.world.compose_fk_expression(self.root_link, base_link).to_rotation()
        base_R_base = cas.RotationMatrix()
        angle = 0.5 * is_rotate_left - 0.5 * is_rotate_right
        base_R_base[0, 0] = cas.cos(angle)
        base_R_base[1, 0] = cas.sin(angle)
        base_R_base[0, 1] = -cas.sin(angle)
        base_R_base[1, 1] = cas.cos(angle)
        base_R_base[2, 2] = 1
        root_R_base_desire = root_R_base.dot(base_R_base)
        task_movement.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 4,
                                                     equality_bounds=[root_R_base_desire[0, 0] - root_R_base[0, 0],
                                                                      root_R_base_desire[1, 0] - root_R_base[1, 0],
                                                                      root_R_base_desire[0, 1] - root_R_base[0, 1],
                                                                      root_R_base_desire[1, 1] - root_R_base[1, 1]
                                                                      ],
                                                     weights=[self.weight * cas.max(is_rotate_left,
                                                                                    is_rotate_right)] * 4,
                                                     task_expression=[root_R_base[0, 0],
                                                                      root_R_base[1, 0],
                                                                      root_R_base[0, 1],
                                                                      root_R_base[1, 1]],
                                                     names=['baser1', 'baser2', 'baser3', 'baser4'])

        is_pickup = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'pickup\']')
        is_putdown = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'putdown\']')
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'hand_palm_link'
        goal_pose.pose.position.x = 0.1
        goal_pose.pose.orientation.w = 1
        m1 = ExpressionMonitor(name='test1')
        m1.expression = is_pickup
        self.add_monitor(m1)
        m2 = ExpressionMonitor(name='test2')
        m2.expression = is_putdown
        self.add_monitor(m2)
        self.add_constraints_of_goal(
            PickUp(root_link=root_link, tip_link=tip_link, goal_pose=goal_pose,
                   hold_condition=cas.logic_not(m1.get_state_expression()),
                   name='pickup', parent_goal_name=str(self)))
        self.add_constraints_of_goal(PutDown(name='putdown', parent_goal_name=str(self)))

        goal_normal = Vector3Stamped()
        goal_normal.header.frame_id = object_name
        goal_normal.vector.y = -1
        tip_normal = Vector3Stamped()
        tip_normal.header.frame_id = 'hand_palm_link'
        tip_normal.vector.y = 1
        is_align = symbol_manager.get_symbol(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].all_commands[\'align\']')
        m3 = ExpressionMonitor(name='test3')
        m3.expression = is_align
        self.add_monitor(m3)
        god_map.motion_goal_manager.add_motion_goal(AlignGripperToObject(root_link=root_link, tip_link=tip_link,
                                                          goal_normal=goal_normal,
                                                          tip_normal=tip_normal,
                                                          hold_condition=cas.logic_not(m3.get_state_expression())))
        # self.add_constraints_of_goal(AlignGripperToObject(root_link=root_link, tip_link=tip_link,
        #                                                   goal_normal=goal_normal,
        #                                                   tip_normal=tip_normal,
        #                                                   hold_condition=cas.logic_not(m3.get_state_expression())))

    def cb(self, data: String):
        if data.data == '':
            self.all_commands = deepcopy(self.all_commands_empty)
            return
        keys = data.data.split(';')
        commands = [self.map_key_command[key] for key in keys]
        self.all_commands = deepcopy(self.all_commands_empty)
        for command in commands:
            self.all_commands[command] = 1


# This code is mostly copied from the original align planes goal
class AlignGripperToObject(Goal):
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 goal_normal: Vector3Stamped,
                 tip_normal: Vector3Stamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 reference_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol,
                 **kwargs):
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param goal_normal:
        :param tip_normal:
        :param root_group: if root_link is not unique, search in this group for matches.
        :param tip_group: if tip_link is not unique, search in this group for matches.
        :param reference_velocity: rad/s
        :param weight:
        """
        if 'root_normal' in kwargs:
            logwarn('Deprecated warning: use goal_normal instead of root_normal')
            goal_normal = kwargs['root_normal']
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)
        self.reference_velocity = reference_velocity
        self.weight = weight

        self.tip_V_tip_normal = transform_msg(self.tip, tip_normal)
        self.tip_V_tip_normal.vector = tf.normalize(self.tip_V_tip_normal.vector)

        self.root_V_root_normal = transform_msg(self.root, goal_normal)
        self.root_V_root_normal.vector = tf.normalize(self.root_V_root_normal.vector)
        sub = rospy.Subscriber('/align_goal', Vector3Stamped, callback=self.callback)

        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name)

        task = self.create_and_add_task('align planes')
        tip_V_tip_normal = cas.Vector3(self.tip_V_tip_normal)
        root_R_tip = god_map.world.compose_fk_expression(self.root, self.tip).to_rotation()
        root_V_tip_normal = root_R_tip.dot(tip_V_tip_normal)
        root_V_root_normal = symbol_manager.get_expr(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].root_V_root_normal',
            input_type_hint=Vector3Stamped)
        task.add_vector_goal_constraints(frame_V_current=root_V_tip_normal,
                                         frame_V_goal=root_V_root_normal,
                                         reference_velocity=self.reference_velocity,
                                         weight=self.weight)
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)

    def callback(self, data: Vector3Stamped):
        self.root_V_root_normal = transform_msg(self.root, data)
        self.root_V_root_normal.vector = tf.normalize(self.root_V_root_normal.vector)


class PickUp(Goal):
    # close the gripper and move up a bit
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 goal_pose: PoseStamped,
                 name: Optional[str] = None,
                 parent_goal_name=None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        super().__init__(name)
        m = CloseGripper(name='closeGripperPayloadMonitor', motion_goal_name=parent_goal_name)
        self.add_monitor(m)
        # self.add_constraints_of_goal(CartesianPose(root_link=root_link,
        #                                            tip_link=tip_link,
        #                                            goal_pose=goal_pose,
        #                                            name='movePickUp',
        #                                            start_condition=m.get_state_expression(),
        #                                            hold_condition=cas.logic_or(cas.logic_not(m.get_state_expression()),
        #                                                                        hold_condition)))


class PutDown(Goal):
    def __init__(self,
                 name: Optional[str] = None,
                 parent_goal_name=None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        super().__init__(name)
        m = CloseGripper(name='openGripperPayloadMonitor',
                         motion_goal_name=parent_goal_name,
                         as_open=True,
                         velocity_threshold=100,
                         effort_threshold=1,
                         effort=100)
        self.add_monitor(m)
