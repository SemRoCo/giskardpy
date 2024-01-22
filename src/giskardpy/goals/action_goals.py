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


class PouringAction(Goal):
    def __init__(self, tip_link: str, root_link: str,
                 tip_group: str = None, root_group: str = None,
                 max_velocity: float = 0.3, weight: float = WEIGHT_ABOVE_CA,
                 name: String = None,
                 state_topic: str ='/pouringActions',
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
                                                     weights=[self.weight * cas.max(is_rotate_left, is_rotate_right)] * 4,
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
