from __future__ import division

from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy import identifier
from giskardpy.goals.cartesian_goals import CartesianOrientation, CartesianPose
from giskardpy.goals.pouring_goals import KeepObjectAbovePlane
import rospy
from std_msgs.msg import String
import math


class PouringAction(Goal):
    def __init__(self, tip_link: str, root_link: str, upright_orientation: QuaternionStamped,
                 down_orientation: QuaternionStamped, container_plane: PointStamped, tilt_joint: str,
                 tip_group: str = None, root_group: str = None,
                 max_velocity: float = 0.3, weight: float = WEIGHT_ABOVE_CA):
        super(PouringAction, self).__init__()
        self.root_link = self.world.search_for_link_name(root_link, root_group)
        self.tip_link2 = self.world.search_for_link_name('hand_camera_frame', tip_group)
        self.tip_link = self.world.search_for_link_name(tip_link, tip_group)
        self.max_vel = max_velocity / 2
        self.weight = weight
        self.upright_orientation = upright_orientation
        self.down_orientation = down_orientation
        self.container_plane = container_plane
        self.root_group = root_group
        self.tip_group = tip_group
        self.tilt_joint = self.world.search_for_joint_name(tilt_joint, tip_group)

    def make_constraints(self):
        root_T_tip = self.get_fk(self.root_link, self.tip_link)
        root_P_tip = root_T_tip.to_position()

        is_forward = self.god_map.to_expr(identifier.pouring_forward)
        is_left = self.god_map.to_expr(identifier.pouring_left)
        is_up = self.god_map.to_expr(identifier.pouring_up)
        is_backward = self.god_map.to_expr(identifier.pouring_backward)
        is_right = self.god_map.to_expr(identifier.pouring_right)
        is_down = self.god_map.to_expr(identifier.pouring_down)

        is_translation = w.min(1, is_forward + is_left + is_up + is_backward + is_right + is_down)
        self.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                            equality_bounds=[
                                                self.max_vel * is_forward + self.max_vel * -1 * is_backward,
                                                self.max_vel * is_left + self.max_vel * -1 * is_right,
                                                self.max_vel * is_up + self.max_vel * -1 * is_down],
                                            weights=[self.weight * is_translation] * 3,
                                            task_expression=root_P_tip[:3],
                                            names=['forward-back', 'left-right', 'up-down'])

        is_uprigth = self.god_map.to_expr(identifier.pouring_keep_upright)
        # z-achse parallel wie in grasp goal von simon abgeguckt
        # self.add_constraints_of_goal(CartesianOrientation(root_link=self.root_link,
        #                                                   root_group=self.root_group,
        #                                                   tip_link=self.tip_link,
        #                                                   tip_group=self.tip_group,
        #                                                   goal_orientation=self.upright_orientation,
        #                                                   max_velocity=self.max_vel,
        #                                                   reference_velocity=self.max_vel,
        #                                                   weight=self.weight * is_uprigth,
        #                                                   name_extra='keep_upright'))
        self.add_vector_goal_constraints(frame_V_current=w.Vector3(root_T_tip[:3, 0]),
                                         frame_V_goal=w.Vector3([0, 0, 1]),
                                         reference_velocity=self.max_vel,
                                         weight=self.weight * is_uprigth,
                                         name='upright')

        is_tilt_left = self.god_map.to_expr(identifier.pouring_tilt_left)
        is_tilt_right = self.god_map.to_expr(identifier.pouring_tilt_right)
        # Todo: I might have to save a reference pose when the pouring starts to define rotation around that
        root_R_tip = self.get_fk(self.root_link, self.tip_link).to_rotation()
        tip_R_tip = w.RotationMatrix()
        angle = -0.5 * is_tilt_left + 0.5 * is_tilt_right
        tip_R_tip[0, 0] = w.cos(angle)
        tip_R_tip[1, 0] = w.sin(angle)
        tip_R_tip[0, 1] = -w.sin(angle)
        tip_R_tip[1, 1] = w.cos(angle)
        tip_R_tip[2, 2] = 1
        root_R_tip_desire = root_R_tip.dot(tip_R_tip)
        self.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 4,
                                            equality_bounds=[root_R_tip_desire[0, 0] - root_R_tip[0, 0],
                                                             root_R_tip_desire[1, 0] - root_R_tip[1, 0],
                                                             root_R_tip_desire[0, 1] - root_R_tip[0, 1],
                                                             root_R_tip_desire[1, 1] - root_R_tip[1, 1]
                                                             ],
                                            weights=[self.weight * w.max(is_tilt_left, is_tilt_right)] * 4,
                                            task_expression=[root_R_tip[0, 0],
                                                             root_R_tip[1, 0],
                                                             root_R_tip[0, 1],
                                                             root_R_tip[1, 1]],
                                            names=['tipr1', 'tipr2', 'tipr3', 'tipr4'])
        self.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 3,
                                            equality_bounds=[0] * 3,
                                            weights=[self.weight * w.max(is_tilt_left, is_tilt_right)] * 3,
                                            task_expression=root_P_tip[:3],
                                            names=['tipp1', 'tipp2', 'tipp3'])
        root_V_tip_z = root_R_tip[:3, 2]
        root_V_z = w.Vector3([0, 0, 1])
        exp = root_V_tip_z.dot(root_V_z[:3])
        self.add_equality_constraint(reference_velocity=self.max_vel,
                                     equality_bound=0 - exp,
                                     weight=self.weight * w.max(is_tilt_left, is_tilt_right),
                                     task_expression=exp)

        is_rotate_left = self.god_map.to_expr(identifier.pouring_rotate_left)
        is_rotate_right = self.god_map.to_expr(identifier.pouring_rotate_right)
        base_link = self.world.search_for_link_name('base_footprint')
        root_R_base = self.get_fk(self.root_link, base_link).to_rotation()
        base_R_base = w.RotationMatrix()
        angle = 0.5 * is_rotate_left - 0.5 * is_rotate_right
        base_R_base[0, 0] = w.cos(angle)
        base_R_base[1, 0] = w.sin(angle)
        base_R_base[0, 1] = -w.sin(angle)
        base_R_base[1, 1] = w.cos(angle)
        base_R_base[2, 2] = 1
        root_R_base_desire = root_R_base.dot(base_R_base)
        self.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 4,
                                            equality_bounds=[root_R_base_desire[0, 0] - root_R_base[0, 0],
                                                             root_R_base_desire[1, 0] - root_R_base[1, 0],
                                                             root_R_base_desire[0, 1] - root_R_base[0, 1],
                                                             root_R_base_desire[1, 1] - root_R_base[1, 1]
                                                             ],
                                            weights=[self.weight * w.max(is_rotate_left, is_rotate_right)] * 4,
                                            task_expression=[root_R_base[0, 0],
                                                             root_R_base[1, 0],
                                                             root_R_base[0, 1],
                                                             root_R_base[1, 1]],
                                            names=['baser1', 'baser2', 'baser3', 'baser4'])
        # v = Vector3Stamped()
        # v.header.frame_id = 'hand_palm_link'
        # v.vector.z = 1
        # self.add_constraints_of_goal(TiltObject(object_link='hand_palm_link',
        #                                         reference_link='map',
        #                                         rotation_axis=v,
        #                                         rotation_velocity=0.9,
        #                                         lower_angle=-2,
        #                                         root_link='map',
        #                                         upper_angle=None,
        #                                         weight=self.weight * is_tilt,
        #                                         name_extra='tilt_object'))
        # r = Vector3Stamped()
        # r.header.frame_id = 'map'
        # r.vector.x = 1
        # self.add_constraints_of_goal(KeepObjectUpright(object_link_axis=v,
        #                                                reference_link_axis=r,
        #                                                root_link='map',
        #                                                weight=self.weight * is_tilt))
        # yaw = self.get_fk(self.root_link, self.tip_link).to_rotation().to_rpy()[2]
        # self.add_equality_constraint(reference_velocity=self.max_vel,
        #                              equality_bound=-0.1,
        #                              weight=self.weight * is_tilt,
        #                              task_expression=yaw,
        #                              name='yaw')
        # self.add_constraints_of_goal(CartesianOrientation(root_link=self.root_link,
        #                                                   root_group=self.root_group,
        #                                                   tip_link=self.tip_link,
        #                                                   tip_group=self.tip_group,
        #                                                   goal_orientation=self.down_orientation,
        #                                                   max_velocity=self.max_vel,
        #                                                   reference_velocity=self.max_vel,
        #                                                   weight=self.weight * is_tilt,
        #                                                   name_extra='tilt'))

        lower_distance = 0.2
        upper_distance = 0.3
        plane_radius = 0
        is_move_to = self.god_map.to_expr(identifier.pouring_move_to)
        self.add_constraints_of_goal(KeepObjectAbovePlane(object_link=self.tip_link,
                                                          plane_center_point=self.container_plane,
                                                          lower_distance=lower_distance,
                                                          upper_distance=upper_distance,
                                                          plane_radius=plane_radius,
                                                          root_link=self.root_link,
                                                          weight=self.weight * is_move_to))

        self.add_debug_expr('forward', is_forward)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'


class PouringAdaptiveTilt(Goal):
    def __init__(self, root, tip, tilt_angle, max_vel=0.3, weight=WEIGHT_BELOW_CA):
        super().__init__()
        self.action_sub = rospy.Subscriber('/adapt_tilt', String, self.callback)
        self.root_link = self.world.search_for_link_name(root, None)
        self.tip_link = self.world.search_for_link_name(tip, None)
        self.root = root
        self.tip = tip
        self.max_vel = max_vel
        self.weight = weight
        self.action_string = ''
        self.tilt_angle = tilt_angle

        self.root_P_current = self.get_fk_evaluated(self.root_link, self.tip_link).to_position()
        self.was_close = False
        self.counter = 0
        self.forward = False
        self.backward = False
        self.current_angle = 0
        self.stop_plan = 0

    # def reached_goal_pose(self):
    #     if self.was_close:
    #         return 1
    #     if abs(abs(self.tilt_angle) - self.current_angle) < 0.01:
    #         self.was_close = True
    #     return 0

    def callback(self, action_string: String):
        self.action_string = action_string.data
        if action_string.data == 'forward':
            self.forward = True
            self.backward = False
            self.stop_plan = 1
        elif action_string.data == 'backward':
            self.backward = True
            self.forward = False
            self.stop_plan = 1
        else:
            self.forward = False
            self.backward = False

    def make_constraints(self):
        root_R_start = w.RotationMatrix([[0, 0, 1, 0],
                                         [0, -1, 0, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, 0, 1]])
        root_R_tip = self.get_fk(self.root_link, self.tip_link).to_rotation()
        stop_planned = self.get_parameter_as_symbolic_expression('stop_plan')
        self.add_debug_expr('stop_planned', stop_planned)
        tip_R_tip = w.RotationMatrix()
        angle = self.tilt_angle
        tip_R_tip[0, 0] = w.cos(angle)
        tip_R_tip[1, 0] = w.sin(angle)
        tip_R_tip[0, 1] = -w.sin(angle)
        tip_R_tip[1, 1] = w.cos(angle)
        tip_R_tip[2, 2] = 1
        root_R_tip_desire = root_R_start.dot(tip_R_tip)
        self.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 4,
                                            equality_bounds=[root_R_tip_desire[0, 0] - root_R_tip[0, 0],
                                                             root_R_tip_desire[1, 0] - root_R_tip[1, 0],
                                                             root_R_tip_desire[0, 1] - root_R_tip[0, 1],
                                                             root_R_tip_desire[1, 1] - root_R_tip[1, 1]
                                                             ],
                                            weights=[self.weight * (1 - stop_planned)] * 4,
                                            task_expression=[root_R_tip[0, 0],
                                                             root_R_tip[1, 0],
                                                             root_R_tip[0, 1],
                                                             root_R_tip[1, 1]],
                                            names=['tipr1', 'tipr2', 'tipr3', 'tipr4'])
        root_V_tip_z = root_R_tip[:3, 2]
        root_V_z = w.Vector3([0, 0, 1])
        exp = root_V_tip_z.dot(root_V_z[:3])
        self.add_equality_constraint(reference_velocity=self.max_vel,
                                     equality_bound=0 - exp,
                                     weight=self.weight,
                                     task_expression=exp)

        tip_R_tip_a = w.RotationMatrix()
        angle_a = -1 * self.get_parameter_as_symbolic_expression('forward') + \
                  1 * self.get_parameter_as_symbolic_expression('backward')
        tip_R_tip_a[0, 0] = w.cos(angle_a)
        tip_R_tip_a[1, 0] = w.sin(angle_a)
        tip_R_tip_a[0, 1] = -w.sin(angle_a)
        tip_R_tip_a[1, 1] = w.cos(angle_a)
        tip_R_tip_a[2, 2] = 1
        root_R_tip_desired_a = root_R_tip.dot(tip_R_tip_a)
        angle = w.angle_between_vector(w.Vector3(root_R_tip[:, 0]), w.Vector3([0, 0, 1]))
        stop_to_large = w.if_greater(angle, 3, 0, 1)
        stop_to_small = w.if_less(angle, 0.1, 0, 1)
        self.add_equality_constraint_vector(reference_velocities=[self.max_vel] * 4,
                                            equality_bounds=[root_R_tip_desired_a[0, 0] - root_R_tip[0, 0],
                                                             root_R_tip_desired_a[1, 0] - root_R_tip[1, 0],
                                                             root_R_tip_desired_a[0, 1] - root_R_tip[0, 1],
                                                             root_R_tip_desired_a[1, 1] - root_R_tip[1, 1]
                                                             ],
                                            weights=[self.weight
                                                     * stop_planned
                                                     * stop_to_large
                                                     * stop_to_small] * 4,
                                            task_expression=[root_R_tip[0, 0],
                                                             root_R_tip[1, 0],
                                                             root_R_tip[0, 1],
                                                             root_R_tip[1, 1]],
                                            names=['tipr1a', 'tipr2a', 'tipr3a', 'tipr4a'])

    def __str__(self) -> str:
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'
