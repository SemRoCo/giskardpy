import math

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy import identifier
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from typing import Optional
from geometry_msgs.msg import Vector3Stamped, PointStamped, QuaternionStamped
from giskardpy.goals.cartesian_goals import CartesianOrientation


class BalanceBall(Goal):
    def __init__(self,
                 ball_name: str,
                 tray_name: str,
                 root_link: str,
                 weight: float = WEIGHT_BELOW_CA,
                 ref_vel: float = 1,
                 root_group: Optional[str] = None, ):
        super().__init__()
        self.weight = weight
        self.ball = self.world.search_for_link_name(ball_name)
        self.tray = self.world.search_for_link_name(tray_name)
        self.root = self.world.search_for_link_name(root_link)
        self.ref_vel = ref_vel
        self.root_group = root_group
        self.tray_name = tray_name
        self.root_name = root_link

    def make_constraints(self):
        root_T_ball = self.get_fk(self.root, self.ball)
        root_T_tray = self.get_fk(self.root, self.tray)

        root_R_tray = root_T_tray.to_rotation()
        tray_R_root_eval = self.get_fk_evaluated(self.tray, self.root).to_rotation()

        root_P_tray = root_T_tray.to_position()
        root_P_ball = root_T_ball.to_position()

        # y_error = root_P_tray[1] - root_P_ball[1]
        # root_V_rot_axis = Vector3Stamped()
        # root_V_rot_axis.vector.x = 1
        # # root_Q_goal = get_quaternion_from_rotation_around_axis(-y_error, root_V_rot_axis, self.root)
        # gain = w.if_less(w.abs(y_error), 0.1, -2, 0.5)
        # gain = w.if_less(w.abs(y_error), 0.02, -1, gain)
        # gain = w.if_less(w.abs(y_error), 0.01, 0.5, gain)
        # # gain = 1
        # root_R_goal = w.RotationMatrix.from_axis_angle(w.Vector3(root_V_rot_axis), -y_error * gain)
        #
        # # self.add_constraints_of_goal(CartesianOrientation(root_link=self.root_name,
        # #                                                   root_group=self.root_group,
        # #                                                   tip_link=self.tray_name,
        # #                                                   tip_group=self.root_group,
        # #                                                   goal_orientation=q,
        # #                                                   max_velocity=self.ref_vel,
        # #                                                   reference_velocity=self.ref_vel,
        # #                                                   weight=self.weight))
        # # self.add_rotation_goal_constraints(frame_R_current=root_R_tray,
        # #                                    frame_R_goal=root_R_goal,
        # #                                    current_R_frame_eval=tray_R_root_eval,
        # #                                    reference_velocity=self.ref_vel,
        # #                                    weight=self.weight)
        # self.add_debug_expr('error', y_error)
        #
        # x_error = root_P_tray[0] - root_P_ball[0]
        # root_V_rot_axis2 = Vector3Stamped()
        # root_V_rot_axis2.vector.y = 1
        # # root_Q_goal = get_quaternion_from_rotation_around_axis(-y_error, root_V_rot_axis, self.root)
        # root_R_goal2 = w.RotationMatrix.from_axis_angle(w.Vector3(root_V_rot_axis2), x_error * 0.2)
        #
        # # root_R_goal = root_R_goal.dot(root_R_goal2)
        # self.add_rotation_goal_constraints(frame_R_current=root_R_tray,
        #                                    frame_R_goal=root_R_goal,
        #                                    current_R_frame_eval=tray_R_root_eval,
        #                                    reference_velocity=self.ref_vel,
        #                                    weight=self.weight)
        # --------------------------------------------------------------------------------------------------------------
        # next position of ball = position + velocity of ball * time
        # velocity of ball = velocity of ball + acc of ball * time
        # acc of ball is proportional to angle of tray => acc = gravity * sin(angle)  # without friction

        tray_P_ball = self.get_fk(self.tray, self.ball).to_position()
        tray_P_ball[2] = 0
        tray_P_ball[3] = 0
        root_V_gravity = root_T_tray.dot(tray_P_ball)
        root_V_gravity = w.if_less(root_P_ball[2], root_P_tray[2] + 0.03, root_V_gravity, -root_V_gravity)
        root_V_gravity /= w.norm(root_V_gravity)

        time = 0.02
        axis, angle = root_R_tray.to_axis_angle()
        rpy = root_R_tray.to_rpy()

        acc = w.Vector3(root_V_gravity) * 9.81 * angle
        vel = self.god_map.to_expr(identifier.ball_velocity) + acc

        root_P_ball_eval = self.get_fk_evaluated(self.root, self.ball).to_position()
        pos = root_P_ball_eval - acc

        self.add_equality_constraint_vector(reference_velocities=[self.ref_vel] * 3,
                                            equality_bounds=root_P_tray[:3] - pos[:3] + w.Vector3([0, 0, -0.03])[:3],
                                            weights=[self.weight] * 3,
                                            task_expression=pos[:3],
                                            names=['1', '2', '3'])

        # self.add_equality_constraint_vector(reference_velocities=[self.ref_vel] * 3,
        #                                     equality_bounds=-(root_P_tray[:3] - pos[:3] + w.Vector3([0, 0, -0.03])[:3]),
        #                                     weights=[self.weight] * 3,
        #                                     task_expression=self.get_expr_velocity(pos[:3]),
        #                                     names=['11', '12', '13'])

        # self.add_inequality_constraint(reference_velocity=self.ref_vel,
        #                                lower_error=0 - angle,
        #                                upper_error=0.4 - angle,
        #                                weight=WEIGHT_BELOW_CA,
        #                                task_expression=angle,
        #                                name='angle')

        self.add_debug_expr('error', root_P_tray[:3] - pos[:3] + w.Vector3([0, 0, -0.03])[:3])

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root}/{self.ball}'
