from geometry_msgs.msg import PointStamped
import giskardpy.casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.my_types import PrefixName


class Caster(Goal):
    def __init__(self, joint_name: str):
        super().__init__()
        self.joint_name = self.world.get_joint_name(joint_name)

    def make_constraints(self):
        joint = self.world.get_joint(self.joint_name)
        link_name = joint.child_link_name
        bf_T_caster = self.get_fk(PrefixName('base_footprint', 'pr2'), link_name)
        bf_R_caster = bf_T_caster.to_rotation()
        axis, angle = bf_R_caster.to_axis_angle()
        limit = 1
        self.add_velocity_constraint(lower_velocity_limit=-limit,
                                     upper_velocity_limit=limit,
                                     weight=WEIGHT_ABOVE_CA,
                                     task_expression=angle,
                                     velocity_limit=limit,
                                     name_suffix='angle')
        self.add_debug_expr('angle', angle)

    def __str__(self) -> str:
        return f'{super().__str__()}/{self.joint_name}'


class Circle(Goal):

    def __init__(self, center: PointStamped, radius: float, tip_link: str, scale: float):
        super().__init__()
        self.center = self.transform_msg(self.world.root_link_name, center)
        self.radius = radius
        self.scale = scale
        self.tip_link_name = self.world.get_link_name(tip_link)

    def make_constraints(self):
        map_T_bf = self.get_fk(self.world.root_link_name, self.tip_link_name)
        t = self.traj_time_in_seconds() * self.scale
        x = w.cos(t) * self.radius
        y = w.sin(t) * self.radius
        map_P_center = w.Point3(self.center)
        map_T_center = w.TransMatrix.from_point_rotation_matrix(map_P_center)
        center_V_center_to_bf_goal = w.Vector3((-x, -y, 0))
        map_V_bf_to_center = map_T_center.dot(center_V_center_to_bf_goal)
        bf_V_y = w.Vector3((0, 1, 0))
        map_V_y = map_T_bf.dot(bf_V_y)
        map_V_y.vis_frame = self.tip_link_name
        map_V_bf_to_center.vis_frame = self.tip_link_name
        map_V_y.scale(1)
        map_V_bf_to_center.scale(1)
        self.add_debug_expr('map_V_y', map_V_y)
        self.add_debug_expr('map_V_bf_to_center', map_V_bf_to_center)
        self.add_vector_goal_constraints(frame_V_current=map_V_y,
                                         frame_V_goal=map_V_bf_to_center,
                                         reference_velocity=0.1,
                                         weight=WEIGHT_ABOVE_CA,
                                         name='orientation')

        center_P_bf_goal = w.Point3((x, y, 0))
        map_P_bf_goal = map_T_center.dot(center_P_bf_goal)
        map_P_bf = map_T_bf.to_position()
        self.add_debug_expr('map_P_bf_goal', map_P_bf_goal)
        self.add_point_goal_constraints(frame_P_current=map_P_bf,
                                        frame_P_goal=map_P_bf_goal,
                                        reference_velocity=0.1,
                                        weight=WEIGHT_BELOW_CA,
                                        name='position')

    def __str__(self) -> str:
        return super().__str__()
