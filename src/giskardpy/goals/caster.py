from geometry_msgs.msg import PointStamped
import giskardpy.casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.my_types import PrefixName


class PR2CasterConstraints(Goal):
    def __init__(self, velocity_limit: float = 1000):
        super().__init__()
        casters = ['fl_caster_rotation_joint',
                   'fr_caster_rotation_joint',
                   'bl_caster_rotation_joint',
                   'br_caster_rotation_joint']
        for caster in casters:
            self.add_constraints_of_goal(Caster(joint_name=caster, velocity_limit=velocity_limit))

    def make_constraints(self):
        pass

    def __str__(self) -> str:
        return super().__str__()


class Caster(Goal):
    def __init__(self, joint_name: str, velocity_limit: float = 1):
        super().__init__()
        self.joint_name = self.world.search_for_joint_name(joint_name)
        self.velocity_limit = velocity_limit

    def make_constraints(self):
        joint = self.world.get_joint(self.joint_name)
        link_name = joint.child_link_name
        bf_T_caster = self.get_fk(PrefixName('base_footprint', 'pr2'), link_name)
        bf_V_x = w.Vector3((1, 0, 0))
        bf_V_caster_x = bf_T_caster.dot(w.Vector3((1, 0, 0)))
        yaw = w.angle_between_vector(bf_V_x, bf_V_caster_x)
        # bf_R_caster = bf_T_caster.to_rotation()
        # yaw = bf_R_caster.to_angle(lambda axis: axis[2])
        # roll, pitch, yaw = bf_R_caster.to_rpy()
        # axis.vis_frame = link_name
        # axis.scale(angle)
        self.add_debug_expr('angle', yaw)
        # self.add_debug_expr('axis', axis)
        # self.add_debug_expr('angle_vel', w.total_derivative(angle, self.joint_velocity_symbols, self.joint_acceleration_symbols))
        # self.add_velocity_constraint(lower_velocity_limit=-1000,
        #                              upper_velocity_limit=1000,
        #                              weight=0.01,
        #                              task_expression=yaw,
        #                              velocity_limit=self.velocity_limit,
        #                              # lower_slack_limit=-1000,
        #                              # upper_slack_limit=0,
        #                              name_suffix='/angle/vel')
        a1 = 100
        a2 = 50
        a3 = 1000
        self.add_acceleration_constraint(lower_acceleration_limit=-a1,
                                         upper_acceleration_limit=a1,
                                         weight=0.01,
                                         task_expression=yaw,
                                         acceleration_limit=a2,
                                         lower_slack_limit=-a3,
                                         upper_slack_limit=a3,
                                         name_suffix='/angle/acc')
        # j1 = 0
        j2 = 1000
        j3 = 5000000
        # self.add_jerk_constraint(lower_jerk_limit=-j3,
        #                          upper_jerk_limit=j3,
        #                          weight=0.0,
        #                          task_expression=yaw,
        #                          acceleration_limit=j2,
        #                          lower_slack_limit=-j3,
        #                          upper_slack_limit=j3,
        #                          name_suffix='/angle/jerk')

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
        t = w.min(t, 30 * self.scale)
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
        # self.add_debug_expr('map_V_y', map_V_y)
        # self.add_debug_expr('map_V_bf_to_center', map_V_bf_to_center)
        self.add_vector_goal_constraints(frame_V_current=map_V_y,
                                         frame_V_goal=map_V_bf_to_center,
                                         reference_velocity=0.1,
                                         weight=WEIGHT_ABOVE_CA,
                                         name='orientation')

        center_P_bf_goal = w.Point3((x, y, 0))
        map_P_bf_goal = map_T_center.dot(center_P_bf_goal)
        map_P_bf = map_T_bf.to_position()
        # self.add_debug_expr('map_P_bf_goal', map_P_bf_goal)
        self.add_point_goal_constraints(frame_P_current=map_P_bf,
                                        frame_P_goal=map_P_bf_goal,
                                        reference_velocity=0.1,
                                        weight=WEIGHT_BELOW_CA,
                                        name='position')

    def __str__(self) -> str:
        return super().__str__()


class Wave(Goal):

    def __init__(self, center: PointStamped, radius: float, tip_link: str, scale: float):
        super().__init__()
        self.center = self.transform_msg(self.world.root_link_name, center)
        self.radius = radius
        self.scale = scale
        self.tip_link_name = self.world.get_link_name(tip_link)

    def make_constraints(self):
        map_T_bf = self.get_fk(self.world.root_link_name, self.tip_link_name)
        t = self.traj_time_in_seconds() * self.scale
        t = w.min(t, 30 * self.scale)
        x = w.sin(t) * self.radius
        # y = w.sin(t) * self.radius
        map_P_center = w.Point3(self.center)
        map_T_center = w.TransMatrix.from_point_rotation_matrix(map_P_center)
        # center_V_center_to_bf_goal = w.Vector3((-x, 0, 0))
        # map_V_bf_to_center = map_T_center.dot(center_V_center_to_bf_goal)
        # bf_V_y = w.Vector3((0, 1, 0))
        # map_V_y = map_T_bf.dot(bf_V_y)
        # map_V_y.vis_frame = self.tip_link_name
        # map_V_bf_to_center.vis_frame = self.tip_link_name
        # map_V_y.scale(1)
        # map_V_bf_to_center.scale(1)
        # self.add_debug_expr('map_V_y', map_V_y)
        # self.add_debug_expr('map_V_bf_to_center', map_V_bf_to_center)
        # self.add_vector_goal_constraints(frame_V_current=map_V_y,
        #                                  frame_V_goal=map_V_bf_to_center,
        #                                  reference_velocity=0.1,
        #                                  weight=WEIGHT_ABOVE_CA,
        #                                  name='orientation')

        center_P_bf_goal = w.Point3((x, 0, 0))
        map_P_bf_goal = map_T_center.dot(center_P_bf_goal)
        map_P_bf = map_T_bf.to_position()
        # self.add_debug_expr('map_P_bf_goal', map_P_bf_goal)
        self.add_point_goal_constraints(frame_P_current=map_P_bf,
                                        frame_P_goal=map_P_bf_goal,
                                        reference_velocity=0.1,
                                        weight=WEIGHT_BELOW_CA,
                                        name='position')

    def __str__(self) -> str:
        return super().__str__()
