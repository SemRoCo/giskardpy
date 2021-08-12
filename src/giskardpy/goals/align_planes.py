from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
from giskardpy import casadi_wrapper as w
import giskardpy.utils.tfwrapper as tf

class AlignPlanes(Goal):
    def __init__(self, root_link, tip_link, root_normal, tip_normal,
                 max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA, **kwargs):
        """
        This Goal will use the kinematic chain between tip and root normal to align both
        :param root_link: str, name of the root link for the kinematic chain
        :param tip_link: str, name of the tip link for the kinematic chain
        :param tip_normal: Vector3Stamped as json, normal at the tip of the kin chain
        :param root_normal: Vector3Stamped as json, normal at the root of the kin chain
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default is WEIGHT_ABOVE_CA
        :param goal_constraint: bool, default False
        """
        self.root = root_link
        self.tip = tip_link
        self.max_velocity = max_angular_velocity
        self.weight = weight

        self.tip_V_tip_normal = tf.transform_vector(self.tip, tip_normal)
        self.tip_V_tip_normal.vector = tf.normalize(self.tip_V_tip_normal.vector)

        self.root_V_root_normal = tf.transform_vector(self.root, root_normal)
        self.root_V_root_normal.vector = tf.normalize(self.root_V_root_normal.vector)

        super(AlignPlanes, self).__init__(**kwargs)

    def __str__(self):
        s = super(AlignPlanes, self).__str__()
        return u'{}/{}/{}_X:{}_Y:{}_Z:{}'.format(s, self.root, self.tip,
                                                 self.tip_V_tip_normal.vector.x,
                                                 self.tip_V_tip_normal.vector.y,
                                                 self.tip_V_tip_normal.vector.z)

    def make_constraints(self):
        max_velocity = self.get_parameter_as_symbolic_expression(u'max_velocity')
        tip_V_tip_normal = self.get_parameter_as_symbolic_expression(u'tip_V_tip_normal')
        root_V_root_normal = self.get_parameter_as_symbolic_expression(u'root_V_root_normal')
        weight = self.get_parameter_as_symbolic_expression(u'weight')
        self.add_minimize_vector_angle_constraints(max_velocity=max_velocity,
                                                   root=self.root,
                                                   tip=self.tip,
                                                   tip_V_tip_normal=tip_V_tip_normal,
                                                   root_V_goal_normal=root_V_root_normal,
                                                   weight=weight)

