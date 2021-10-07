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
        super(AlignPlanes, self).__init__(**kwargs)
        self.root = root_link
        self.tip = tip_link
        self.max_velocity = max_angular_velocity
        self.weight = weight

        self.tip_V_tip_normal = self.transform_msg(self.tip, tip_normal)
        self.tip_V_tip_normal.vector = tf.normalize(self.tip_V_tip_normal.vector)

        self.root_V_root_normal = self.transform_msg(self.root, root_normal)
        self.root_V_root_normal.vector = tf.normalize(self.root_V_root_normal.vector)


    def __str__(self):
        s = super(AlignPlanes, self).__str__()
        return u'{}/{}/{}_X:{}_Y:{}_Z:{}'.format(s, self.root, self.tip,
                                                 self.tip_V_tip_normal.vector.x,
                                                 self.tip_V_tip_normal.vector.y,
                                                 self.tip_V_tip_normal.vector.z)

    def make_constraints(self):
        tip_V_tip_normal = self.get_parameter_as_symbolic_expression(u'tip_V_tip_normal')
        root_R_tip = w.rotation_of(self.get_fk(self.root, self.tip))
        root_V_tip_normal = w.dot(root_R_tip, tip_V_tip_normal)
        root_V_root_normal = self.get_parameter_as_symbolic_expression(u'root_V_root_normal')
        self.add_vector_goal_constraints(frame_V_current=root_V_tip_normal,
                                         frame_V_goal=root_V_root_normal,
                                         reference_velocity=self.max_velocity,
                                         weight=self.weight)

