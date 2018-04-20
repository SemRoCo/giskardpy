import symengine_wrappers as sw
from collections import OrderedDict

from giskardpy.input_system import JointStatesInput, FrameInput, Point3Input
from giskardpy.qp_problem_builder import QProblemBuilder, SoftConstraint
from giskardpy.symengine_robot import Robot


class Controller(object):
    def __init__(self, urdf):
        self._soft_constraints = OrderedDict()
        self.robot = Robot(urdf)

    def add_constraints(self, soft_constraints):
        self._soft_constraints.update(soft_constraints)

    def init(self, controlled_joints=None):
        if controlled_joints is None:
            controlled_joint_symbols = self.get_controlled_joint_symbols()
            controlled_joints = self.get_controlled_joints()
        else:
            controlled_joint_symbols = [self.robot.get_joint_symbol_map().joint_map[x] for x in controlled_joints]
        self.qp_problem_builder = QProblemBuilder({k: self.robot.joint_constraints[k] for k in controlled_joints},
                                                  {k: self.robot.hard_constraints[k] for k in controlled_joints if
                                                   k in self.robot.hard_constraints},
                                                  self._soft_constraints,
                                                  controlled_joint_symbols)

    def get_controlled_joints(self):
        return list(self.robot.joint_states_input.joint_map.keys())

    def get_controlled_joint_symbols(self):
        return list(self.robot.joint_states_input.joint_map.values())

    def get_cmd(self, substitutions):
        next_cmd = self.qp_problem_builder.get_cmd(substitutions)
        real_next_cmd = {}
        for joint_name in self.get_controlled_joints():
            joint_expr = str(self.robot.joint_states_input.joint_map[joint_name])
            if joint_expr in next_cmd:
                real_next_cmd[joint_name] = next_cmd[joint_expr]
        return real_next_cmd


class JointController(Controller):
    def __init__(self, urdf):
        super(JointController, self).__init__(urdf)
        # TODO use symbol for weight
        self.default_weight = 1
        self._set_default_goal_joint_states()

    def _set_default_goal_joint_states(self):
        m = OrderedDict()
        for joint_name in self.robot.joint_states_input.joint_map:
            m[joint_name] = sw.Symbol('goal_{}'.format(joint_name))
        self.goal_joint_states = JointStatesInput(m)

    def init(self, current_joints=None, joint_goals=None):
        """
        :param current_joints: InputArray
        :param joint_goals: InputArray
        """

        # TODO add chain
        self.robot.set_joint_symbol_map(current_joints)
        if joint_goals is not None:
            self.goal_joint_states = joint_goals

        self._soft_constraints = self.make_soft_constraints(self.robot.joint_states_input, self.goal_joint_states, 0)
        super(JointController, self).init()

    def make_soft_constraints(self, current_joints, joint_goals, weights):
        """
        :param current_joints: JointStatesInput
        :param joint_goals: JointStatesInput
        :param weights: TODO
        :return:
        """
        soft_constraints = {}
        for joint_name, current_joint_symbol in current_joints.joint_map.items():
            sc = SoftConstraint(lower=joint_goals.joint_map[joint_name] - current_joint_symbol,
                                upper=joint_goals.joint_map[joint_name] - current_joint_symbol,
                                weight=1,
                                expression=current_joint_symbol)
            soft_constraints[joint_name] = sc
        return soft_constraints


# TODO
# def get_controlled_joint_symbols(self):
#     return self.robot.get_chain_joint_symbols(self.root, self.tip)
def position_conv(goal_position, current_position, weights=(1, 1, 1), trans_gain=3, max_trans_speed=0.3):
    soft_constraints = {}

    # position control
    trans_error_vector = goal_position - current_position
    trans_error = sw.norm(trans_error_vector)
    trans_scale = sw.fake_Min(trans_error * trans_gain, max_trans_speed)
    trans_control = trans_error_vector / trans_error * trans_scale

    soft_constraints['align {} x position'.format(0)] = SoftConstraint(lower=trans_control[0],
                                                                       upper=trans_control[0],
                                                                       weight=weights[0],
                                                                       expression=current_position[0])
    soft_constraints['align {} y position'.format(0)] = SoftConstraint(lower=trans_control[1],
                                                                       upper=trans_control[1],
                                                                       weight=weights[1],
                                                                       expression=current_position[1])
    soft_constraints['align {} z position'.format(0)] = SoftConstraint(lower=trans_control[2],
                                                                       upper=trans_control[2],
                                                                       weight=weights[2],
                                                                       expression=current_position[2])

    return soft_constraints


def rotation_conv(goal_rotation, current_rotation, current_evaluated_rotation, weights=(1, 1, 1),
                  rot_gain=3, max_rot_speed=0.5):
    soft_constraints = {}
    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * goal_rotation))
    capped_angle = sw.fake_Min(angle * rot_gain, max_rot_speed)
    axis = axis
    r_rot_control = axis * capped_angle

    hack = sw.rotation3_axis_angle([0, 0, 1], 0.0001)

    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
    c_aa = (axis * angle)

    soft_constraints['align {} rotation 0'.format(0)] = SoftConstraint(lower=r_rot_control[0],
                                                                       upper=r_rot_control[0],
                                                                       weight=weights[0],
                                                                       expression=c_aa[0])
    soft_constraints['align {} rotation 1'.format(0)] = SoftConstraint(lower=r_rot_control[1],
                                                                       upper=r_rot_control[1],
                                                                       weight=weights[1],
                                                                       expression=c_aa[1])
    soft_constraints['align {} rotation 2'.format(0)] = SoftConstraint(lower=r_rot_control[2],
                                                                       upper=r_rot_control[2],
                                                                       weight=weights[2],
                                                                       expression=c_aa[2])
    return soft_constraints


def link_to_any_avoidance(link_name, current_pose, current_pose_eval, point_on_link, other_point, lower_limit=0.03,
                          upper_limit=1e9, weight=(100, 10, 10)):
    soft_constraints = {}
    name = '{} to any collision'.format(link_name)

    dist = sw.euclidean_distance((current_pose * sw.inverse_frame(current_pose_eval) * point_on_link), other_point)
    soft_constraints['{} x'.format(name)] = SoftConstraint(lower=lower_limit - dist,
                                                           upper=upper_limit,
                                                           weight=weight[0],
                                                           expression=dist)

    # controllable_distance = (current_pose * sw.inverse_frame(current_pose_eval) * point_on_link) - other_point
    # lower_limit = controllable_distance / sw.norm(controllable_distance) * lower_limit
    # soft_constraints['{} x'.format(name)] = SoftConstraint(lower=lower_limit[0],
    #                                                        upper=upper_limit,
    #                                                        weight=weight[0],
    #                                                        expression=controllable_distance[0])
    # soft_constraints['{} y'.format(name)] = SoftConstraint(lower=lower_limit[1],
    #                                                        upper=upper_limit,
    #                                                        weight=weight[1],
    #                                                        expression=controllable_distance[1])
    # soft_constraints['{} z'.format(name)] = SoftConstraint(lower=lower_limit[2],
    #                                                        upper=upper_limit,
    #                                                        weight=weight[2],
    #                                                        expression=controllable_distance[2])
    return soft_constraints
