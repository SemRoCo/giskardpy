import symengine_wrappers as spw
from collections import OrderedDict

from giskardpy.input_system import JointStatesInput, FrameInput
from giskardpy.qp_problem_builder import QProblemBuilder, SoftConstraint
from giskardpy.robot_constraints import Robot


class Controller(object):
    def __init__(self, urdf):
        self._soft_constraints = OrderedDict()
        self.robot = Robot(urdf)

    def init(self, *args, **kwargs):
        controlled_joints = list(self.robot.joint_states_input.joint_map.values())
        self.qp_problem_builder = QProblemBuilder(self.robot.joint_constraints,
                                                  self.robot.hard_constraints,
                                                  self._soft_constraints,
                                                  controlled_joints)

    def get_cmd(self, substitutions):
        return self.qp_problem_builder.get_cmd(substitutions)


class JointController(Controller):
    def __init__(self, urdf):
        super(JointController, self).__init__(urdf)
        # TODO use symbol for weight
        self.default_weight = 1
        self._set_default_goal_joint_states()

    def _set_default_goal_joint_states(self):
        m = OrderedDict()
        for joint_name in self.robot.joint_states_input.joint_map:
            m[joint_name] = spw.Symbol('goal_{}'.format(joint_name))
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


class CartesianController(Controller):
    def __init__(self, urdf):
        super(CartesianController, self).__init__(urdf)
        # TODO use symbol for weight
        self.default_weight = 1
        self._set_default_inputs()
        self.default_trans_gain = 3
        self.max_trans_speed = 0.3
        self.max_rot_speed = 0.5  # rad/s
        self.default_rot_gain = 3

    def _set_default_inputs(self):
        self.start_pose = FrameInput('start_x',
                                     'start_y',
                                     'start_z',
                                     'start_qx',
                                     'start_qy',
                                     'start_qz',
                                     'start_qw')
        self.goal_pose = FrameInput('goal_x',
                                    'goal_y',
                                    'goal_z',
                                    'goal_qx',
                                    'goal_qy',
                                    'goal_qz',
                                    'goal_qw')
        self.current_evaluated = FrameInput('current_evaled_x',
                                            'current_evaled_y',
                                            'current_evaled_z',
                                            'current_evaled_qx',
                                            'current_evaled_qy',
                                            'current_evaled_qz',
                                            'current_evaled_qw')

    def init(self, root, tip, current_joints=None, start_pose=None, goal_pose=None, current_pose=None,
             current_evaluated=None):
        """
        :param current_joints: InputArray
        :param joint_goals: InputArray
        """

        # TODO add chain
        self.robot.set_joint_symbol_map(current_joints)
        if start_pose is not None:
            self.start_pose = start_pose
        if goal_pose is not None:
            self.goal_pose = goal_pose
        if current_evaluated is not None:
            self.current_evaluated = current_evaluated

        self._soft_constraints = self.make_soft_constraints(self.start_pose.get_frame(),
                                                            self.goal_pose.get_frame(),
                                                            self.robot.get_fk_expression(root, tip),
                                                            self.current_evaluated.get_frame())
        super(CartesianController, self).init()

    def make_soft_constraints(self, start_pose, goal_pose, current_pose, current_evaluated, weights=(1, 1, 1, 1, 1, 1)):
        """
        :param start_pose: FrameInput
        :param goal_pose: FrameInput
        :param current_pose:
        :param current_evaluated: FrameInput
        :param weights:
        :return:
        """
        soft_constraints = {}

        start_position = spw.pos_of(start_pose)
        start_rotation = spw.rot_of(start_pose)

        current_position = spw.pos_of(current_pose)
        current_rotation = spw.rot_of(current_pose)

        current_evaluated_position = spw.pos_of(start_pose)
        current_evaluated_rotation = spw.rot_of(start_pose)

        goal_position = spw.pos_of(goal_pose)
        goal_rotation = spw.rot_of(goal_pose)

        # position control
        trans_error_vector = goal_position - current_position
        trans_error = spw.norm(trans_error_vector)
        trans_scale = spw.fake_Min(trans_error * self.default_trans_gain, self.max_trans_speed)
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

        axis, angle = spw.axis_angle_from_matrix((current_rotation.T * goal_rotation))
        capped_angle = spw.fake_Min(angle * self.default_rot_gain, self.max_rot_speed)
        axis = axis
        r_rot_control = axis * capped_angle

        hack = spw.rotation3_axis_angle([0, 0, 1], 0.0001)

        axis, angle = spw.axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
        c_aa = (axis * angle)

        soft_constraints['align {} rotation 0'.format(0)] = SoftConstraint(lower=r_rot_control[0],
                                                                           upper=r_rot_control[0],
                                                                           weight=weights[3],
                                                                           expression=c_aa[0])
        soft_constraints['align {} rotation 1'.format(0)] = SoftConstraint(lower=r_rot_control[1],
                                                                           upper=r_rot_control[1],
                                                                           weight=weights[4],
                                                                           expression=c_aa[1])
        soft_constraints['align {} rotation 2'.format(0)] = SoftConstraint(lower=r_rot_control[2],
                                                                           upper=r_rot_control[2],
                                                                           weight=weights[5],
                                                                           expression=c_aa[2])

        return soft_constraints
