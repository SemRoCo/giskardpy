from collections import namedtuple, OrderedDict
from time import time

from tf.transformations import quaternion_from_matrix
from urdf_parser_py.urdf import URDF

from giskardpy import USE_SYMENGINE
from giskardpy.input_system import ControllerInputArray
from giskardpy.qp_problem_builder import HardConstraint, JointConstraint
import numpy as np

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw

Joint = namedtuple('Joint', ['symbol', 'velocity_limit', 'lower', 'upper', 'limitless'])

def hacky_urdf_parser_fix(urdf_str):
    fixed_urdf = ''
    delete = False
    black_list = ['transmission']
    black_open = ['<{}'.format(x) for x in black_list]
    black_close = ['</{}'.format(x) for x in black_list]
    for line in urdf_str.split('\n'):
        if len([x for x in black_open if x in line]) > 0:
            delete = True
        if len([x for x in black_close if x in line]) > 0:
            delete = False
            continue
        if not delete:
            fixed_urdf += line + '\n'
    return fixed_urdf

class Robot(object):
    # TODO add joint vel to internal state
    def __init__(self, default_joint_value=0.0, default_joint_weight=0.001, urdf_str=None, root_link='base_footprint',
                 tip_links=()):
        self.root_link = None
        self.default_joint_value = default_joint_value
        self.default_joint_weight = default_joint_weight
        self.urdf_robot = None
        self._joints = OrderedDict()
        self.default_joint_vel_limit = 0.5

        self.frames = {}
        self._state = OrderedDict()
        self.hard_constraints = OrderedDict()
        self.joint_constraints = OrderedDict()
        if urdf_str is not None:
            urdf = hacky_urdf_parser_fix(urdf_str)
            self.load_from_urdf_string(urdf, root_link, tip_links)
            for joint_name in self.weight_input.get_float_names():
                self.set_joint_weight(joint_name, default_joint_weight)

    def get_state(self):
        return self._state

    def get_joint_state(self):
        return OrderedDict((k, v) for k, v in self.get_state().items() if k in self.get_joint_names())

    def get_joint_names(self):
        return self._joints.keys()

    def _update_observables(self, updates):
        self._state.update(updates)

    def set_joint_state(self, new_joint_state):
        self._update_observables(self.joint_states_input.get_update_dict(**new_joint_state))

    def set_joint_weight(self, joint_name, weight):
        self._update_observables(self.weight_input.get_update_dict(**{joint_name: weight}))

    def get_joint_state_input(self):
        return self.joint_states_input

    # @profile
    def add_chain_joints(self, root_link, tip_link):
        """
        Returns a dict with joint names as keys and sympy symbols
        as values for all 1-dof movable robot joints in URDF between
        ROOT_LINK and TIP_LINK.

        :param root_link: str, denoting the root of the kin. chain
        :param tip_link: str, denoting the tip of the kin. chain
        :return: dict{str, sympy.Symbol}, with symbols for all joints in chain
        """

        jointsAndLinks = self.urdf_robot.get_chain(root_link, tip_link, True, True, True)
        parentFrame = self.frames[root_link]
        for i in range(1, len(jointsAndLinks), 2):
            joint_name = jointsAndLinks[i]
            link_name = jointsAndLinks[i + 1]
            joint = self.urdf_robot.joint_map[joint_name]

            if joint_name not in self._joints:
                if joint.type in ['fixed', 'revolute', 'continuous', 'prismatic']:
                    self.frames[link_name] = parentFrame
                    self.frames[link_name] *= spw.translation3(*joint.origin.xyz)
                    self.frames[link_name] *= spw.rotation3_rpy(*joint.origin.rpy)
                else:
                    raise Exception('Joint type "{}" is not supported by urdf parser.'.format(joint.type))

                if joint.type != 'fixed':
                    self._joints[joint_name] = Joint(spw.Symbol(joint_name),
                                                     min(joint.limit.velocity, self.default_joint_vel_limit),
                                                     joint.limit.lower,
                                                     joint.limit.upper,
                                                     joint.type == 'continuous')

                if joint.type == 'revolute' or joint.type == 'continuous':
                    self.frames[link_name] *= spw.rotation3_axis_angle(spw.vec3(*joint.axis), spw.Symbol(joint_name))

                elif joint.type == 'prismatic':
                    self.frames[link_name] *= spw.translation3(*(spw.point3(*joint.axis) * spw.Symbol(joint_name))[:3])

            parentFrame = self.frames[link_name]

    # @profile
    def load_from_urdf(self, urdf_robot, root_link, tip_links, root_frame=None):
        """
        Returns a dict with joint names as keys and sympy symbols
        as values for all 1-dof movable robot joints in URDF between
        ROOT_LINK and TIP_LINKS.

        :param urdf_robot: URDF.Robot, obtained from URDF parser.
        :param root_link: str, denoting the root of the kin. tree
        :param tip_links: str, denoting the tips of the kin. tree
        :return: dict{str, sympy.Symbol}, with symbols for all joints in tree
        """
        self.urdf_robot = urdf_robot
        self.root_link = root_link

        self.frames[root_link] = root_frame if root_frame is not None else spw.eye(4)
        self.end_effectors = tip_links
        t = time()
        for tip_link in tip_links:
            self.add_chain_joints(root_link, tip_link)
        print('add chain joints took {}'.format(time() - t))

        self.joint_states_input = ControllerInputArray(self.get_joint_names())
        self.weight_input = ControllerInputArray(self.get_joint_names(), suffix='cc_weight')

        for i, (joint_name, joint) in enumerate(self._joints.items()):
            joint_symbol = self.joint_states_input.to_symbol(joint_name)
            weight_symbol = self.weight_input.to_symbol(joint_name)
            self._state[joint_name] = self.default_joint_value
            self._state[self.weight_input.to_str_symbol(joint_name)] = self.default_joint_weight

            if not joint.limitless:
                self.hard_constraints[joint_name] = HardConstraint(lower=joint.lower - joint_symbol,
                                                                   upper=joint.upper - joint_symbol,
                                                                   expression=joint_symbol)

            self.joint_constraints[joint_name] = JointConstraint(lower=-joint.velocity_limit,
                                                                 upper=joint.velocity_limit,
                                                                 weight=weight_symbol)
        t = time()
        self.make_np_frames()
        print('make np frame took {}'.format(time() - t))

    # @profile
    def make_np_frames(self):
        self.fast_frames = []
        for f, expression in self.frames.items():
            self.fast_frames.append((f, spw.speed_up(expression, list(expression.free_symbols))))
        self.fast_frames = OrderedDict(self.fast_frames)

    def get_eef_position(self):
        eef = {}
        for end_effector in self.end_effectors:
            eef_joints = self.frames[end_effector].free_symbols
            eef_joint_symbols = [self.get_joint_state_input().to_str_symbol(str(x)) for x in eef_joints]
            js = {k: self.get_state()[k] for k in eef_joint_symbols}
            evaled_frame = self.fast_frames[end_effector](**js)
            # eef_pos = np.array(pos_of(evaled_frame).tolist(), dtype=float)[:-1].reshape(3)
            # eef_rot = np.array(rot_of(evaled_frame).tolist(), dtype=float)
            # eef_rot = quaternion_from_matrix(eef_rot)
            eef[end_effector] = spw.Matrix(evaled_frame.tolist())
        return eef

    def get_eef_position2(self):
        eef = self.get_eef_position()
        for eef_name, pose in eef.items():
            evaled_frame = np.array(pose).astype(float)
            eef_pos = evaled_frame[:3, 3]
            eef_rot = evaled_frame[:4, :3]
            eef_rot = np.hstack((eef_rot, np.zeros((4, 1))))
            eef_rot[3, 3] = 1
            eef_rot = quaternion_from_matrix(eef_rot)
            eef[eef_name] = np.concatenate((eef_rot, eef_pos))
        return eef

    def load_from_urdf_path(self, urdf_path, root_link, tip_links):
        return self.load_from_urdf(URDF.from_xml_file(urdf_path), root_link, tip_links)

    def load_from_urdf_string(self, urdf_strg, root_link, tip_links):
        return self.load_from_urdf(URDF.from_xml_string(urdf_strg), root_link, tip_links)

    def get_name(self):
        return self.__class__.__name__

    def __str__(self):
        return "{}'s state:\n{}".format(self.get_name(),
                                        '\n'.join('{}:{:.3f}'.format(joint_name, value) for joint_name, value in
                                                  self.get_state().items()))
