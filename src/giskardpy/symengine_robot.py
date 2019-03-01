import hashlib
from collections import namedtuple, OrderedDict
import symengine_wrappers as spw

from giskardpy import WorldObjImpl
from giskardpy.qp_problem_builder import HardConstraint, JointConstraint
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import cube_volume, cube_surface, sphere_volume, cylinder_volume, cylinder_surface, keydefaultdict, \
    suppress_stdout, suppress_stderr

Joint = namedtuple('Joint', ['symbol', 'velocity_limit', 'lower', 'upper', 'type', 'frame'])

class Robot(WorldObjImpl):
    def __init__(self, urdf, default_joint_vel_limit, default_joint_weight, controlled_joints):
        """
        :param urdf:
        :type urdf: str
        :param joints_to_symbols_map: maps urdf joint names to symbols
        :type joints_to_symbols_map: dict
        :param default_joint_vel_limit: all velocity limits which are undefined or higher than this will be set to this
        :type default_joint_vel_limit: Symbol
        """
        super(Robot, self).__init__(urdf, controlled_joints)
        self.default_joint_velocity_limit = default_joint_vel_limit
        self.default_weight = default_joint_weight
        self.fks = {}
        self._joint_to_frame = {}
        self.joint_to_symbol_map = keydefaultdict(lambda x: spw.Symbol(x))

    def reinitialize(self, joints_to_symbols_map=None):
        """
        :param joints_to_symbols_map: maps urdf joint names to symbols
        :type joints_to_symbols_map: dict
        """
        if joints_to_symbols_map is not None:
            self.joint_to_symbol_map.update(joints_to_symbols_map)
        self._create_frames_expressions()
        self._create_constraints()

    def _create_frames_expressions(self):
        for joint_name, urdf_joint in self._urdf_robot.joint_map.items():
            if self.is_joint_controllable(joint_name):
                joint_symbol = self.get_joint_symbol(joint_name)
            if self.is_joint_mimic(joint_name):
                multiplier = 1 if urdf_joint.mimic.multiplier is None else urdf_joint.mimic.multiplier
                offset = 0 if urdf_joint.mimic.offset is None else urdf_joint.mimic.offset
                joint_symbol = self.get_joint_symbol(urdf_joint.mimic.joint) * multiplier + offset

            if self.is_joint_type_supported(joint_name):
                if urdf_joint.origin is not None:
                    xyz = urdf_joint.origin.xyz if urdf_joint.origin.xyz is not None else [0, 0, 0]
                    rpy = urdf_joint.origin.rpy if urdf_joint.origin.rpy is not None else [0, 0, 0]
                    joint_frame = spw.translation3(*xyz) * spw.rotation_matrix_from_rpy(*rpy)
                else:
                    joint_frame = spw.eye(4)
            else:
                # TODO more specific exception
                raise Exception(u'Joint type "{}" is not supported by urdf parser.'.format(urdf_joint.type))

            if self.is_rotational_joint(joint_name):
                joint_frame *= spw.rotation_matrix_from_axis_angle(spw.vector3(*urdf_joint.axis), joint_symbol)
            elif self.is_translational_joint(joint_name):
                joint_frame *= spw.translation3(*(spw.point3(*urdf_joint.axis) * joint_symbol)[:3])

            self._joint_to_frame[joint_name] = joint_frame

    def _create_constraints(self):
        """
        Creates hard and joint constraints.
        """
        self.hard_constraints = OrderedDict()
        self.joint_constraints = OrderedDict()
        for i, joint_name in enumerate(self.get_joint_names_controllable()):
            lower_limit, upper_limit = self.get_joint_limits(joint_name)
            joint_symbol = self.get_joint_symbol(joint_name)
            velocity_limit = self.get_joint_velocity_limit_expr(joint_name)

            if lower_limit is not None and upper_limit is not None:
                self.hard_constraints[joint_name] = HardConstraint(lower=lower_limit - joint_symbol,
                                                                   upper=upper_limit - joint_symbol,
                                                                   expression=joint_symbol)

            self.joint_constraints[joint_name] = JointConstraint(lower=-velocity_limit,
                                                                 upper=velocity_limit,
                                                                 weight=self.default_weight)

    def get_fk_expression(self, root_link, tip_link):
        """
        :type root_link: str
        :type tip_link: str
        :return: 4d matrix describing the transformation from root_link to tip_link
        :rtype: spw.Matrix
        """
        if (root_link, tip_link) not in self.fks:
            fk = spw.eye(4)
            for joint_name in self.get_joint_names_from_chain(root_link, tip_link):
                fk *= self.get_joint_frame(joint_name)
            self.fks[root_link, tip_link] = fk
        return self.fks[root_link, tip_link]

    # JOINT FUNCTIONS

    def get_joint_symbols(self):
        """
        :return: dict mapping urdf joint name to symbol
        :rtype: dict
        """
        return {joint_name: self.get_joint_symbol(joint_name) for joint_name in self.get_joint_names_controllable()}

    def get_joint_velocity_limit_expr(self, joint_name):
        """
        :param joint_name: name of the joint in the urdf
        :type joint_name: str
        :return: minimum of default velocity limit and limit specified in urdf
        :rtype: float
        """
        limit = self._urdf_robot.joint_map[joint_name].limit
        if limit is None or limit.velocity is None:
            return self.default_joint_velocity_limit
        else:
            return spw.Min(limit.velocity, self.default_joint_velocity_limit)

    def get_joint_frame(self, joint_name):
        """
        :param joint_name: name of the joint in the urdf
        :type joint_name: str
        :return: matrix expression describing the transformation caused by this joint
        :rtype: spw.Matrix
        """
        return self._joint_to_frame[joint_name]

    def get_joint_symbol(self, joint_name):
        """
        :param joint_name: name of the joint in the urdf
        :type joint_name: str
        :rtype: spw.Symbol
        """
        return self.joint_to_symbol_map[joint_name]