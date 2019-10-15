from collections import namedtuple, OrderedDict, defaultdict
from itertools import combinations

from geometry_msgs.msg import PoseStamped

import symengine_wrappers as spw
from giskardpy import WORLD_IMPLEMENTATION
from giskardpy.pybullet_world_object import PyBulletWorldObject
from giskardpy.qp_problem_builder import HardConstraint, JointConstraint
from giskardpy.utils import KeyDefaultDict, \
    homo_matrix_to_pose
from giskardpy.world_object import WorldObject

Joint = namedtuple(u'Joint', [u'symbol', u'velocity_limit', u'lower', u'upper', u'type', u'frame'])

if WORLD_IMPLEMENTATION == u'pybullet':
    Backend = PyBulletWorldObject
else:
    Backend = WorldObject


class Robot(Backend):
    def __init__(self, urdf, joint_weights=None, base_pose=None, controlled_joints=None, path_to_data_folder=u'',
                 joint_vel_limit=None,
                 joint_acc_limit=None, calc_self_collision_matrix=True, symengine_backend='llvm', symengine_opt_level=0,
                 *args, **kwargs):
        """
        :param urdf:
        :type urdf: str
        :param joints_to_symbols_map: maps urdfs joint names to symbols
        :type joints_to_symbols_map: dict
        :param joint_vel_limit: all velocity limits which are undefined or higher than this will be set to this
        :type joint_vel_limit: Symbol
        """
        self.symengine_backend = symengine_backend
        self.symengine_opt_level = symengine_opt_level
        self._fk_expressions = {}
        self._fks = {}
        self._evaluated_fks = {}
        self._joint_to_frame = {}
        if joint_weights is None:
            self._joint_weights = defaultdict(lambda: 0)
        else:
            self._joint_weights = joint_weights
        if joint_vel_limit is None:
            self._joint_velocity_limit = defaultdict(lambda: 100)  # use urdf limits per default
        else:
            self._joint_velocity_limit = joint_vel_limit

        if joint_acc_limit is None:
            self._joint_acc_limit = defaultdict(lambda: 100)  # no acceleration limit per default
        else:
            self._joint_acc_limit = joint_acc_limit
        self._joint_position_symbols = KeyDefaultDict(lambda x: spw.Symbol(x))  # don't iterate over this map!!
        self._joint_velocity_symbols = KeyDefaultDict(lambda x: 0)  # don't iterate over this map!!
        super(Robot, self).__init__(urdf, base_pose, controlled_joints, path_to_data_folder, calc_self_collision_matrix,
                                    *args, **kwargs)
        self.reinitialize()
        self.update_self_collision_matrix(added_links=set(combinations(self.get_link_names_with_collision(), 2)))

    @property
    def hard_constraints(self):
        return self._hard_constraints

    @property
    def joint_constraints(self):
        return self._joint_constraints

    @Backend.joint_state.setter
    def joint_state(self, value):
        """
        :param joint_state:
        :type joint_state: dict
        :return:
        """
        Backend.joint_state.fset(self, value)
        self.__joint_state_positions = {str(self._joint_position_symbols[k]): v.position for k, v in
                                        self.joint_state.items()}
        self._evaluated_fks.clear()

    def get_joint_state_positions(self):
        try:
            return self.__joint_state_positions
        except:
            return {str(self._joint_position_symbols[x]): 0 for x in self.get_controllable_joints()}

    def reinitialize(self, joint_position_symbols=None, joint_vel_symbols=None):
        """
        :param joint_position_symbols: maps urdfs joint names to symbols
        :type joint_position_symbols: dict
        """
        super(Robot, self).reinitialize()
        if joint_position_symbols is not None:
            # TODO don't change the input parameter
            self._joint_position_symbols = joint_position_symbols
            self._joint_velocity_symbols = joint_vel_symbols
        self._fk_expressions = {}
        self._create_frames_expressions()
        self._create_constraints()
        self.init_fast_fks()

    def update_self_collision_matrix(self, added_links=None, removed_links=None):
        if self._calc_self_collision_matrix:
            super(Robot, self).update_self_collision_matrix(added_links, removed_links)

    def _create_frames_expressions(self):
        for joint_name, urdf_joint in self._urdf_robot.joint_map.items():
            if self.is_joint_controllable(joint_name):
                joint_symbol = self.get_joint_position_symbol(joint_name)
            if self.is_joint_mimic(joint_name):
                multiplier = 1 if urdf_joint.mimic.multiplier is None else urdf_joint.mimic.multiplier
                offset = 0 if urdf_joint.mimic.offset is None else urdf_joint.mimic.offset
                joint_symbol = self.get_joint_position_symbol(urdf_joint.mimic.joint) * multiplier + offset

            if self.is_joint_type_supported(joint_name):
                if urdf_joint.origin is not None:
                    xyz = urdf_joint.origin.xyz if urdf_joint.origin.xyz is not None else [0, 0, 0]
                    rpy = urdf_joint.origin.rpy if urdf_joint.origin.rpy is not None else [0, 0, 0]
                    joint_frame = spw.translation3(*xyz) * spw.rotation_matrix_from_rpy(*rpy)
                else:
                    joint_frame = spw.eye(4)
            else:
                # TODO more specific exception
                raise TypeError(u'Joint type "{}" is not supported by urdfs parser.'.format(urdf_joint.type))

            if self.is_rotational_joint(joint_name):
                joint_frame *= spw.rotation_matrix_from_axis_angle(spw.vector3(*urdf_joint.axis), joint_symbol)
            elif self.is_translational_joint(joint_name):
                joint_frame *= spw.translation3(*(spw.point3(*urdf_joint.axis) * joint_symbol)[:3])

            self._joint_to_frame[joint_name] = joint_frame

    def _create_constraints(self):
        """
        Creates hard and joint constraints.
        """
        self._hard_constraints = OrderedDict()
        self._joint_constraints = OrderedDict()
        for i, joint_name in enumerate(self.get_joint_names_controllable()):
            lower_limit, upper_limit = self.get_joint_limits(joint_name)
            joint_symbol = self.get_joint_position_symbol(joint_name)
            velocity_limit = self.get_joint_velocity_limit_expr(joint_name)

            if not self.is_joint_continuous(joint_name):
                self._hard_constraints[joint_name] = HardConstraint(lower=lower_limit - joint_symbol,
                                                                    upper=upper_limit - joint_symbol,
                                                                    expression=joint_symbol)

                self._joint_constraints[joint_name] = JointConstraint(lower=spw.Max(lower_limit - joint_symbol,
                                                                                    spw.Max(-velocity_limit,
                                                                                            self.get_joint_velocity_symbol(
                                                                                                joint_name) -
                                                                                            self._joint_acc_limit[
                                                                                                joint_name])),
                                                                      upper=spw.Min(upper_limit - joint_symbol,
                                                                                    spw.Min(velocity_limit,
                                                                                            self.get_joint_velocity_symbol(
                                                                                                joint_name) +
                                                                                            self._joint_acc_limit[
                                                                                                joint_name])),
                                                                      weight=self._joint_weights[joint_name])
            else:
                self._joint_constraints[joint_name] = JointConstraint(lower=spw.Max(-velocity_limit,
                                                                                    self.get_joint_velocity_symbol(
                                                                                        joint_name) -
                                                                                    self._joint_acc_limit[
                                                                                        joint_name]),
                                                                      upper=spw.Min(velocity_limit,
                                                                                    self.get_joint_velocity_symbol(
                                                                                        joint_name) +
                                                                                    self._joint_acc_limit[
                                                                                        joint_name]),
                                                                      weight=self._joint_weights[joint_name])

    def get_fk_expression(self, root_link, tip_link):
        """
        :type root_link: str
        :type tip_link: str
        :return: 4d matrix describing the transformation from root_link to tip_link
        :rtype: spw.Matrix
        """
        if (root_link, tip_link) not in self._fk_expressions:
            fk = spw.eye(4)
            root_chain, _, tip_chain = self.get_split_chain(root_link, tip_link, links=False)
            for joint_name in root_chain:
                fk *= spw.inverse_frame(self.get_joint_frame(joint_name))
            for joint_name in tip_chain:
                fk *= self.get_joint_frame(joint_name)
            self._fk_expressions[root_link, tip_link] = fk
        return self._fk_expressions[root_link, tip_link]

    def get_fk_pose(self, root, tip):
        homo_m = self.get_fk_np(root, tip)
        p = PoseStamped()
        p.header.frame_id = root
        p.pose = homo_matrix_to_pose(homo_m)
        return p

    def get_fk_np(self, root, tip):
        try:
            return self._evaluated_fks[root, tip]
        except KeyError:
            homo_m = self._fks[root, tip](**self.get_joint_state_positions())
            self._evaluated_fks[root, tip] = homo_m
            return homo_m

    def init_fast_fks(self):
        def f(key):
            root, tip = key
            fk = self.get_fk_expression(root, tip)
            m = spw.speed_up(fk, fk.free_symbols, backend=self.symengine_backend, opt_level=self.symengine_opt_level)
            return m

        self._fks = KeyDefaultDict(f)

    # JOINT FUNCTIONS

    def get_joint_symbols(self):
        """
        :return: dict mapping urdfs joint name to symbol
        :rtype: dict
        """
        return {joint_name: self.get_joint_position_symbol(joint_name) for joint_name in
                self.get_joint_names_controllable()}

    def get_joint_velocity_limit_expr(self, joint_name):
        """
        :param joint_name: name of the joint in the urdfs
        :type joint_name: str
        :return: minimum of default velocity limit and limit specified in urdfs
        :rtype: float
        """
        limit = self._urdf_robot.joint_map[joint_name].limit
        t = spw.Symbol(u'rosparam_sample_period')  # TODO this should be a parameter
        if limit is None or limit.velocity is None:
            return self._joint_velocity_limit[joint_name]
        else:
            return spw.Min(limit.velocity, self._joint_velocity_limit[joint_name]) * t

    def get_joint_frame(self, joint_name):
        """
        :param joint_name: name of the joint in the urdfs
        :type joint_name: str
        :return: matrix expression describing the transformation caused by this joint
        :rtype: spw.Matrix
        """
        return self._joint_to_frame[joint_name]

    def get_joint_position_symbol(self, joint_name):
        """
        :param joint_name: name of the joint in the urdfs
        :type joint_name: str
        :rtype: spw.Symbol
        """
        return self._joint_position_symbols[joint_name]

    def get_joint_velocity_symbol(self, joint_name):
        """
        :param joint_name: name of the joint in the urdfs
        :type joint_name: str
        :rtype: spw.Symbol
        """
        return self._joint_velocity_symbols[joint_name]
