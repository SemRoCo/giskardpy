import traceback
from collections import namedtuple, OrderedDict, defaultdict
from copy import deepcopy
from itertools import combinations

from geometry_msgs.msg import PoseStamped

from giskardpy import WORLD_IMPLEMENTATION, symbolic_wrapper as w
from giskardpy.pybullet_world_object import PyBulletWorldObject
from giskardpy.qp_problem_builder import HardConstraint, JointConstraint
from giskardpy.utils import KeyDefaultDict, \
    homo_matrix_to_pose, memoize
from giskardpy.world_object import WorldObject

Joint = namedtuple(u'Joint', [u'symbol', u'velocity_limit', u'lower', u'upper', u'type', u'frame'])

if WORLD_IMPLEMENTATION == u'pybullet':
    Backend = PyBulletWorldObject
else:
    Backend = WorldObject


class Robot(Backend):
    def __init__(self, urdf, base_pose=None, controlled_joints=None, path_to_data_folder=u'', *args, **kwargs):
        """
        :param urdf:
        :type urdf: str
        :param joints_to_symbols_map: maps urdfs joint names to symbols
        :type joints_to_symbols_map: dict
        :param joint_vel_limit: all velocity limits which are undefined or higher than this will be set to this
        :type joint_vel_limit: Symbol
        """
        self._fk_expressions = {}
        self._fks = {}
        self._evaluated_fks = {}
        self._joint_to_frame = {}
        self._joint_position_symbols = KeyDefaultDict(lambda x: w.Symbol(x))  # don't iterate over this map!!
        self._joint_velocity_symbols = KeyDefaultDict(lambda x: 0)  # don't iterate over this map!!
        self._joint_velocity_linear_limit = KeyDefaultDict(lambda x: 10000) # don't overwrite urdf limits by default
        self._joint_velocity_angular_limit = KeyDefaultDict(lambda x: 100000)
        self._joint_acc_linear_limit = defaultdict(lambda: 100)  # no acceleration limit per default
        self._joint_acc_angular_limit = defaultdict(lambda: 100)  # no acceleration limit per default
        self._joint_weights = defaultdict(lambda: 0)
        super(Robot, self).__init__(urdf, base_pose, controlled_joints, path_to_data_folder, *args, **kwargs)
        self.reinitialize()

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
        # self._evaluated_fks.clear()
        self.get_fk_np.memo.clear()

    @memoize
    def get_controlled_parent_joint(self, link_name):
        joint = self.get_parent_joint_of_link(link_name)
        while joint not in self.controlled_joints:
            joint = self.get_parent_joint_of_joint(joint)
        return joint

    def get_joint_state_positions(self):
        try:
            return self.__joint_state_positions
        except:
            return {str(self._joint_position_symbols[x]): 0 for x in self.get_controllable_joints()}

    def reinitialize(self):
        """
        :param joint_position_symbols: maps urdfs joint names to symbols
        :type joint_position_symbols: dict
        """
        super(Robot, self).reinitialize()
        self._fk_expressions = {}
        self._create_frames_expressions()
        self._create_constraints()
        self.init_fast_fks()

    def set_joint_position_symbols(self, symbols):
        self._joint_position_symbols = symbols

    def set_joint_velocity_limit_symbols(self, linear, angular):
        self._joint_velocity_linear_limit = linear
        self._joint_velocity_angular_limit = angular

    def set_joint_velocity_symbols(self, symbols):
        self._joint_velocity_symbols = symbols

    def set_joint_acceleration_limit_symbols(self, linear, angular):
        self._joint_acc_linear_limit = linear
        self._joint_acc_angular_limit = angular

    def set_joint_weight_symbols(self, symbols):
        self._joint_weights = symbols

    def update_joint_symbols(self, position, velocity, weights,
                             linear_velocity_limit, angular_velocity_limit,
                             linear_acceleration_limit, angular_acceleration_limit):
        self.set_joint_position_symbols(position)
        self.set_joint_velocity_symbols(velocity)
        self.set_joint_weight_symbols(weights)
        self.set_joint_velocity_limit_symbols(linear_velocity_limit, angular_velocity_limit)
        self.set_joint_acceleration_limit_symbols(linear_acceleration_limit, angular_acceleration_limit)
        self.reinitialize()

    def update_self_collision_matrix(self, added_links=None, removed_links=None):
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
                    joint_frame = w.dot(w.translation3(*xyz), w.rotation_matrix_from_rpy(*rpy))
                else:
                    joint_frame = w.eye(4)
            else:
                # TODO more specific exception
                raise TypeError(u'Joint type "{}" is not supported by urdfs parser.'.format(urdf_joint.type))

            if self.is_joint_rotational(joint_name):
                joint_frame = w.dot(joint_frame,
                                    w.rotation_matrix_from_axis_angle(w.vector3(*urdf_joint.axis), joint_symbol))
            elif self.is_joint_prismatic(joint_name):
                translation_axis = (w.point3(*urdf_joint.axis) * joint_symbol)
                joint_frame = w.dot(joint_frame, w.translation3(translation_axis[0],
                                                                translation_axis[1],
                                                                translation_axis[2]))

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

                self._joint_constraints[joint_name] = JointConstraint(lower=-velocity_limit,
                                                                      upper=velocity_limit,
                                                                      weight=self._joint_weights[joint_name])
            else:
                self._joint_constraints[joint_name] = JointConstraint(lower=-velocity_limit,
                                                                      upper=velocity_limit,
                                                                      weight=self._joint_weights[joint_name])

    def get_fk_expression(self, root_link, tip_link):
        """
        :type root_link: str
        :type tip_link: str
        :return: 4d matrix describing the transformation from root_link to tip_link
        :rtype: spw.Matrix
        """
        fk = w.eye(4)
        root_chain, _, tip_chain = self.get_split_chain(root_link, tip_link, links=False)
        for joint_name in root_chain:
            fk = w.dot(fk, w.inverse_frame(self.get_joint_frame(joint_name)))
        for joint_name in tip_chain:
            fk = w.dot(fk, self.get_joint_frame(joint_name))
        # FIXME there is some reference fuckup going on, but i don't know where; deepcopy is just a quick fix
        return deepcopy(fk)

    def get_fk_pose(self, root, tip):
        try:
            homo_m = self.get_fk_np(root, tip)
            p = PoseStamped()
            p.header.frame_id = root
            p.pose = homo_matrix_to_pose(homo_m)
        except Exception as e:
            traceback.print_exc()
            pass
        return p

    @memoize
    def get_fk_np(self, root, tip):
        return self._fks[root, tip](**self.get_joint_state_positions())

    def init_fast_fks(self):
        def f(key):
            root, tip = key
            fk = self.get_fk_expression(root, tip)
            m = w.speed_up(fk, w.free_symbols(fk))
            return m

        self._fks = KeyDefaultDict(f)

    # JOINT FUNCTIONS

    @memoize
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
        t = w.Symbol(u'rosparam_general_options_sample_period')  # TODO this should be a parameter
        if self.is_joint_prismatic(joint_name):
            limit_symbol = self._joint_velocity_linear_limit[joint_name]
        else:
            limit_symbol = self._joint_velocity_angular_limit[joint_name]
        if limit is None or limit.velocity is None:
            return limit_symbol
        else:
            return w.Min(limit.velocity, limit_symbol) * t

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
