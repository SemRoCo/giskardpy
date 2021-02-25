from __future__ import division

import traceback
from collections import OrderedDict, defaultdict
from copy import deepcopy

from geometry_msgs.msg import PoseStamped

from giskardpy import WORLD_IMPLEMENTATION, casadi_wrapper as w
from giskardpy import identifier
from giskardpy.data_types import SingleJointState, JointConstraint, HardConstraint
from giskardpy.god_map import GodMap
from giskardpy.pybullet_world_object import PyBulletWorldObject
from giskardpy.utils import KeyDefaultDict, \
    homo_matrix_to_pose, memoize
from giskardpy.world_object import WorldObject

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
        self._joint_velocity_linear_limit = KeyDefaultDict(lambda x: 10000)  # don't overwrite urdf limits by default
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

    @memoize
    def get_controlled_leaf_joints(self):
        leaves = self.get_leaves()
        result = []
        for link_name in leaves:
            has_collision = self.has_link_collision(link_name)
            joint_name = self.get_parent_joint_of_link(link_name)
            while True:
                if joint_name is None:
                    break
                if joint_name in self.controlled_joints:
                    if has_collision:
                        result.append(joint_name)
                    break
                parent_link = self.get_parent_link_of_joint(joint_name)
                has_collision = has_collision or self.has_link_collision(parent_link)
                joint_name = self.get_parent_joint_of_joint(joint_name)
            else:  # if not break
                pass
        return set(result)

    @memoize
    def get_directly_controllable_collision_links(self, joint_name):
        if joint_name not in self.controlled_joints:
            return []
        link_name = self.get_child_link_of_joint(joint_name)
        links = [link_name]
        collision_links = []
        while links:
            link_name = links.pop(0)
            parent_joint = self.get_parent_joint_of_link(link_name)

            if parent_joint != joint_name and parent_joint in self.controlled_joints:
                continue
            if self.has_link_collision(link_name):
                collision_links.append(link_name)
            else:
                child_links = self.get_child_links_of_link(link_name)
                if child_links:
                    links.extend(child_links)
        return collision_links

    def get_joint_state_positions(self):
        try:
            return self.__joint_state_positions
        except:
            return {str(self._joint_position_symbols[x]): 0 for x in self.get_movable_joints()}

    def reinitialize(self):
        """
        :param joint_position_symbols: maps urdfs joint names to symbols
        :type joint_position_symbols: dict
        """
        super(Robot, self).reinitialize()
        self._fk_expressions = {}
        self._create_frames_expressions()
        # self._create_constraints()
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
            if self.is_joint_movable(joint_name):
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

    def _create_constraints(self, god_map):
        """
        Creates hard and joint constraints.
        :type god_map: GodMap
        """
        self._hard_constraints = OrderedDict()
        self._joint_constraints = OrderedDict()
        for i, joint_name in enumerate(self.get_joint_names_controllable()):
            lower_limit, upper_limit = self.get_joint_limits(joint_name)
            joint_symbol = self.get_joint_position_symbol(joint_name)
            sample_period = god_map.to_symbol(identifier.sample_period)
            velocity_limit = self.get_joint_velocity_limit_expr(joint_name)  # * sample_period
            acceleration_limit = self.get_joint_acceleration_limit_expr(joint_name)  # * sample_period
            acceleration_limit2 = acceleration_limit * sample_period

            weight = self._joint_weights[joint_name]
            weight = weight * (1. / (velocity_limit)) ** 2
            last_joint_velocity = god_map.to_symbol(identifier.last_joint_states + [joint_name, u'velocity'])

            if not self.is_joint_continuous(joint_name):
                self._joint_constraints[joint_name] = JointConstraint(
                    lower_v=-velocity_limit,
                    upper_v=velocity_limit,
                    weight_v=0.0, # TODO is that right?
                    lower_a=w.limit(w.velocity_limit_from_position_limit(acceleration_limit,
                                                                         lower_limit,
                                                                         joint_symbol,
                                                                         sample_period) - last_joint_velocity,
                                    -acceleration_limit2,
                                    acceleration_limit2),
                    upper_a=w.limit(w.velocity_limit_from_position_limit(acceleration_limit,
                                                                         upper_limit,
                                                                         joint_symbol,
                                                                         sample_period) - last_joint_velocity,
                                    -acceleration_limit2,
                                    acceleration_limit2),
                    weight_a=weight,
                    linear_weight=0)
                self._hard_constraints[joint_name] = HardConstraint(velocity_limit,
                                                                    velocity_limit)
            else:
                self._joint_constraints[joint_name] = JointConstraint(
                    lower_v=-velocity_limit,
                    upper_v=velocity_limit,
                    weight_v=0.0, # TODO is that right?
                    lower_a=-acceleration_limit2,
                    upper_a=acceleration_limit2,
                    weight_a=weight,
                    linear_weight=0)

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
            print(e)
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
        if self.is_joint_prismatic(joint_name):
            limit_symbol = self._joint_velocity_linear_limit[joint_name]
        else:
            limit_symbol = self._joint_velocity_angular_limit[joint_name]
        if limit is None or limit.velocity is None:
            return limit_symbol
        else:
            return w.min(limit.velocity, limit_symbol)

    def get_joint_acceleration_limit_expr(self, joint_name):
        """
        :param joint_name: name of the joint in the urdfs
        :type joint_name: str
        :return: minimum of default velocity limit and limit specified in urdfs
        :rtype: float
        """
        limit = self._urdf_robot.joint_map[joint_name].limit
        if self.is_joint_prismatic(joint_name):
            limit_symbol = self._joint_acc_linear_limit[joint_name]
        else:
            limit_symbol = self._joint_acc_angular_limit[joint_name]
        if limit is None or limit.effort is None:
            return limit_symbol
        else:
            return w.min(limit.effort, limit_symbol)

    def get_joint_velocity_limit_expr_evaluated(self, joint_name, god_map):
        """
        :param joint_name: name of the joint in the urdfs
        :type joint_name: str
        :return: minimum of default velocity limit and limit specified in urdfs
        :rtype: float
        """
        limit = self.get_joint_velocity_limit_expr(joint_name)
        f = w.speed_up(limit, w.free_symbols(limit))
        return f.call2(god_map.get_values(f.str_params))[0][0]

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

    def get_joint_position_symbols(self):
        return [self.get_joint_position_symbol(joint_name) for joint_name in self.controlled_joints]

    def get_joint_velocity_symbol(self, joint_name):
        """
        :param joint_name: name of the joint in the urdfs
        :type joint_name: str
        :rtype: spw.Symbol
        """
        return self._joint_velocity_symbols[joint_name]

    def get_joint_velocity_symbols(self):
        return [self.get_joint_velocity_symbol(joint_name) for joint_name in self.controlled_joints]

    def generate_joint_state(self, f):
        """
        :param f: lambda joint_info: float
        :return:
        """
        js = {}
        for joint_name in sorted(self.controlled_joints):
            sjs = SingleJointState()
            sjs.name = joint_name
            sjs.position = f(joint_name)
            js[joint_name] = sjs
        return js

    def link_order(self, link_a, link_b):
        """
        TODO find a better name
        this function is used when deciding for which order to calculate the collisions
        true if link_a < link_b
        :type link_a: str
        :type link_b: str
        :rtype: bool
        """
        try:
            self.get_controlled_parent_joint(link_a)
        except KeyError:
            return False
        try:
            self.get_controlled_parent_joint(link_b)
        except KeyError:
            return True
        return link_a < link_b

    @memoize
    def get_chain_reduced_to_controlled_joints(self, link_a, link_b):
        chain = self.get_chain(link_b, link_a)
        for i, thing in enumerate(chain):
            if i % 2 == 1 and thing in self.controlled_joints:
                new_link_b = chain[i - 1]
                break
        else:
            raise KeyError(u'no controlled joint in chain between {} and {}'.format(link_a, link_b))
        for i, thing in enumerate(reversed(chain)):
            if i % 2 == 1 and thing in self.controlled_joints:
                new_link_a = chain[len(chain) - i]
                break
        else:
            raise KeyError(u'no controlled joint in chain between {} and {}'.format(link_a, link_b))
        return new_link_a, new_link_b
