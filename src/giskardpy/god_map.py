import copy
import numbers
from collections import defaultdict
from copy import copy, deepcopy
from multiprocessing import Lock

import numpy as np
from geometry_msgs.msg import Pose, Point, Vector3, PoseStamped, PointStamped, Vector3Stamped

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.utils.config_loader import upload_config_file_to_paramserver


def set_default_in_override_block(block_identifier, god_map):
    default_value = god_map.get_data(block_identifier[:-1] + ['default'])
    override = god_map.get_data(block_identifier)
    d = defaultdict(lambda: default_value)
    if isinstance(override, dict):
        if isinstance(default_value, dict):
            for key, value in override.items():
                o = deepcopy(default_value)
                o.update(value)
                override[key] = o
        d.update(override)
    god_map.set_data(block_identifier, d)
    return KeyDefaultDict(lambda key: god_map.to_symbol(block_identifier + [key]))


def get_member(identifier, member):
    """
    :param identifier:
    :type identifier: Union[None, dict, list, tuple, object]
    :param member:
    :type member: str
    :return:
    """
    try:
        return identifier[member]
    except TypeError:
        if callable(identifier):
            return identifier(*member)
        try:
            return getattr(identifier, member)
        except TypeError:
            pass
    except IndexError:
        return identifier[int(member)]
    except RuntimeError:
        pass


class GetMember(object):
    def __init__(self):
        self.member = None
        self.child = None

    def init_call(self, identifier, data):
        self.member = identifier[0]
        sub_data = self.c(data)
        if len(identifier) == 2:
            self.child = GetMemberLeaf()
            return self.child.init_call(identifier[-1], sub_data)
        elif len(identifier) > 2:
            self.child = GetMember()
            return self.child.init_call(identifier[1:], sub_data)
        return sub_data

    def __call__(self, a):
        return self.c(a)

    def c(self, a):
        try:
            r = a[self.member]
            self.c = self.return_dict
            return r
        except TypeError:
            if callable(a):
                r = a(*self.member)
                self.c = self.return_function_result
                return r
            try:
                r = getattr(a, self.member)
                self.c = self.return_attribute
                return r
            except TypeError:
                pass
        except IndexError:
            r = a[int(self.member)]
            self.c = self.return_list
            return r
        except RuntimeError:
            pass
        raise KeyError(a)

    def return_dict(self, a):
        return self.child.c(a[self.member])

    def return_list(self, a):
        return self.child.c(a[int(self.member)])

    def return_attribute(self, a):
        return self.child.c(getattr(a, self.member))

    def return_function_result(self, a):
        return self.child.c(a(*self.member))


class GetMemberLeaf(object):
    def __init__(self):
        self.member = None
        self.child = None

    def init_call(self, member, data):
        self.member = member
        return self.c(data)

    def __call__(self, a):
        return self.c(a)

    def c(self, a):
        try:
            r = a[self.member]
            self.c = self.return_dict
            return r
        except TypeError:
            if callable(a):
                r = a(*self.member)
                self.c = self.return_function_result
                return r
            try:
                r = getattr(a, self.member)
                self.c = self.return_attribute
                return r
            except TypeError:
                pass
        except IndexError:
            r = a[int(self.member)]
            self.c = self.return_list
            return r
        except RuntimeError:
            pass
        raise KeyError(a)

    def return_dict(self, a):
        return a[self.member]

    def return_list(self, a):
        return a[int(self.member)]

    def return_attribute(self, a):
        return getattr(a, self.member)

    def return_function_result(self, a):
        return a(*self.member)


def get_data(identifier, data):
    """
    :param identifier: Identifier in the form of ['pose', 'position', 'x'],
                       to access class member: robot.joint_state = ['robot', 'joint_state']
                       to access dicts: robot.joint_state['torso_lift_joint'] = ['robot', 'joint_state', ('torso_lift_joint')]
                       to access lists or other indexable stuff: robot.l[-1] = ['robot', 'l', -1]
                       to access functions: lib.str_to_ascii('muh') = ['lib', 'str_to_acii', ['muh']]
                       to access functions without params: robot.get_pybullet_id() = ['robot', 'get_pybullet_id', []]
    :type identifier: list
    :return: object that is saved at key
    """
    try:
        if len(identifier) == 1:
            shortcut = GetMemberLeaf()
            result = shortcut.init_call(identifier[0], data)
        else:
            shortcut = GetMember()
            result = shortcut.init_call(identifier, data)
    except AttributeError as e:
        raise KeyError(e)
    except IndexError as e:
        raise KeyError(e)
    return result, shortcut


class GodMap(object):
    """
    Data structure used by tree to exchange information.
    """

    def __init__(self):
        self._data = {}
        self.expr_separator = '_'
        self.key_to_expr = {}
        self.expr_to_key = {}
        self.last_expr_values = {}
        self.shortcuts = {}
        self.lock = Lock()

    @classmethod
    @profile
    def init_from_paramserver(cls, node_name, upload_config=True):
        import rospy
        from giskardpy.data_types import order_map

        if upload_config:
            upload_config_file_to_paramserver()

        self = cls()
        self.set_data(identifier.rosparam, rospy.get_param(node_name))
        self.set_data(identifier.robot_description, rospy.get_param('robot_description'))
        path_to_data_folder = self.get_data(identifier.data_folder)
        # fix path to data folder
        if not path_to_data_folder.endswith('/'):
            path_to_data_folder += '/'
        self.set_data(identifier.data_folder, path_to_data_folder)

        set_default_in_override_block(identifier.external_collision_avoidance, self)
        set_default_in_override_block(identifier.self_collision_avoidance, self)
        # weights
        for i, key in enumerate(self.get_data(identifier.joint_weights), start=1):
            set_default_in_override_block(identifier.joint_weights + [order_map[i], 'override'], self)

        # limits
        for i, key in enumerate(self.get_data(identifier.joint_limits), start=1):
            set_default_in_override_block(identifier.joint_limits + [order_map[i], 'linear', 'override'], self)
            set_default_in_override_block(identifier.joint_limits + [order_map[i], 'angular', 'override'], self)

        return self

    def __copy__(self):
        god_map_copy = GodMap()
        god_map_copy._data = copy(self._data)
        god_map_copy.key_to_expr = copy(self.key_to_expr)
        god_map_copy.expr_to_key = copy(self.expr_to_key)
        return god_map_copy

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

    def unsafe_get_data(self, identifier):
        """

        :param identifier: Identifier in the form of ['pose', 'position', 'x'],
                           to access class member: robot.joint_state = ['robot', 'joint_state']
                           to access dicts: robot.joint_state['torso_lift_joint'] = ['robot', 'joint_state', ('torso_lift_joint')]
                           to access lists or other indexable stuff: robot.l[-1] = ['robot', 'l', -1]
                           to access functions: lib.str_to_ascii('muh') = ['lib', 'str_to_acii', ['muh']]
                           to access functions without params: robot.get_pybullet_id() = ['robot', 'get_pybullet_id', []]
        :type identifier: list
        :return: object that is saved at key
        """
        identifier = tuple(identifier)
        try:
            if identifier not in self.shortcuts:
                result, shortcut = get_data(identifier, self._data)
                if shortcut:
                    self.shortcuts[identifier] = shortcut
                return result
            return self.shortcuts[identifier].c(self._data)
        except Exception as e:
            e2 = type(e)('{}; path: {}'.format(e, identifier))
            raise e2

    def get_data(self, identifier):
        with self.lock:
            r = self.unsafe_get_data(identifier)
        return r

    def clear_cache(self):
        # TODO should be possible without clear cache
        self.shortcuts = {}

    def to_symbol(self, identifier):
        """
        All registered identifiers will be included in self.get_symbol_map().
        :type identifier: list
        :return: the symbol corresponding to the identifier
        :rtype: sw.Symbol
        """
        assert isinstance(identifier, list) or isinstance(identifier, tuple)
        identifier = tuple(identifier)
        identifier_parts = identifier
        if identifier not in self.key_to_expr:
            expr = w.Symbol(self.expr_separator.join([str(x) for x in identifier]))
            if expr in self.expr_to_key:
                raise Exception('{} not allowed in key'.format(self.expr_separator))
            self.key_to_expr[identifier] = expr
            self.expr_to_key[str(expr)] = identifier_parts
        return self.key_to_expr[identifier]

    def to_expr(self, identifier):
        data = self.get_data(identifier)
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if isinstance(data, numbers.Number):
            return self.to_symbol(identifier)
        if isinstance(data, Pose):
            return self.pose_msg_to_frame(identifier)
        elif isinstance(data, PoseStamped):
            return self.pose_msg_to_frame(identifier + ['pose'])
        elif isinstance(data, Point):
            return self.point_msg_to_point3(identifier)
        elif isinstance(data, PointStamped):
            return self.point_msg_to_point3(identifier + ['point'])
        elif isinstance(data, Vector3):
            return self.vector_msg_to_vector3(identifier)
        elif isinstance(data, Vector3Stamped):
            return self.vector_msg_to_vector3(identifier + ['vector'])
        elif isinstance(data, list):
            return self.list_to_symbol_matrix(identifier, data)
        elif isinstance(data, np.ndarray):
            return self.list_to_symbol_matrix(identifier, data)
        else:
            raise NotImplementedError('to_expr not implemented for type {}.'.format(type(data)))

    def list_to_symbol_matrix(self, identifier, data):
        def replace_nested_list(l, f, start_index=None):
            if start_index is None:
                start_index = []
            result = []
            for i, entry in enumerate(l):
                index = start_index + [i]
                if isinstance(entry, list):
                    result.append(replace_nested_list(entry, f, index))
                else:
                    result.append(f(index))
            return result

        return w.Matrix(replace_nested_list(data, lambda index: self.to_symbol(identifier + index)))

    def list_to_point3(self, identifier):
        return w.point3(
            x=self.to_symbol(identifier + [0]),
            y=self.to_symbol(identifier + [1]),
            z=self.to_symbol(identifier + [2]),
        )

    def list_to_vector3(self, identifier):
        return w.vector3(
            x=self.to_symbol(identifier + [0]),
            y=self.to_symbol(identifier + [1]),
            z=self.to_symbol(identifier + [2]),
        )

    def list_to_translation3(self, identifier):
        return w.translation3(
            x=self.to_symbol(identifier + [0]),
            y=self.to_symbol(identifier + [1]),
            z=self.to_symbol(identifier + [2]),
        )

    def list_to_frame(self, identifier):
        return w.Matrix(
            [
                [
                    self.to_symbol(identifier + [0, 0]),
                    self.to_symbol(identifier + [0, 1]),
                    self.to_symbol(identifier + [0, 2]),
                    self.to_symbol(identifier + [0, 3])
                ],
                [
                    self.to_symbol(identifier + [1, 0]),
                    self.to_symbol(identifier + [1, 1]),
                    self.to_symbol(identifier + [1, 2]),
                    self.to_symbol(identifier + [1, 3])
                ],
                [
                    self.to_symbol(identifier + [2, 0]),
                    self.to_symbol(identifier + [2, 1]),
                    self.to_symbol(identifier + [2, 2]),
                    self.to_symbol(identifier + [2, 3])
                ],
                [
                    0, 0, 0, 1
                ],
            ]
        )

    def pose_msg_to_frame(self, identifier):
        return w.frame_quaternion(
            x=self.to_symbol(identifier + ['position', 'x']),
            y=self.to_symbol(identifier + ['position', 'y']),
            z=self.to_symbol(identifier + ['position', 'z']),
            qx=self.to_symbol(identifier + ['orientation', 'x']),
            qy=self.to_symbol(identifier + ['orientation', 'y']),
            qz=self.to_symbol(identifier + ['orientation', 'z']),
            qw=self.to_symbol(identifier + ['orientation', 'w']),
        )

    def point_msg_to_point3(self, identifier):
        return w.point3(
            x=self.to_symbol(identifier + ['x']),
            y=self.to_symbol(identifier + ['y']),
            z=self.to_symbol(identifier + ['z']),
        )

    def vector_msg_to_vector3(self, identifier):
        return w.vector3(
            x=self.to_symbol(identifier + ['x']),
            y=self.to_symbol(identifier + ['y']),
            z=self.to_symbol(identifier + ['z']),
        )

    def get_values(self, symbols):
        """
        :return: a dict which maps all registered expressions to their values or 0 if there is no number entry
        :rtype: list
        """
        # TODO potential speedup by only updating entries that have changed
        # its a trap, this function only looks slow with lineprofiler
        with self.lock:
            return self.unsafe_get_values(symbols)

    def unsafe_get_values(self, symbols):
        """
        :return: a dict which maps all registered expressions to their values or 0 if there is no number entry
        :rtype: list
        """
        return [self.unsafe_get_data(self.expr_to_key[expr]) for expr in symbols]

    def evaluate_expr(self, expr):
        fs = w.free_symbols(expr)
        fss = [str(s) for s in fs]
        f = w.speed_up(expr, fs)
        result = f.call2(self.get_values(fss))
        if len(result) == 1:
            return result[0][0]
        else:
            return result

    def get_registered_symbols(self):
        """
        :rtype: list
        """
        return self.key_to_expr.values()

    def unsafe_set_data(self, identifier, value):
        """

        :param identifier: e.g. ['pose', 'position', 'x']
        :type identifier: list
        :param value:
        :type value: object
        """
        if len(identifier) == 0:
            raise ValueError('key is empty')
        namespace = identifier[0]
        if namespace not in self._data:
            if len(identifier) > 1:
                raise KeyError('Can not access member of unknown namespace: {}'.format(identifier))
            else:
                self._data[namespace] = value
        else:
            result = self._data[namespace]
            for member in identifier[1:-1]:
                result = get_member(result, member)
            if len(identifier) > 1:
                member = identifier[-1]
                if isinstance(result, dict):
                    result[member] = value
                elif isinstance(result, list):
                    result[int(member)] = value
                else:
                    setattr(result, member, value)
            else:
                self._data[namespace] = value

    def set_data(self, identifier, value):
        with self.lock:
            self.unsafe_set_data(identifier, value)
