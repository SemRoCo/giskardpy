import copy
from copy import copy
from multiprocessing import Lock

from giskardpy import w

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


def get_data(identifier, data, default_value=0.0):
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
    # TODO deal with unused identifiers
    result = data
    try:
        for member in identifier:
            result = get_member(result, member)
    except AttributeError:
        return default_value
    except KeyError as e:
        # traceback.print_exc()
        # raise KeyError(identifier)
        # TODO is this really a good idea?
        # I do this because it automatically sets weights for unused goals to 0
        return default_value
    except IndexError:
        return default_value
    return result


class GodMap(object):
    """
    Data structure used by plugins to exchange information.
    """

    # TODO give this fucker a lock
    def __init__(self):
        self._data = {}
        self.expr_separator = u'_'
        self.key_to_expr = {}
        self.expr_to_key = {}
        self.default_value = 0
        self.last_expr_values = {}
        self.lock = Lock()

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

    def get_data(self, identifier):
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
        return get_data(identifier, self._data, self.default_value)

    def safe_get_data(self, identifier):
        with self.lock:
            r = self.get_data(identifier)
        return r

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
                raise Exception(u'{} not allowed in key'.format(self.expr_separator))
            self.key_to_expr[identifier] = expr
            self.expr_to_key[str(expr)] = identifier_parts
        return self.key_to_expr[identifier]

    def get_values(self, symbols):
        """
        :return: a dict which maps all registered expressions to their values or 0 if there is no number entry
        :rtype: dict
        """
        # TODO potential speedup by only updating entries that have changed
        # its a trap, this function only looks slow with lineprofiler
        with self.lock:
            # if exprs is None:
            #     exprs = self.expr_to_key.keys()
            # return {expr: self.get_data(self.expr_to_key[expr]) for expr in exprs}
            return [self.get_data(self.expr_to_key[expr]) for expr in symbols]

    def get_registered_symbols(self):
        """
        :rtype: list
        """
        return self.key_to_expr.values()

    def set_data(self, identifier, value):
        """

        :param identifier: e.g. ['pose', 'position', 'x']
        :type identifier: list
        :param value:
        :type value: object
        """
        if len(identifier) == 0:
            raise ValueError(u'key is empty')
        namespace = identifier[0]
        if namespace not in self._data:
            if len(identifier) > 1:
                raise KeyError(u'Can not access member of unknown namespace: {}'.format(identifier))
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

    def safe_set_data(self, identifier, value):
        with self.lock:
            self.set_data(identifier, value)
