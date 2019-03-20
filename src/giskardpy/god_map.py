import copy
from multiprocessing import Lock

import symengine_wrappers as sw
from copy import copy

from giskardpy.utils import is_iterable


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

    def _get_member(self, identifier,  member):
        """
        :param identifier:
        :type identifier: Union[None, dict, list, tuple, object]
        :param member:
        :type member: str
        :return:
        """
        if identifier is None:
            raise AttributeError()
        if callable(identifier):
            # TODO this solution calls identifier multiple times if the result is an array, make it faster
            if is_iterable(member) and not isinstance(member, str) and not isinstance(member, unicode):
                return identifier(*member)
            else:
                return identifier(member)
        try:
            return identifier[member]
        except TypeError:
            try:
                return identifier[int(member)]
            except (TypeError, ValueError):
                try:
                    return getattr(identifier, member)
                except TypeError as e:
                    pass
        except IndexError:
            return identifier[int(member)]

    def get_data(self, identifier):
        """

        :param identifier: Identifier in the form of ['pose', 'position', 'x']
        :type identifier: list
        :return: object that is saved at key
        """
        # TODO deal with unused identifiers
        # assert isinstance(key, list) or isinstance(key, tuple)
        # key = tuple(key)
        namespace = identifier[0]
        result = self._data.get(namespace)
        for member in identifier[1:]:
            try:
                result = self._get_member(result, member)
            except AttributeError:
                result = self.default_value
            except KeyError as e:
                # traceback.print_exc()
                # raise KeyError(identifier)
                # TODO is this really a good idea?
                # I do this because it automatically sets weights for unused goals to 0
                result = self.default_value
        if callable(result):
            return result(self)
        else:
            return result

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
            expr = sw.Symbol(self.expr_separator.join([str(x) for x in identifier]))
            if expr in self.expr_to_key:
                raise Exception(u'{} not allowed in key'.format(self.expr_separator))
            self.key_to_expr[identifier] = expr
            self.expr_to_key[str(expr)] = identifier_parts
        return self.key_to_expr[identifier]

    def get_symbol_map(self, exprs=None):
        """
        :return: a dict which maps all registered expressions to their values or 0 if there is no number entry
        :rtype: dict
        """
        #TODO potential speedup by only updating entries that have changed
        with self.lock:
            if exprs is None:
                exprs = self.expr_to_key.keys()
            return {expr: self.get_data(self.expr_to_key[expr]) for expr in exprs}

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
                result = self._get_member(result, member)
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
