import copy
import traceback

import symengine as se
from copy import copy


class GodMap(object):
    def __init__(self):
        self._data = {}
        self.separator = '/'
        self.expr_separator = '_'
        self.key_to_expr = {}
        self.expr_to_key = {}
        self.default_value = 0
        self.last_expr_values = {}

    def __copy__(self):
        god_map_copy = GodMap()
        god_map_copy._data = copy(self._data)
        god_map_copy.key_to_expr = copy(self.key_to_expr)
        god_map_copy.expr_to_key = copy(self.expr_to_key)
        return god_map_copy

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
            return self._get_member(identifier(self), member)
        elif ',' in member: # handle tuple member
            member = member.replace('(','')
            member = member.replace(')','')
            member = tuple(member.split(','))
        try:
            return identifier[member]
        except TypeError:
            try:
                return identifier[int(member)]
            except (TypeError, ValueError):
                return getattr(identifier, member)

    def get_data(self, key):
        """

        :param key: Key in the from of "foo/bar" or ["foo, bar"]
        :type key: Union[list, str]
        :return: object that is saved at key
        """
        # TODO deal with unused identifiers
        if isinstance(key, str):
            identifier_parts = key.split(self.separator)
        else:
            identifier_parts = key
        namespace = identifier_parts[0]
        result = self._data.get(namespace)
        for member in identifier_parts[1:]:
            try:
                result = self._get_member(result, member)
            except AttributeError:
                result = self.default_value
            except TypeError:
                pass
            except KeyError as e:
                # TODO is this really a good idea?
                # traceback.print_exc()
                # raise KeyError(key)
                result = self.default_value
        if callable(result):
            return result(self)
        else:
            return result

    def get_expr(self, key):
        if isinstance(key, str):
            identifier_parts = key.split(self.separator)
        else:
            identifier_parts = key
            key = '/'.join([str(x) for x in key])
        if key not in self.key_to_expr:
            expr = se.Symbol(key.replace(self.separator, self.expr_separator))
            if expr in self.expr_to_key:
                raise Exception('{} not allowed in key'.format(self.expr_separator))
            self.key_to_expr[key] = expr
            self.expr_to_key[str(expr)] = identifier_parts
        return self.key_to_expr[key]

    def get_expr_values(self):
        #TODO potential speedup by only updating entries that have changed
        return {expr: self.get_data(key) for expr, key in self.expr_to_key.items()}

    def get_free_symbols(self):
        return self.key_to_expr.values()

    def set_data(self, key, value):
        if isinstance(key, str):
            identifier_parts = key.split(self.separator)
        else:
            identifier_parts = [str(x) for x in key]
        namespace = identifier_parts[0]
        if namespace not in self._data:
            if len(identifier_parts) > 1:
                raise KeyError('Can not access member of unknown namespace: {}'.format(key))
            else:
                self._data[namespace] = value
        else:
            result = self._data[namespace]
            for member in identifier_parts[1:-1]:
                result = self._get_member(result, member)
            if len(identifier_parts) > 1:
                member = identifier_parts[-1]
                if isinstance(result, dict):
                    result[member] = value
                elif isinstance(result, list):
                    result[int(member)] = value
                else:
                    setattr(result, member, value)
            else:
                self._data[namespace] = value


