import copy
import symengine as se
from copy import copy


class GodMap(object):
    def __init__(self):
        self._data = {}
        self.separator = '/'
        self.expr_separator = '_'
        self.key_to_expr = {}
        self.expr_to_key = {}

    def __copy__(self):
        god_map_copy = GodMap()
        god_map_copy._data = copy(self._data)
        god_map_copy.key_to_expr = copy(self.key_to_expr)
        god_map_copy.expr_to_key = copy(self.expr_to_key)
        return god_map_copy

    def _get_member(self, identifier,  member):
        if isinstance(identifier, dict):
            return identifier[member]
        elif isinstance(identifier, list) or (isinstance(identifier, tuple) and member.isdigit()):
            return identifier[int(member)]
        else:
            return getattr(identifier, member)

    def get_data(self, key):
        # TODO deal with unused identifiers
        identifier_parts = key.split(self.separator)
        namespace = identifier_parts[0]
        result = self._data.get(namespace)
        for member in identifier_parts[1:]:
            try:
                result = self._get_member(result, member)
            except AttributeError:
                result = 0
        return result

    def get_expr(self, key):
        if key not in self.key_to_expr:
            expr = se.Symbol(key.replace(self.separator, self.expr_separator))
            if expr in self.expr_to_key:
                raise Exception('{} not allowed in key'.format(self.expr_separator))
            self.key_to_expr[key] = expr
            self.expr_to_key[str(expr)] = key
        return self.key_to_expr[key]

    def get_key(self, expr):
        return self.expr_to_key[str(expr)]

    def get_expr_values(self):
        return {str(self.get_expr(key)): self.get_data(key) for key in self.key_to_expr}

    def set_data(self, key, value):
        identifier_parts = key.split(self.separator)
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


