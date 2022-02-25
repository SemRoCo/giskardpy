import os
from copy import deepcopy

import rospkg
import rospy
import yaml

from giskardpy.utils.utils import resolve_ros_iris

rospack = rospkg.RosPack()


def get_ros_pkg_path(ros_pkg):
    return rospack.get_path(ros_pkg)


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        """
        Initialise Loader by setting the root directory, specifying keywords for finding config files from other
        ROS packages.
        """

        try:
            self.config_root = os.path.split(stream.name)[0]
            self.ros_package_keywords = ['ros://', 'package://']
            self.giskardpy_root = get_ros_pkg_path('giskardpy')
        except AttributeError:
            self.config_root = os.path.curdir

        super(Loader, self).__init__(stream)


def find_parent_of_key(key, value, keys):
    """
    Will return the values of the dict managing the given key and
    returns the key-chain leading from the nested dict value to the
    dict containing the given key.

    :type key: str
    :type value: dict
    :type keys: list(str)
    :rtype: tuple(dict, list(str))
    :returns: tuple containing a dict and the key-chain in a list
    """
    if key in value and key not in value[key]:
        return value, keys
    else:
        for k, v in value.items():
            if type(v) == dict:
                new_keys = deepcopy(keys)
                new_keys.append(k)
                ret = find_parent_of_key(key, v, new_keys)
                if ret is not None:
                    return ret


def nested_update(dic, keys, value):
    """
    Will update the nested dict dic with the key-chain keys with the given value.

    :type dic: dict
    :type keys: list(str)
    :type value: dict or str
    :rtype: dict
    :returns: updated dict
    """
    if keys:
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]].update(value)
    else:
        dic.update(value)


def update_nested_dicts(d, u):
    """
    Will update the values in the nested dict d from nested dict u and
    add new key-value-pairs from nested dict u into nested dict d." \

    :type d: dict
    :type u: dict
    :rtype: dict
    :returns: updated nested dict
    """
    for k, v in u.items():
        if type(v) == dict and d.get(k, {}) is not None:
            d[k] = update_nested_dicts(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def update_parents(d, merge_key):
    """
    Will recursively merge the dict containing the given key merge_key with the value in d[merge_key].

    :type d: dict
    :returns: dict
    """
    if merge_key not in d:
        return d
    else:
        root_data = deepcopy(d)
        while True:
            data_keys_tuple = find_parent_of_key(merge_key, root_data, [])
            if data_keys_tuple is not None:
                data, keys = data_keys_tuple
            else:
                break
            parent_data = data[merge_key]
            data.pop(merge_key)
            updated = update_nested_dicts(parent_data, data)
            nested_update(root_data, keys, updated)
        return root_data


def get_filename(file_or_ros_path_str, loader, node, root):
    """Returns file name referenced at given node by using the given loader."""

    indices = [i for i, x in enumerate(loader.ros_package_keywords) if x in file_or_ros_path_str]
    if indices:
        if len(indices) != 1:
            raise SyntaxError('Invalid ros package path: please use ros:// or package:// as path prefix.')
        removed_key_word = file_or_ros_path_str.replace(loader.ros_package_keywords[indices[0]], '')
        path_split = removed_key_word.split('/')
        package_path = get_ros_pkg_path(path_split[0])
        filename = package_path + removed_key_word.replace(path_split[0], '')
    else:
        filename = os.path.abspath(os.path.join(root, file_or_ros_path_str))
    return filename


def construct_include(loader, node):
    """Load config file referenced at given node by using the given loader."""

    file_str_or_list = loader.construct_scalar(node)
    ret = dict()

    if isinstance(file_str_or_list, list):
        files_to_load = file_str_or_list
    else:
        files_to_load = [file_str_or_list]

    for file_or_ros_path_str in files_to_load:
        filename = get_filename(file_or_ros_path_str, loader, node, loader.config_root)
        extension = os.path.splitext(filename)[1].lstrip('.')

        with open(filename, 'r') as f:
            if extension in ('yaml', 'yml'):
                loaded_dict = yaml.load(f, Loader)
                update_nested_dicts(ret, loaded_dict)
            else:
                loaded_str = ''.join(f.readlines())
                update_nested_dicts(ret, {file_or_ros_path_str: loaded_str})

    return ret


def construct_find(loader, node):
    """Find directory or file referenced at given node by using the given loader."""
    file_or_ros_path_str = loader.construct_scalar(node)
    return get_filename(file_or_ros_path_str, loader, node, loader.giskardpy_root)


def load_robot_yaml(path, merge_key='parent'):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader)
        updated = update_parents(data, merge_key)
        return cast_values_in_nested_dict(updated, float)


def cast_values_in_nested_dict(d, constructor):
    """
    Will try to cast the values in the given nested dict d with the given constructor.
    :type d: dict
    :type constructor: type
    :rtype: dict
    """
    for k, v in d.items():
        if isinstance(v, dict) and d.get(k, {}) is not None:
            cast_values_in_nested_dict(d.get(k, {}), constructor)
        elif isinstance(v, list):
            v_new = []
            for w in v:
                if isinstance(w, str) or isinstance(w, list):
                    tmp = {None: w}
                    cast_values_in_nested_dict(tmp, constructor)
                    v_new.append(tmp[None])
            d.update({k: v_new})
        else:
            if isinstance(d[k], str):
                try:
                    d.update({k: constructor(d[k])})
                except ValueError:
                    pass
    return d


def ros_load_robot_config(config_file, old_data=None, test=False):
    config = load_robot_yaml(resolve_ros_iris(config_file))
    if test:
        config = update_nested_dicts(deepcopy(config),
                                     load_robot_yaml(get_ros_pkg_path('giskardpy') + '/config/test.yaml'))
    if 'action_server' not in config:
        config = update_nested_dicts(deepcopy(config),
                                     load_robot_yaml(get_ros_pkg_path('giskardpy') + '/config/action_server.yaml'))
    if config and not rospy.is_shutdown():
        if old_data is None:
            old_data = {}
        old_data.update(config)
        rospy.set_param('~', old_data)
        return True
    return False


def get_namespaces(d, namespace_seperator='/'):
    """
    This function tries to find namespaces in the given dictionary by searching its top level keys
    for the namespace_seperator. Moreover, the prefix entry in the values is checked too (see example below).
    Therefore, a namespace is registered if it is specified in the top level key name or in the value part
    with the key word 'prefix'.
    If no namespaces are found a list is returned holding as many empty strings entries as top level keys
    exist in the given dictionary.

    Valid dict with namespacing:
        dict = {
        namespace1/something:
            something: 2
            other: 1
            prefix: namespace1 (optional)
        something:
            ...
            prefix: namespace2 (optional)
        /something:
            ...
            prefix: namespace3 (optional)
        /namespace4/something: (also okay)
            ...
            prefix: namespace4 (optional)
        }

    :type d: dict
    :type namespace_seperator: str
    :rtype: list of str
    """
    single_robot_namespace = ''
    no_namespaces = list()
    namespaces = list()
    for i, (key_name, values) in enumerate(d.items()):
        prefix_namespace = None
        name_namespace = None

        # Namespacing by specifying the prefix keyword
        if 'prefix' in values:
            prefix_namespace = values['prefix']

        # Namespacing by the action server name
        # Namespacing: e.g. pr2_a/base
        if key_name.count(namespace_seperator) > 0:
            if key_name.count(namespace_seperator) == 1:
                pass
            # Namespacing: e.g. /pr2_a/base
            elif key_name.count(namespace_seperator) == 2 and \
                    key_name.index(namespace_seperator) == 0:
                key_name = key_name[1:]
            else:
                raise Exception('{} is an invalid combination of a namespace and'
                                ' the action server name.'.format(key_name))
            name_namespace = key_name[:key_name.index(namespace_seperator)]
            if name_namespace == single_robot_namespace:
                name_namespace = None

        # Check prefix_namespace with the namespace in the action server name
        if prefix_namespace is None and name_namespace is None:
            no_namespaces.append(key_name)
            continue
        else:
            if prefix_namespace is not None:
                if name_namespace is None:
                    namespaces.append(prefix_namespace)
                elif prefix_namespace != name_namespace:
                    raise ('Prefix namespace {} differs from the namespace specified '
                            'in the action server name {}'.format(prefix_namespace, name_namespace))
                else:
                    namespaces.append(prefix_namespace)
            else:
                namespaces.append(name_namespace)

    if len(namespaces) == 0:
        namespaces = [single_robot_namespace] * len(d.items())
    elif len(namespaces) != len(d.items()):
        raise Exception('The entries {} have no namespacing but the others do.'.format(str(no_namespaces)))

    return namespaces


yaml.add_constructor('!include', construct_include, Loader)
yaml.add_constructor('!find', construct_find, Loader)
