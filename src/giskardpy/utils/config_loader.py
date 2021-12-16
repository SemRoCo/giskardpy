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
            self.ros_package_keywords = [u'ros://', u'package://']
            self.giskardpy_root = get_ros_pkg_path(u'giskardpy')
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
                keys.append(k)
                return find_parent_of_key(key, v, keys)


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


def get_filename(loader, node, root):
    """Returns file name referenced at given node by using the given loader."""

    file_or_ros_path_str = loader.construct_scalar(node)
    indices = [i for i, x in enumerate(loader.ros_package_keywords) if x in file_or_ros_path_str]
    if indices:
        if len(indices) != 1:
            raise SyntaxError(u'Invalid ros package path: please use ros:// or package:// as path prefix.')
        removed_key_word = file_or_ros_path_str.replace(loader.ros_package_keywords[indices[0]], '')
        path_split = removed_key_word.split('/')
        package_path = get_ros_pkg_path(path_split[0])
        filename = package_path + removed_key_word.replace(path_split[0], '')
    else:
        filename = os.path.abspath(os.path.join(root, file_or_ros_path_str))
    return filename


def construct_include(loader, node):
    """Load config file referenced at given node by using the given loader."""

    filename = get_filename(loader, node, loader.config_root)
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        else:
            return ''.join(f.readlines())


def construct_find(loader, node):
    """Find directory or file referenced at given node by using the given loader."""
    return get_filename(loader, node, loader.giskardpy_root)


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
                                     load_robot_yaml(get_ros_pkg_path(u'giskardpy') + u'/config/test.yaml'))
    if config and not rospy.is_shutdown():
        if old_data is None:
            old_data = {}
        old_data.update(config)
        rospy.set_param('~', old_data)
        return True
    return False


yaml.add_constructor('!include', construct_include, Loader)
yaml.add_constructor('!find', construct_find, Loader)