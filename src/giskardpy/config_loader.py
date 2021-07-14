import os
from copy import deepcopy
import yaml
import rospkg


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        """
        Initialise Loader by setting the root directory, specifying keywords for finding config files from other
        ROS packages and creating a RosPack object.
        """

        try:
            self.config_root = os.path.split(stream.name)[0]
            self.ros_package_keywords = [u'ros://', u'package://']
            self.rospack = rospkg.RosPack()
            self.giskardpy_root = self.rospack.get_path(u'giskardpy')
        except AttributeError:
            self.config_root = os.path.curdir

        super(Loader, self).__init__(stream)


def get_filename(loader, node, root):
    """Returns file name referenced at given node by using the given loader."""

    file_or_ros_path_str = loader.construct_scalar(node)
    indices = [i for i, x in enumerate(loader.ros_package_keywords) if x in file_or_ros_path_str]
    if indices:
        if len(indices) != 1:
            raise SyntaxError(u'Invalid ros package path: please use ros:// or package:// as path prefix.')
        removed_key_word = file_or_ros_path_str.replace(loader.ros_package_keywords[indices[0]], '')
        path_split = removed_key_word.split('/')
        package_path = loader.rospack.get_path(path_split[0])
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
    if key in value:
        yield value, keys
    else:
        for k, v in value.items():
            if type(v) == dict:
                keys.append(k)
                yield find_parent_of_key(key, v, keys)


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


def update_parents(d):
    """
    Will merge the dict containing the key 'parent' with the value in d['parent'].

    :type d: dict
    :returns: dict
    """
    root_data = deepcopy(d)
    gen = find_parent_of_key('parent', root_data, [])
    while True:
        try:
            data, keys = next(gen)
        except StopIteration:
            break
        parent_data = data['parent']
        data.pop('parent')
        updated = update_nested_dicts(parent_data, data)
        nested_update(root_data, keys, updated)
    return root_data


def load_robot_yaml(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader)
        return update_parents(data)


yaml.add_constructor('!include', construct_include, Loader)
yaml.add_constructor('!find', construct_find, Loader)
