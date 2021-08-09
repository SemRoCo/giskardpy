import os
from copy import deepcopy

import rospy
import yaml

import giskardpy.utils.utils as utils


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
            self.giskardpy_root = utils.get_ros_pkg_path(u'giskardpy')
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
        package_path = utils.get_ros_pkg_path(path_split[0])
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


def load_robot_yaml(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader)
        return utils.update_parents(data)


def ros_load_robot_config(config_file, test=False):
    config = load_robot_yaml(utils.get_ros_pkg_path(u'giskardpy') + u'/config/' + config_file)
    if test:
        config = utils.update_nested_dicts(deepcopy(config),
                                     load_robot_yaml(utils.get_ros_pkg_path(u'giskardpy') + u'/config/test.yaml'))
    if config and not rospy.is_shutdown():
        rospy.set_param('~', config)
        return True
    return False


yaml.add_constructor('!include', construct_include, Loader)
yaml.add_constructor('!find', construct_find, Loader)
