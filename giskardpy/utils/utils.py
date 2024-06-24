from __future__ import division
import hashlib

# I only do this, because otherwise test/test_integration_pr2.py::TestWorldManipulation::test_unsupported_options
# fails on github actions
import urdf_parser_py.urdf as up

import errno
import inspect
import json
import os
import pkgutil
import sys
from contextlib import contextmanager
from functools import cached_property
from typing import Type, Optional, Dict, Any

from giskardpy.god_map import god_map
from giskardpy.middleware import logging
from giskardpy.middleware.ros1.ros1_interface import resolve_ros_iris


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stdout


@contextmanager
def suppress_stdout():
    devnull = os.open('/dev/null', os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(devnull)


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


def get_all_classes_in_package(package_name: str, parent_class: Optional[Type] = None, silent: bool = False) \
        -> Dict[str, Type]:
    """
    :param package_name: e.g. giskardpy.goals
    :param parent_class: e.g. Goal
    :return:
    """
    classes = {}
    package = __import__(package_name, fromlist="dummy")
    for importer, module_name, ispkg in pkgutil.iter_modules(package.__path__):
        try:
            new_classes = get_all_classes_in_module(f'{package.__name__}.{module_name}', parent_class)
        except Exception as e:
            if not silent:
                logging.logwarn(f'Failed to load {module_name}: {str(e)}')
            continue
        classes.update(new_classes)
    return classes


def get_all_classes_in_module(module_name: str, parent_class: Optional[Type] = None) -> Dict[str, Type]:
    """
    :param module_name: e.g. giskardpy.goals
    :param parent_class: e.g. Goal
    :return:
    """
    classes = {}
    module = __import__(module_name, fromlist="dummy")
    for class_name, class_type in inspect.getmembers(module, inspect.isclass):
        if parent_class is None or issubclass(class_type, parent_class) and module_name in str(class_type):
            classes[class_name] = class_type
    return classes


def limits_from_urdf_joint(urdf_joint):
    lower_limits = {}
    upper_limits = {}
    if not urdf_joint.type == 'continuous':
        try:
            lower_limits[0] = max(urdf_joint.safety_controller.soft_lower_limit, urdf_joint.limit.lower)
            upper_limits[0] = min(urdf_joint.safety_controller.soft_upper_limit, urdf_joint.limit.upper)
        except AttributeError:
            try:
                lower_limits[0] = urdf_joint.limit.lower
                upper_limits[0] = urdf_joint.limit.upper
            except AttributeError:
                pass
    try:
        lower_limits[1] = -urdf_joint.limit.velocity
        upper_limits[1] = urdf_joint.limit.velocity
    except AttributeError:
        pass
    return lower_limits, upper_limits


#
# CONVERSION FUNCTIONS FOR ROS MESSAGES
#


def print_dict(d):
    print('{')
    for key, value in d.items():
        print(f'{key}: {value},')
    print('}')


def write_dict(d, f):
    json.dump(d, f, sort_keys=True, indent=4, separators=(',', ': '))
    f.write('\n')


def create_path(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def cm_to_inch(cm):
    return cm * 0.393701


def get_file_hash(file_path: str, algorithm: str = "sha256") -> Optional[str]:
    try:
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        print(f"Error occurred while hashing: {e}")
        return None


def clear_cached_properties(instance: Any):
    """
    Clears the cache of all cached_property attributes of an instance.

    Args:
        instance: The instance for which to clear all cached_property caches.
    """
    for attr in dir(instance):
        if isinstance(getattr(type(instance), attr, None), cached_property):
            if attr in instance.__dict__:
                del instance.__dict__[attr]


def resolve_ros_iris_in_urdf(input_urdf):
    """
    Replace all instances of ROS IRIs with a urdfs string with global paths in the file system.
    :param input_urdf: URDF in which the ROS IRIs shall be replaced.
    :type input_urdf: str
    :return: URDF with replaced ROS IRIs.
    :rtype: str
    """
    output_urdf = ''
    for line in input_urdf.split('\n'):
        output_urdf += resolve_ros_iris(line)
        output_urdf += '\n'
    return output_urdf


def write_to_tmp(file_name: str, file_str: str) -> str:
    """
    Writes a URDF string into a temporary file on disc. Used to deliver URDFs to PyBullet that only loads file.
    :param file_name: Name of the temporary file without any path information, e.g. 'pr2.urdfs'
    :param file_str: URDF as an XML string that shall be written to disc.
    :return: Complete path to where the urdfs was written, e.g. '/tmp/pr2.urdfs'
    """
    new_path = to_tmp_path(file_name)
    create_path(new_path)
    with open(new_path, 'w') as f:
        f.write(file_str)
    return new_path


def to_tmp_path(file_name: str) -> str:
    path = god_map.tmp_folder
    return resolve_ros_iris(f'{path}{file_name}')


def load_from_tmp(file_name: str):
    new_path = to_tmp_path(file_name)
    create_path(new_path)
    with open(new_path, 'r') as f:
        loaded_file = f.read()
    return loaded_file


def fix_obj(file_name):
    logging.loginfo(f'Attempting to fix {file_name}.')
    with open(file_name, 'r') as f:
        lines = f.readlines()
        fixed_obj = ''
        for line in lines:
            if line.startswith('f '):
                new_line = 'f '
                for part in line.split(' ')[1:]:
                    f_number = part.split('//')[0]
                    new_part = '//'.join([f_number, f_number])
                    new_line += ' ' + new_part
                fixed_obj += new_line + '\n'
            else:
                fixed_obj += line
        with open(file_name, 'w') as f:
            f.write(fixed_obj)


def string_shortener(original_str: str, max_lines: int, max_line_length: int) -> str:
    if len(original_str) < max_line_length:
        return original_str
    lines = []
    start = 0
    for _ in range(max_lines):
        end = start + max_line_length
        lines.append(original_str[start:end])
        if end >= len(original_str):
            break
        start = end

    result = '\n'.join(lines)

    # Check if string is cut off and add "..."
    if len(original_str) > start:
        result = result + '...'

    return result


class ImmutableDict(dict):
    """
    A dict that prevent reassignment of keys.
    """

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f'Key "{key}" already exists. Cannot reassign value.')
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self.__setitem__(k, v)


def is_running_in_pytest():
    return "pytest" in sys.modules
