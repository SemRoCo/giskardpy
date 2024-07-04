from typing import Tuple
from xml.etree import ElementTree as ET

import numpy as np


def robot_name_from_urdf_string(urdf_string):
    return urdf_string.split('robot name="')[1].split('"')[0]


def hacky_urdf_parser_fix(urdf: str, blacklist: Tuple[str] = ('transmission', 'gazebo')) -> str:
    # Parse input string
    root = ET.fromstring(urdf)

    # Iterate through each section in the blacklist
    for section_name in blacklist:
        # Find all sections with the given name and remove them
        for elem in root.findall(f".//{section_name}"):
            parent = root.find(f".//{section_name}/..")
            if parent is not None:
                parent.remove(elem)

    # Turn back to string
    return ET.tostring(root, encoding='unicode')


def sphere_volume(radius):
    """
    :type radius: float
    :rtype: float
    """
    return (4 / 3.) * np.pi * radius ** 3


def sphere_surface(radius):
    """
    :type radius: float
    :rtype: float
    """
    return 4 * np.pi * radius ** 2


def cube_volume(length, width, height):
    """
    :type length: float
    :type width: float
    :type height: float
    :rtype: float
    """
    return length * width * height


def cube_surface(length, width, height):
    """
    :type length: float
    :type width: float
    :type height: float
    :rtype: float
    """
    return 2 * (length * width + length * height + width * height)


def cylinder_volume(r, h):
    """
    :type r: float
    :type h: float
    :rtype: float
    """
    return np.pi * r ** 2 * h


def cylinder_surface(r, h):
    """
    :type r: float
    :type h: float
    :rtype: float
    """
    return 2 * np.pi * r * (h + r)
