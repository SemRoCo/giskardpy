import numpy as np
from shape_msgs.msg import SolidPrimitive

from giskard_msgs.msg import WorldBody


def robot_name_from_urdf_string(urdf_string):
    return urdf_string.split('robot name="')[1].split('"')[0]


def hacky_urdf_parser_fix(urdf_str):
    # this function is inefficient but the tested urdfs's aren't big enough for it to be a problem
    fixed_urdf = ''
    delete = False
    black_list = ['transmission', 'gazebo']
    black_open = ['<{}'.format(x) for x in black_list]
    black_close = ['</{}'.format(x) for x in black_list]
    for line in urdf_str.split('\n'):
        if len([x for x in black_open if x in line]) > 0:
            delete = True
        if len([x for x in black_close if x in line]) > 0:
            delete = False
            continue
        if not delete:
            fixed_urdf += line + '\n'
    return fixed_urdf


def make_world_body_box(x_length: float = 1, y_length: float = 1, z_length: float = 1) -> WorldBody:
    box = WorldBody()
    box.type = WorldBody.PRIMITIVE_BODY
    box.shape.type = SolidPrimitive.BOX
    box.shape.dimensions.append(x_length)
    box.shape.dimensions.append(y_length)
    box.shape.dimensions.append(z_length)
    return box


def make_world_body_sphere(radius=1):
    sphere = WorldBody()
    sphere.type = WorldBody.PRIMITIVE_BODY
    sphere.shape.type = SolidPrimitive.SPHERE
    sphere.shape.dimensions.append(radius)
    return sphere


def make_world_body_cylinder(height=1, radius=1):
    cylinder = WorldBody()
    cylinder.type = WorldBody.PRIMITIVE_BODY
    cylinder.shape.type = SolidPrimitive.CYLINDER
    cylinder.shape.dimensions = [0, 0]
    cylinder.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT] = height
    cylinder.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS] = radius
    return cylinder


def make_urdf_world_body(name, urdf):
    wb = WorldBody()
    wb.type = wb.URDF_BODY
    wb.urdf = urdf
    return wb


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
