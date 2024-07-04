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

from shape_msgs.msg import SolidPrimitive

from giskard_msgs.msg import WorldBody
from giskardpy.god_map import god_map
from giskardpy.middleware import middleware


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
