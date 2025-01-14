from __future__ import annotations

import abc
from abc import ABC
from typing import Tuple, Optional, List, Union, Type

import numpy as np
import urdf_parser_py.urdf as up

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import derivative_map, PrefixName, Derivatives
from giskardpy.god_map import god_map
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.symbol_manager import symbol_manager
from line_profiler import profile

def urdf_joint_to_class(urdf_joint: up.Joint) -> Union[Type[FixedJoint], Type[RevoluteJoint], Type[PrismaticJoint]]:
    if urdf_joint.type == 'fixed':
        joint_class = FixedJoint
    elif urdf_joint.type == 'prismatic':
        joint_class = PrismaticJoint
    elif urdf_joint.type in {'revolute', 'continuous'}:
        # if 'caster_rotation' in urdf_joint.name:
        #     joint_class = PR2CasterJoint
        # else:
        joint_class = RevoluteJoint
    else:
        raise NotImplementedError(
            f'Joint type \'{urdf_joint.type}\' of \'{urdf_joint.name}\' is not implemented.')
    return joint_class


def urdf_joint_to_limits(urdf_joint: up.Joint) -> Tuple[derivative_map, derivative_map]:
    lower_limits = {}
    upper_limits = {}
    if not urdf_joint.type == 'continuous':
        try:
            lower_limits[Derivatives.position] = max(urdf_joint.safety_controller.soft_lower_limit,
                                                     urdf_joint.limit.lower)
            upper_limits[Derivatives.position] = min(urdf_joint.safety_controller.soft_upper_limit,
                                                     urdf_joint.limit.upper)
        except AttributeError:
            try:
                lower_limits[Derivatives.position] = urdf_joint.limit.lower
                upper_limits[Derivatives.position] = urdf_joint.limit.upper
            except AttributeError:
                pass
    try:
        lower_limits[Derivatives.velocity] = -urdf_joint.limit.velocity
        upper_limits[Derivatives.velocity] = urdf_joint.limit.velocity
    except AttributeError:
        pass
    if urdf_joint.mimic is not None:
        if urdf_joint.mimic.multiplier is not None:
            multiplier = urdf_joint.mimic.multiplier
        else:
            multiplier = 1
        if urdf_joint.mimic.offset is not None:
            offset = urdf_joint.mimic.offset
        else:
            offset = 0
        for d2 in Derivatives.range(Derivatives.position, Derivatives.velocity):
            lower_limits[d2] -= offset
            upper_limits[d2] -= offset
            if multiplier < 0:
                upper_limits[d2], lower_limits[d2] = lower_limits[d2], upper_limits[d2]
            upper_limits[d2] /= multiplier
            lower_limits[d2] /= multiplier
    return lower_limits, upper_limits


def urdf_to_joint(urdf_joint: up.Joint, prefix: str) \
        -> Union[FixedJoint, RevoluteJoint, PrismaticJoint]:
    joint_class = urdf_joint_to_class(urdf_joint)
    if urdf_joint.origin is not None:
        translation_offset = urdf_joint.origin.xyz
        rotation_offset = urdf_joint.origin.rpy
    else:
        translation_offset = None
        rotation_offset = None
    if translation_offset is None:
        translation_offset = [0, 0, 0]
    if rotation_offset is None:
        rotation_offset = [0, 0, 0]
    parent_T_child = cas.TransMatrix.from_xyz_rpy(x=translation_offset[0],
                                                  y=translation_offset[1],
                                                  z=translation_offset[2],
                                                  roll=rotation_offset[0],
                                                  pitch=rotation_offset[1],
                                                  yaw=rotation_offset[2])
    joint_name = PrefixName(urdf_joint.name, prefix)
    parent_link_name = PrefixName(urdf_joint.parent, prefix)
    child_link_name = PrefixName(urdf_joint.child, prefix)
    if joint_class == FixedJoint:
        return joint_class(name=joint_name,
                           parent_link_name=parent_link_name,
                           child_link_name=child_link_name,
                           parent_T_child=parent_T_child)
    is_mimic = urdf_joint.mimic is not None
    multiplier = None
    offset = None
    if is_mimic:
        if urdf_joint.mimic.multiplier is not None:
            multiplier = urdf_joint.mimic.multiplier
        else:
            multiplier = 1
        if urdf_joint.mimic.offset is not None:
            offset = urdf_joint.mimic.offset
        else:
            offset = 0

        free_variable_name = PrefixName(urdf_joint.mimic.joint, prefix)
    else:
        free_variable_name = joint_name

    lower_limits, upper_limits = urdf_joint_to_limits(urdf_joint)

    return joint_class(name=joint_name,
                       free_variable_name=free_variable_name,
                       parent_link_name=parent_link_name,
                       child_link_name=child_link_name,
                       parent_T_child=parent_T_child,
                       axis=urdf_joint.axis,
                       lower_limits=lower_limits,
                       upper_limits=upper_limits,
                       multiplier=multiplier,
                       offset=offset)


class Joint(ABC):
    name: PrefixName
    parent_link_name: PrefixName
    child_link_name: PrefixName
    parent_T_child: cas.TransMatrix

    def __str__(self) -> str:
        return f'{self.name}: {self.parent_link_name}<-{self.child_link_name}'

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_urdf(cls, urdf_joint: up.Joint, prefix: str) -> Union[FixedJoint, RevoluteJoint, PrismaticJoint]:
        return urdf_to_joint(urdf_joint, prefix)


class VirtualFreeVariables(ABC):
    @abc.abstractmethod
    def update_state(self, dt: float) -> None: ...


class MovableJoint(Joint):
    free_variables: List[FreeVariable]

    @abc.abstractmethod
    def get_free_variable_names(self) -> List[PrefixName]: ...


class FixedJoint(Joint):
    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 parent_T_child: Optional[cas.TransMatrix] = None):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.parent_T_child = parent_T_child or cas.TransMatrix()


class Joint6DOF(Joint):
    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.x = god_map.world.add_virtual_free_variable(name=PrefixName('x', self.name))
        self.y = god_map.world.add_virtual_free_variable(name=PrefixName('y', self.name))
        self.z = god_map.world.add_virtual_free_variable(name=PrefixName('z', self.name))
        self.qx = god_map.world.add_virtual_free_variable(name=PrefixName('qx', self.name))
        self.qy = god_map.world.add_virtual_free_variable(name=PrefixName('qy', self.name))
        self.qz = god_map.world.add_virtual_free_variable(name=PrefixName('qz', self.name))
        self.qw = god_map.world.add_virtual_free_variable(name=PrefixName('qw', self.name))
        god_map.world.state[self.qw.name].position = 1
        parent_P_child = cas.Point3((self.x.get_symbol(Derivatives.position),
                                     self.y.get_symbol(Derivatives.position),
                                     self.z.get_symbol(Derivatives.position)))
        parent_R_child = cas.Quaternion((self.qx.get_symbol(Derivatives.position),
                                         self.qy.get_symbol(Derivatives.position),
                                         self.qz.get_symbol(Derivatives.position),
                                         self.qw.get_symbol(Derivatives.position))).to_rotation_matrix()
        self.parent_T_child = cas.TransMatrix.from_point_rotation_matrix(parent_P_child, parent_R_child)

    def update_transform(self, parent_T_child: cas.TransMatrix) -> None:
        position = parent_T_child.to_position().to_np()
        orientation = parent_T_child.to_rotation().to_quaternion().to_np()
        god_map.world.state[self.x.name].position = position[0]
        god_map.world.state[self.y.name].position = position[1]
        god_map.world.state[self.z.name].position = position[2]
        god_map.world.state[self.qx.name].position = orientation[0]
        god_map.world.state[self.qy.name].position = orientation[1]
        god_map.world.state[self.qz.name].position = orientation[2]
        god_map.world.state[self.qw.name].position = orientation[3]


class OneDofJoint(MovableJoint):
    axis: Tuple[float, float, float]
    multiplier: float
    offset: float
    free_variable: FreeVariable

    def __init__(self,
                 name: PrefixName,
                 free_variable_name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 lower_limits: derivative_map,
                 upper_limits: derivative_map,
                 parent_T_child: Optional[cas.TransMatrix] = None,
                 multiplier: Optional[float] = None,
                 offset: Optional[float] = None):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.parent_T_child = parent_T_child or cas.TransMatrix()
        self.original_parent_T_child = cas.TransMatrix(self.parent_T_child)
        if multiplier is None:
            self.multiplier = 1
        else:
            self.multiplier = multiplier
        if offset is None:
            self.offset = 0
        else:
            self.offset = offset
        self.axis = axis
        if free_variable_name in god_map.world.free_variables:
            self.free_variable = god_map.world.free_variables[free_variable_name]
        else:
            self.free_variable = god_map.world.add_free_variable(free_variable_name, lower_limits, upper_limits)
        self.free_variables = [self.free_variable]

    def get_free_variable_names(self) -> List[PrefixName]:
        return [self.free_variable.name]

    def get_symbol(self, derivative: Derivatives) -> cas.Expression:
        return self.free_variable.get_symbol(derivative) * self.multiplier + self.offset

    def get_limit_expressions(self, order: Derivatives) -> Optional[Tuple[cas.Expression, cas.Expression]]:
        lower_limit = self.free_variable.get_lower_limit(order) * self.multiplier + self.offset
        upper_limit = self.free_variable.get_upper_limit(order) * self.multiplier + self.offset
        return lower_limit, upper_limit


class RevoluteJoint(OneDofJoint):

    def __init__(self,
                 name: PrefixName,
                 free_variable_name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 lower_limits: derivative_map,
                 upper_limits: derivative_map,
                 parent_T_child: Optional[cas.TransMatrix] = None,
                 multiplier: float = 1,
                 offset: float = 0):
        super().__init__(name=name,
                         free_variable_name=free_variable_name,
                         parent_link_name=parent_link_name,
                         child_link_name=child_link_name,
                         axis=axis,
                         parent_T_child=parent_T_child,
                         lower_limits=lower_limits,
                         upper_limits=upper_limits,
                         multiplier=multiplier,
                         offset=offset)
        motor_expression = self.free_variable.get_symbol(Derivatives.position) * self.multiplier + self.offset
        rotation_axis = cas.Vector3(self.axis)
        parent_R_child = cas.RotationMatrix.from_axis_angle(rotation_axis, motor_expression)
        self.parent_T_child = self.parent_T_child.dot(cas.TransMatrix(parent_R_child))


class PrismaticJoint(OneDofJoint):

    def __init__(self,
                 name: PrefixName,
                 free_variable_name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 lower_limits: derivative_map,
                 upper_limits: derivative_map,
                 parent_T_child: Optional[cas.TransMatrix] = None,
                 multiplier: float = 1,
                 offset: float = 0):
        super().__init__(name=name,
                         free_variable_name=free_variable_name,
                         parent_link_name=parent_link_name,
                         child_link_name=child_link_name,
                         axis=axis,
                         parent_T_child=parent_T_child,
                         lower_limits=lower_limits,
                         upper_limits=upper_limits,
                         multiplier=multiplier,
                         offset=offset)
        motor_expression = self.free_variable.get_symbol(Derivatives.position) * self.multiplier + self.offset
        translation_axis = cas.Vector3(self.axis) * motor_expression
        parent_T_child = cas.TransMatrix.from_xyz_rpy(x=translation_axis[0],
                                                      y=translation_axis[1],
                                                      z=translation_axis[2])
        self.parent_T_child = self.parent_T_child.dot(parent_T_child)


class OmniDrive(MovableJoint, VirtualFreeVariables):
    x: FreeVariable
    y: FreeVariable
    z: FreeVariable
    roll: FreeVariable
    pitch: FreeVariable
    yaw: FreeVariable
    x_vel: FreeVariable
    y_vel: FreeVariable

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 translation_limits: Optional[derivative_map] = None,
                 rotation_limits: Optional[derivative_map] = None,
                 x_name: Optional[PrefixName] = None,
                 y_name: Optional[PrefixName] = None,
                 yaw_name: Optional[PrefixName] = None):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        if x_name is not None:
            self.x_name = x_name
        else:
            self.x_name = PrefixName('x', self.name)
        if y_name is not None:
            self.y_name = y_name
        else:
            self.y_name = PrefixName('y', self.name)
        if yaw_name is not None:
            self.yaw_vel_name = yaw_name
        else:
            self.yaw_vel_name = PrefixName('yaw', self.name)
        if translation_limits is None:
            self.translation_limits = {
                Derivatives.velocity: 0.5,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None
            }
        else:
            self.translation_limits = translation_limits

        if rotation_limits is None:
            self.rotation_limits = {
                Derivatives.velocity: 0.6,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None
            }
        else:
            self.rotation_limits = rotation_limits

        self.create_free_variables()
        self.create_parent_T_child()

    def create_parent_T_child(self) -> None:
        odom_T_bf = cas.TransMatrix.from_xyz_rpy(x=self.x.get_symbol(Derivatives.position),
                                                 y=self.y.get_symbol(Derivatives.position),
                                                 yaw=self.yaw.get_symbol(Derivatives.position))
        bf_T_bf_vel = cas.TransMatrix.from_xyz_rpy(x=self.x_vel.get_symbol(Derivatives.position),
                                                   y=self.y_vel.get_symbol(Derivatives.position))
        bf_vel_T_bf = cas.TransMatrix.from_xyz_rpy(x=0,
                                                   y=0,
                                                   z=self.z.get_symbol(Derivatives.position),
                                                   roll=self.roll.get_symbol(Derivatives.position),
                                                   pitch=self.pitch.get_symbol(Derivatives.position),
                                                   yaw=0)
        self.parent_T_child = odom_T_bf.dot(bf_T_bf_vel).dot(bf_vel_T_bf)

    def create_free_variables(self) -> None:
        translation_lower_limits = {derivative: -limit if limit is not None else None for derivative, limit in self.translation_limits.items()}
        rotation_lower_limits = {derivative: -limit if limit is not None else None for derivative, limit in self.rotation_limits.items()}

        self.x = god_map.world.add_virtual_free_variable(name=self.x_name)
        self.y = god_map.world.add_virtual_free_variable(name=self.y_name)
        self.z = god_map.world.add_virtual_free_variable(name=PrefixName('z', self.name))

        self.roll = god_map.world.add_virtual_free_variable(name=PrefixName('roll', self.name))
        self.pitch = god_map.world.add_virtual_free_variable(name=PrefixName('pitch', self.name))
        self.yaw = god_map.world.add_free_variable(name=self.yaw_vel_name,
                                                   lower_limits=rotation_lower_limits,
                                                   upper_limits=self.rotation_limits)

        self.x_vel = god_map.world.add_free_variable(name=PrefixName('x_vel', self.name),
                                                     lower_limits=translation_lower_limits,
                                                     upper_limits=self.translation_limits)
        self.y_vel = god_map.world.add_free_variable(name=PrefixName('y_vel', self.name),
                                                     lower_limits=translation_lower_limits,
                                                     upper_limits=self.translation_limits)
        self.free_variables = [self.x_vel, self.y_vel, self.yaw]

    def update_transform(self, new_parent_T_child: cas.TransMatrix) -> None:
        position = new_parent_T_child.to_position()
        roll, pitch, yaw = new_parent_T_child.to_rotation().to_rpy()
        god_map.world.state[self.x.name].position = position.x.to_np()
        god_map.world.state[self.y.name].position = position.y.to_np()
        god_map.world.state[self.z.name].position = position.z.to_np()
        god_map.world.state[self.roll.name].position = roll.to_np()
        god_map.world.state[self.pitch.name].position = pitch.to_np()
        god_map.world.state[self.yaw.name].position = yaw.to_np()

    def update_state(self, dt: float) -> None:
        state = god_map.world.state
        state[self.x_vel.name].position = 0
        state[self.y_vel.name].position = 0

        x_vel = state[self.x_vel.name].velocity
        y_vel = state[self.y_vel.name].velocity
        delta = state[self.yaw.name].position
        state[self.x.name].velocity = (np.cos(delta) * x_vel - np.sin(delta) * y_vel)
        state[self.x.name].position += state[self.x.name].velocity * dt
        state[self.y.name].velocity = (np.sin(delta) * x_vel + np.cos(delta) * y_vel)
        state[self.y.name].position += state[self.y.name].velocity * dt

    def get_free_variable_names(self) -> List[PrefixName]:
        return [self.x.name, self.y.name, self.yaw.name]


class DiffDrive(MovableJoint, VirtualFreeVariables):
    x: FreeVariable
    y: FreeVariable
    z: FreeVariable
    roll: FreeVariable
    pitch: FreeVariable
    yaw: FreeVariable
    x_vel: FreeVariable

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 translation_limits: Optional[derivative_map] = None,
                 rotation_limits: Optional[derivative_map] = None):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        if translation_limits is None:
            self.translation_limits = {
                Derivatives.velocity: 0.5,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None
            }
        else:
            self.translation_limits = translation_limits

        if rotation_limits is None:
            self.rotation_limits = {
                Derivatives.velocity: 0.6,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None
            }
        else:
            self.rotation_limits = rotation_limits

        self.create_free_variables()
        self.create_parent_T_child()

    def create_parent_T_child(self) -> None:
        odom_T_bf = cas.TransMatrix.from_xyz_rpy(x=self.x.get_symbol(Derivatives.position),
                                                 y=self.y.get_symbol(Derivatives.position),
                                                 yaw=self.yaw.get_symbol(Derivatives.position))
        bf_T_bf_vel = cas.TransMatrix.from_xyz_rpy(x=self.x_vel.get_symbol(Derivatives.position))
        self.parent_T_child = cas.dot(odom_T_bf, bf_T_bf_vel)

    def create_free_variables(self) -> None:
        translation_lower_limits = {derivative: -limit if limit is not None else limit for derivative, limit in self.translation_limits.items()}
        rotation_lower_limits = {derivative: -limit if limit is not None else limit for derivative, limit in self.rotation_limits.items()}

        self.x = god_map.world.add_virtual_free_variable(name=PrefixName('x', self.name))
        self.y = god_map.world.add_virtual_free_variable(name=PrefixName('y', self.name))
        self.z = god_map.world.add_virtual_free_variable(name=PrefixName('z', self.name))

        self.roll = god_map.world.add_virtual_free_variable(name=PrefixName('roll', self.name))
        self.pitch = god_map.world.add_virtual_free_variable(name=PrefixName('pitch', self.name))

        self.x_vel = god_map.world.add_free_variable(name=PrefixName('x_vel', self.name),
                                                     lower_limits=translation_lower_limits,
                                                     upper_limits=self.translation_limits)
        self.yaw = god_map.world.add_free_variable(name=PrefixName('yaw', self.name),
                                                   lower_limits=rotation_lower_limits,
                                                   upper_limits=self.rotation_limits)
        self.free_variables = [self.x_vel, self.yaw]

    def update_transform(self, new_parent_T_child: cas.TransMatrix) -> None:
        position = new_parent_T_child.to_position()
        roll, pitch, yaw = new_parent_T_child.to_rotation().to_rpy()
        god_map.world.state[self.x.name].position = position.x
        god_map.world.state[self.y.name].position = position.y
        god_map.world.state[self.z.name].position = position.z
        god_map.world.state[self.roll.name].position = roll
        god_map.world.state[self.pitch.name].position = pitch
        god_map.world.state[self.yaw.name].position = yaw

    def update_state(self, dt: float) -> None:
        state = god_map.world.state
        state[self.x_vel.name].position = 0

        x_vel = state[self.x_vel.name].velocity
        yaw = state[self.yaw.name].position
        state[self.x.name].velocity = np.cos(yaw) * x_vel
        state[self.x.name].position += state[self.x.name].velocity * dt
        state[self.y.name].velocity = np.sin(yaw) * x_vel
        state[self.y.name].position += state[self.y.name].velocity * dt

    def get_free_variable_names(self) -> List[PrefixName]:
        return [self.x.name, self.y.name, self.yaw.name]


class OmniDrivePR22(MovableJoint, VirtualFreeVariables):
    x: FreeVariable
    y: FreeVariable
    z: FreeVariable
    roll: FreeVariable
    pitch: FreeVariable
    forward_vel: FreeVariable
    yaw1_vel: FreeVariable
    yaw: FreeVariable

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 translation_limits: Optional[derivative_map] = None,
                 rotation_limits: Optional[derivative_map] = None):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        if translation_limits is None:
            self.translation_limits = {
                Derivatives.velocity: 0.5,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None
            }
        else:
            self.translation_limits = translation_limits

        if rotation_limits is None:
            self.rotation_limits = {
                Derivatives.velocity: 0.6,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None
            }
        else:
            self.rotation_limits = rotation_limits

        self.create_free_variables()
        self.create_parent_T_child()

    # def set_initial_state(self):
    # god_map.get_world().state[self.caster_yaw1_name].position = 1

    def create_free_variables(self) -> None:
        translation_lower_limits = {derivative: -limit for derivative, limit in self.translation_limits.items()}
        rotation_lower_limits = {derivative: -limit for derivative, limit in self.rotation_limits.items()}
        caster_upper_limits = {
            Derivatives.velocity: 100,
            Derivatives.acceleration: np.inf,
            Derivatives.jerk: None,
        }
        caster_lower_limits = {
            Derivatives.velocity: -100,
            Derivatives.acceleration: -np.inf,
            Derivatives.jerk: None,
        }

        self.x = god_map.world.add_virtual_free_variable(name=PrefixName('x', self.name))
        self.y = god_map.world.add_virtual_free_variable(name=PrefixName('y', self.name))
        self.z = god_map.world.add_virtual_free_variable(name=PrefixName('z', self.name))

        self.roll = god_map.world.add_virtual_free_variable(name=PrefixName('roll', self.name))
        self.pitch = god_map.world.add_virtual_free_variable(name=PrefixName('pitch', self.name))
        self.yaw = god_map.world.add_virtual_free_variable(name=PrefixName('yaw', self.name))

        self.forward_vel = god_map.world.add_free_variable(name=PrefixName('forward_vel', self.name),
                                                           lower_limits=translation_lower_limits,
                                                           upper_limits=self.translation_limits)
        self.yaw1_vel = god_map.world.add_free_variable(name=PrefixName('yaw1_vel', self.name),
                                                        lower_limits=caster_lower_limits,
                                                        upper_limits=caster_upper_limits)
        self.yaw = god_map.world.add_free_variable(name=PrefixName('yaw2_vel', self.name),
                                                   lower_limits=rotation_lower_limits,
                                                   upper_limits=self.rotation_limits)
        self.free_variables = [self.forward_vel, self.yaw1_vel, self.yaw]
        god_map.world.state[self.forward_vel.name].position = 0.25

        self.yaw1_vel.quadratic_weights[Derivatives.velocity] = 0.0001
        self.yaw1_vel.quadratic_weights[Derivatives.acceleration] = 0
        self.yaw1_vel.quadratic_weights[Derivatives.jerk] = 0.1

    def get_free_variable_names(self) -> List[PrefixName]:
        return [self.forward_vel.name, self.yaw1_vel.name, self.yaw.name]

    @profile
    def create_parent_T_child(self) -> None:
        hack = symbol_manager.hack
        odom_T_bf = cas.TransMatrix.from_xyz_rpy(x=self.x.get_symbol(Derivatives.position),
                                                 y=self.y.get_symbol(Derivatives.position),
                                                 )
        yaw1 = self.yaw1_vel.get_symbol(Derivatives.position)
        forward = self.forward_vel.get_symbol(Derivatives.position)
        yaw2 = self.yaw.get_symbol(Derivatives.position)
        a = cas.TransMatrix.from_xyz_rpy(yaw=yaw1)
        b = cas.TransMatrix.from_xyz_rpy(x=forward * hack)
        # c = cas.TransMatrix.from_xyz_rpy(x=-forward)
        d = cas.TransMatrix.from_xyz_rpy(yaw=-yaw1)
        z = cas.TransMatrix.from_xyz_rpy(x=0,
                                         y=0,
                                         z=self.z.get_symbol(Derivatives.position),
                                         roll=self.roll.get_symbol(Derivatives.position),
                                         pitch=self.pitch.get_symbol(Derivatives.position),
                                         yaw=yaw2)
        self.parent_T_child = odom_T_bf.dot(a).dot(b).dot(d).dot(z)

    def update_state(self, dt: float) -> None:
        state = god_map.world.state
        god_map.world.state[self.forward_vel.name].position = 0.25
        # god_map.get_world().state[self.yaw1_vel.name].position = 0
        # god_map.get_world().state[self.yaw2_vel.name].position = 0

        # state[self.x_vel_name].position = 0
        # vel = state[self.x_vel_name].velocity
        # vel_direction = self.normalize_and_return_vel_direction()
        # x_vel, y_vel, rot_vel = vel_direction * vel
        # rot_vel = state[self.rot_vel_name].velocity
        yaw1_position = state[self.yaw1_vel.name].position
        # yaw1_velocity = state[self.caster_yaw1_name].velocity
        # yaw2_velocity = state[self.caster_yaw2_name].velocity
        # total_yaw_velocity = yaw1_velocity + yaw2_velocity
        forward_velocity = state[self.forward_vel.name].velocity
        state[self.x.name].velocity = np.cos(yaw1_position) * forward_velocity
        state[self.x.name].position += state[self.x.name].velocity * dt
        state[self.y.name].velocity = np.sin(yaw1_position) * forward_velocity
        state[self.y.name].position += state[self.y.name].velocity * dt
        # state[self.yaw_name].velocity = total_yaw_velocity
        # state[self.yaw_name].position += total_yaw_velocity * dt

    def update_transform(self, new_parent_T_child: cas.TransMatrix) -> None:
        position = new_parent_T_child.to_position()
        roll, pitch, yaw = new_parent_T_child.to_rotation().to_rpy()
        god_map.world.state[self.x.name].position = position.x
        god_map.world.state[self.y.name].position = position.y
        god_map.world.state[self.z.name].position = position.z
        god_map.world.state[self.roll.name].position = roll
        god_map.world.state[self.pitch.name].position = pitch
        god_map.world.state[self.yaw.name].position = yaw


class PR2CasterJoint(MovableJoint):
    def __init__(self,
                 name: PrefixName,
                 free_variable_name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 parent_T_child: cas.TransMatrix,
                 lower_limits: derivative_map,
                 upper_limits: derivative_map,
                 multiplier: Optional[float] = None,
                 offset: Optional[float] = None):
        self.mimiced_joint_name = 'pr2/brumbrum'
        self.name = name
        self.free_variable_name = free_variable_name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.axis = axis
        self.parent_T_child = parent_T_child
        self.odom_joint = god_map.world.joints[self.mimiced_joint_name]
        if isinstance(self.odom_joint, OmniDrive):
            x_vel = self.odom_joint.x_vel.get_symbol(Derivatives.velocity)
            y_vel = self.odom_joint.y_vel.get_symbol(Derivatives.velocity)
            yaw_vel = self.odom_joint.yaw_vel.get_symbol(Derivatives.velocity)
        elif isinstance(self.odom_joint, OmniDrivePR22):
            forward_velocity = self.odom_joint.forward_vel.get_symbol(Derivatives.velocity)
            yaw1_position = self.odom_joint.yaw1_vel.get_symbol(Derivatives.position)
            yaw2_position = self.odom_joint.yaw.get_symbol(Derivatives.position)
            yaw2_velocity = self.odom_joint.yaw.get_symbol(Derivatives.velocity)
            x_vel = cas.cos(yaw1_position - yaw2_position) * forward_velocity
            y_vel = cas.sin(yaw1_position - yaw2_position) * forward_velocity
            yaw_vel = yaw2_velocity
        # caster_link = god_map.get_world().joints[self.name].child_link_name
        parent_P_child = self.parent_T_child.to_position()
        new_vel_x, new_vel_y = self.pointVel2D(parent_P_child[0],
                                               parent_P_child[1],
                                               x_vel,
                                               y_vel,
                                               yaw_vel)
        steer_angle_desired = cas.if_else(condition=cas.logic_and(cas.equal(x_vel, 0),
                                                                  cas.equal(y_vel, 0),
                                                                  cas.equal(yaw_vel, 0)),
                                          if_result=0,
                                          else_result=cas.atan2(new_vel_y, new_vel_x))

        rotation_axis = cas.Vector3(self.axis)
        parent_R_child = cas.RotationMatrix.from_axis_angle(rotation_axis, steer_angle_desired)
        parent_T_child = cas.TransMatrix(parent_R_child)
        self.parent_T_child = self.parent_T_child.dot(parent_T_child)

    @staticmethod
    def pointVel2D(pos_x, pos_y, vel_x, vel_y, vel_z) -> Tuple[cas.Expression, cas.Expression]:
        new_vel_x = vel_x - pos_y * vel_z
        new_vel_y = vel_y + pos_x * vel_z
        return new_vel_x, new_vel_y

    def get_free_variable_names(self) -> List[PrefixName]:
        return []


class JustinTorso(Joint):
    q1: FreeVariable
    q2: FreeVariable
    q3: cas.Expression

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 parent_T_child: cas.TransMatrix,
                 q1: FreeVariable,
                 q2: FreeVariable):
        self.parent_T_child = parent_T_child
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.axis = axis

        self.q1 = q1
        self.q2 = q2
        self.q3 = -self.q2.get_symbol(Derivatives.position) - self.q1.get_symbol(Derivatives.position)

        rotation_axis = cas.Vector3(self.axis)
        parent_R_child = cas.RotationMatrix.from_axis_angle(rotation_axis, self.q3)
        self.parent_T_child = self.parent_T_child.dot(cas.TransMatrix(parent_R_child))
