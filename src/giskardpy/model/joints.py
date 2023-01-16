from __future__ import annotations
import abc
from abc import ABC
from typing import Dict, Tuple, Optional, List, Union, Type
from functools import cached_property
import numpy as np
import urdf_parser_py.urdf as up

import giskardpy.casadi_wrapper as w
from giskardpy import identifier
from giskardpy.data_types import JointStates
from giskardpy.my_types import PrefixName, Derivatives
from giskardpy.my_types import my_string, derivative_joint_map, derivative_map
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.utils.utils import blackboard_god_map


def urdf_joint_to_class(urdf_joint: up.Joint) -> Union[Type[FixedJoint], Type[RevoluteJoint],
                                                       Type[PrismaticJoint], Type[MimicRevoluteJoint],
                                                       Type[MimicContinuousJoint], Type[MimicPrismaticJoint]]:
    if urdf_joint.type == 'fixed':
        joint_class = FixedJoint
    elif urdf_joint.mimic is not None:
        if urdf_joint.type == 'prismatic':
            joint_class = MimicPrismaticJoint
        elif urdf_joint.type == 'revolute':
            joint_class = MimicRevoluteJoint
        elif urdf_joint.type == 'continuous':
            joint_class = MimicContinuousJoint
        else:
            raise NotImplementedError(
                f'Joint type \'{urdf_joint.type}\' of \'{urdf_joint.name}\' is not implemented.')
    else:
        if urdf_joint.type == 'prismatic':
            joint_class = PrismaticJoint
        elif urdf_joint.type == 'revolute':
            joint_class = RevoluteJoint
        elif urdf_joint.type == 'continuous':
            if 'caster_rotation' in urdf_joint.name:
                joint_class = PR2CasterJoint
            else:
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
    return lower_limits, upper_limits


def urdf_to_joint(urdf_joint: up.Joint, prefix: str) -> Union[FixedJoint, RevoluteJoint,
                                                              PrismaticJoint, MimicRevoluteJoint, MimicContinuousJoint,
                                                              MimicPrismaticJoint]:
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
    parent_T_child = w.TransMatrix.from_xyz_rpy(x=translation_offset[0],
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
    elif not issubclass(joint_class, DependentJoint):
        lower_limits, upper_limits = urdf_joint_to_limits(urdf_joint)
        return joint_class(name=joint_name,
                           parent_link_name=parent_link_name,
                           child_link_name=child_link_name,
                           parent_T_child=parent_T_child,
                           lower_limits=lower_limits,
                           upper_limits=upper_limits)
    else:
        lower_limits, upper_limits = urdf_joint_to_limits(urdf_joint)
        multiplier = urdf_joint.mimic.multiplier
        offset = urdf_joint.mimic.offset
        return joint_class(name=joint_name,
                           parent_link_name=parent_link_name,
                           child_link_name=child_link_name,
                           parent_T_child=parent_T_child,
                           lower_limits=lower_limits,
                           upper_limits=upper_limits,
                           multiplier=multiplier,
                           offset=offset)


class Joint(ABC):
    name: PrefixName
    parent_link_name: PrefixName
    child_link_name: PrefixName

    @property
    @abc.abstractmethod
    def parent_T_child(self) -> w.TransMatrix: ...

    @property
    def god_map(self):
        return blackboard_god_map()

    @property
    def world(self):
        return self.god_map.get_data(identifier.world)

    @abc.abstractmethod
    def update_parent_T_child(self, new_parent_T_child: w.TransMatrix): ...

    def __str__(self):
        return f'{self.name}: {self.parent_link_name}<-{self.child_link_name}'

    def __repr__(self):
        return str(self)


class DependentJoint(ABC): ...


class ActuatedJoint(Joint, ABC):
    free_variables: List[FreeVariable]

    @property
    def world_state(self):
        return self.world.world_state

    @abc.abstractmethod
    def set_initial_state(self):
        ...

    @abc.abstractmethod
    def update_state(self, new_cmds: Dict[int, Dict[str, float]], dt: float):
        ...

    @abc.abstractmethod
    def update_derivative_limits(self, derivative: Derivatives, new_limit: w.symbol_expr_float):
        ...

    @abc.abstractmethod
    def update_derivative_weight(self, derivative: Derivatives, new_weight: w.symbol_expr_float):
        ...

    def clear_cache(self):
        try:
            del self.parent_T_child
        except AttributeError as e:
            pass  # parent_T_child hasn't been called yet


class OneDofJoint(ActuatedJoint):
    _parent_T_child: w.TransMatrix

    def update_state(self, new_cmds: Dict[int, Dict[str, float]], dt: float):
        for free_variable in self.free_variables:
            try:
                vel = new_cmds[Derivatives.velocity][free_variable.position_name]
            except KeyError as e:
                # joint is currently not part of the optimization problem
                continue
            self.world_state[free_variable.name][Derivatives.position] += vel * dt
            self.world_state[free_variable.name][Derivatives.velocity] = vel
            for derivative, cmd in new_cmds.items():
                cmd_ = cmd[free_variable.position_name]
                self.world_state[free_variable.name][derivative] = cmd_

    def update_parent_T_child(self, new_parent_T_child: w.TransMatrix):
        del self.parent_T_child
        self._parent_T_child = new_parent_T_child

    def update_derivative_limits(self, derivative: Derivatives, new_limit: w.symbol_expr_float):
        self.free_variables[0].set_lower_limit(derivative=derivative, limit=-new_limit)
        self.free_variables[0].set_upper_limit(derivative=derivative, limit=new_limit)

    def update_derivative_weight(self, derivative: Derivatives, new_weight: w.symbol_expr_float):
        self.free_variables[0].quadratic_weights[derivative] = new_weight


class FixedJoint(Joint):
    def __init__(self, name: PrefixName, parent_link_name: PrefixName, child_link_name: PrefixName,
                 parent_T_child: w.TransMatrix):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self._parent_T_child = parent_T_child

    def update_parent_T_child(self, new_parent_T_child: w.TransMatrix):
        self._parent_T_child = new_parent_T_child

    @property
    def parent_T_child(self) -> w.TransMatrix:
        return self._parent_T_child


class RevoluteJoint(OneDofJoint):
    axis: Tuple[float, float, float]

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 parent_T_child: w.TransMatrix,
                 lower_limits: derivative_map,
                 upper_limits: derivative_map):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self._parent_T_child = parent_T_child
        self.axis = axis
        self.rotation = FreeVariable(name=self.name,
                                     god_map=self.god_map,
                                     lower_limits=lower_limits,
                                     upper_limits=upper_limits)
        self.free_variables.append(self.rotation)

    @cached_property
    def parent_T_child(self):
        rotation_axis = w.Vector3(self.axis)
        parent_R_child = w.RotationMatrix.from_axis_angle(rotation_axis, self.rotation.get_symbol(Derivatives.position))
        return self._parent_T_child.dot(w.TransMatrix(parent_R_child))

    def set_initial_state(self):
        lower_limit = self.rotation.get_lower_limit(derivative=Derivatives.position,
                                                    evaluated=True)
        upper_limit = self.rotation.get_upper_limit(derivative=Derivatives.position,
                                                    evaluated=True)
        center = (upper_limit + lower_limit) / 2
        self.world.world_state[self.rotation.name].position = center


class PrismaticJoint(OneDofJoint):
    axis: Tuple[float, float, float]

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 parent_T_child: w.TransMatrix,
                 lower_limits: derivative_map,
                 upper_limits: derivative_map):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self._parent_T_child = parent_T_child
        self.axis = axis
        self.translation = FreeVariable(name=self.name,
                                        god_map=self.god_map,
                                        lower_limits=lower_limits,
                                        upper_limits=upper_limits)
        self.free_variables.append(self.translation)

    @property
    def parent_T_child(self) -> w.TransMatrix:
        translation_axis = w.Point3(self.axis) * self.translation.get_symbol(Derivatives.position)
        parent_T_child = w.TransMatrix.from_xyz_rpy(x=translation_axis[0],
                                                    y=translation_axis[1],
                                                    z=translation_axis[2])
        return self._parent_T_child.dot(parent_T_child)

    def set_initial_state(self):
        lower_limit = self.translation.get_lower_limit(derivative=Derivatives.position,
                                                       evaluated=True)
        upper_limit = self.translation.get_upper_limit(derivative=Derivatives.position,
                                                       evaluated=True)
        center = (upper_limit + lower_limit) / 2
        self.world.world_state[self.translation.name].position = center


# class ContinuousJoint(OneDofJoint):
#     axis: Tuple[float, float, float]
#
#     def __init__(self,
#                  name: PrefixName,
#                  parent_link_name: PrefixName,
#                  child_link_name: PrefixName,
#                  axis: Tuple[float, float, float],
#                  parent_T_child: w.TransMatrix,
#                  lower_limits: derivative_map,
#                  upper_limits: derivative_map):
#         self.name = name
#         self.parent_link_name = parent_link_name
#         self.child_link_name = child_link_name
#         self._parent_T_child = parent_T_child
#         self.axis = axis


class MimicRevoluteJoint(Joint, DependentJoint):
    axis: Tuple[float, float, float]

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 parent_T_child: w.TransMatrix,
                 lower_limits: derivative_map,
                 upper_limits: derivative_map,
                 multiplier: float,
                 offset: float):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self._parent_T_child = parent_T_child
        self.axis = axis
        self.multiplier = multiplier
        self.offset = offset


class MimicPrismaticJoint(Joint, DependentJoint):
    axis: Tuple[float, float, float]

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 parent_T_child: w.TransMatrix,
                 lower_limits: derivative_map,
                 upper_limits: derivative_map,
                 multiplier: float,
                 offset: float):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self._parent_T_child = parent_T_child
        self.axis = axis
        self.multiplier = multiplier
        self.offset = offset


class MimicContinuousJoint(Joint, DependentJoint):
    axis: Tuple[float, float, float]

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 parent_T_child: w.TransMatrix,
                 lower_limits: derivative_map,
                 upper_limits: derivative_map,
                 multiplier: float,
                 offset: float):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self._parent_T_child = parent_T_child
        self.axis = axis
        self.multiplier = multiplier
        self.offset = offset


class PR2CasterJoint(Joint, DependentJoint):
    axis: Tuple[float, float, float]

    def __init__(self,
                 name: PrefixName,
                 parent_link_name: PrefixName,
                 child_link_name: PrefixName,
                 axis: Tuple[float, float, float],
                 parent_T_child: w.TransMatrix,
                 lower_limits: derivative_map,
                 upper_limits: derivative_map,
                 multiplier: float,
                 offset: float):
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self._parent_T_child = parent_T_child
        self.axis = axis
        self.multiplier = multiplier
        self.offset = offset


class OmniDrive(ActuatedJoint):
    def __init__(self,
                 parent_link_name: my_string,
                 child_link_name: my_string,
                 name: Optional[my_string] = 'brumbrum',
                 group_name: Optional[str] = None,
                 translation_velocity_limit: Optional[float] = 0.5,
                 rotation_velocity_limit: Optional[float] = 0.6,
                 translation_acceleration_limit: Optional[float] = None,
                 rotation_acceleration_limit: Optional[float] = None,
                 translation_jerk_limit: Optional[float] = 5,
                 rotation_jerk_limit: Optional[float] = 10,
                 odom_x_name: Optional[str] = None,
                 odom_y_name: Optional[str] = None,
                 odom_yaw_name: Optional[str] = None,
                 **kwargs):
        self.name = PrefixName(name, group_name)
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.translation_acceleration_limit = translation_acceleration_limit
        self.rotation_acceleration_limit = rotation_acceleration_limit
        self.translation_jerk_limit = translation_jerk_limit
        self.rotation_jerk_limit = rotation_jerk_limit
        self.translation_names = [PrefixName('odom_x', group_name),
                                  PrefixName('odom_y', group_name),
                                  PrefixName('odom_z', group_name)]
        if odom_x_name is not None:
            self.translation_names[0] = PrefixName(odom_x_name, group_name)
        if odom_y_name is not None:
            self.translation_names[1] = PrefixName(odom_y_name, group_name)
        self.orientation_names = [PrefixName('odom_roll', group_name),
                                  PrefixName('odom_pitch', group_name),
                                  PrefixName('odom_yaw', group_name)]
        if odom_yaw_name is not None:
            self.orientation_names[2] = PrefixName(odom_yaw_name, group_name)
        # self.orientation_names = ['odom_qx', 'odom_qy', 'odom_qz', 'odom_qw']
        # self.rot_name = 'odom_rot'
        self.x_vel_name = PrefixName('odom_x_vel', group_name)
        self.y_vel_name = PrefixName('odom_y_vel', group_name)
        self.rot_vel_name = PrefixName('odom_yaw_vel', group_name)
        self.translation_variables: List[FreeVariable] = []
        self.orientation_variables: List[FreeVariable] = []

    def create_free_variables(self):
        translation_upper_limits = {}
        if self.translation_velocity_limit is not None:
            translation_upper_limits[Derivatives.velocity] = self.translation_velocity_limit
        if self.translation_acceleration_limit is not None:
            translation_upper_limits[Derivatives.acceleration] = self.translation_acceleration_limit
        if self.translation_jerk_limit is not None:
            translation_upper_limits[Derivatives.jerk] = self.translation_jerk_limit
        translation_lower_limits = {k: -v for k, v in translation_upper_limits.items()}

        rotation_upper_limits = {}
        if self.rotation_velocity_limit is not None:
            rotation_upper_limits[Derivatives.velocity] = self.rotation_velocity_limit
        if self.rotation_acceleration_limit is not None:
            rotation_upper_limits[Derivatives.acceleration] = self.rotation_acceleration_limit
        if self.rotation_jerk_limit is not None:
            rotation_upper_limits[Derivatives.jerk] = self.rotation_jerk_limit
        rotation_lower_limits = {k: -v for k, v in rotation_upper_limits.items()}

        for translation_variable_name in self.translation_names:
            self.translation_variables.append(FreeVariable(name=translation_variable_name,
                                                           god_map=self.god_map,
                                                           lower_limits=translation_lower_limits,
                                                           upper_limits=translation_upper_limits))

        for orientation_variable_name in self.orientation_names:
            self.orientation_variables.append(FreeVariable(name=orientation_variable_name,
                                                           god_map=self.god_map,
                                                           lower_limits=rotation_lower_limits,
                                                           upper_limits=rotation_upper_limits))
        self.yaw = self.orientation_variables[-1]
        self.x_vel = FreeVariable(name=self.x_vel_name,
                                  god_map=self.god_map,
                                  lower_limits=translation_lower_limits,
                                  upper_limits=translation_upper_limits)
        self.y_vel = FreeVariable(name=self.y_vel_name,
                                  god_map=self.god_map,
                                  lower_limits=translation_lower_limits,
                                  upper_limits=translation_upper_limits)
        self.yaw_vel = FreeVariable(name=self.rot_vel_name,
                                    god_map=self.god_map,
                                    lower_limits=rotation_lower_limits,
                                    upper_limits=rotation_upper_limits)

    @property
    def position_variable_names(self):
        return [self.x_name, self.y_name, self.yaw_name]

    @profile
    def _joint_transformation(self):
        odom_T_bf = w.TransMatrix.from_xyz_rpy(x=self.x.get_symbol(Derivatives.position),
                                               y=self.y.get_symbol(Derivatives.position),
                                               yaw=self.yaw.get_symbol(Derivatives.position))
        bf_T_bf_vel = w.TransMatrix.from_xyz_rpy(x=self.x_vel.get_symbol(Derivatives.position),
                                                 y=self.y_vel.get_symbol(Derivatives.position),
                                                 yaw=self.yaw_vel.get_symbol(Derivatives.position))
        bf_vel_T_bf = w.TransMatrix.from_xyz_rpy(x=0,
                                                 y=0,
                                                 z=self.translation_variables[2].get_symbol(Derivatives.position),
                                                 roll=self.orientation_variables[0].get_symbol(Derivatives.position),
                                                 pitch=self.orientation_variables[1].get_symbol(Derivatives.position),
                                                 yaw=0)
        return odom_T_bf.dot(bf_T_bf_vel).dot(bf_vel_T_bf)

    @property
    def x(self):
        return self.translation_variables[0]

    @property
    def y(self):
        return self.translation_variables[1]

    @property
    def z(self):
        return self.translation_variables[2]

    @property
    def x_name(self):
        return self.translation_names[0]

    @property
    def y_name(self):
        return self.translation_names[1]

    @property
    def z_name(self):
        return self.translation_names[2]

    @property
    def roll_name(self):
        return self.orientation_names[0]

    @property
    def pitch_name(self):
        return self.orientation_names[1]

    @property
    def yaw_name(self):
        return self.orientation_names[2]

    def update_state(self, new_cmds: derivative_joint_map, dt: float):
        state = self.world.state
        for free_variable in self.free_variable_list:
            try:
                vel = new_cmds[Derivatives.velocity][free_variable.position_name]
            except KeyError as e:
                # joint is currently not part of the optimization problem
                continue
            state[free_variable.name].velocity = vel
            if len(new_cmds) >= 2:
                acc = new_cmds[Derivatives.acceleration][free_variable.position_name]
                state[free_variable.name].acceleration = acc
            if len(new_cmds) >= 3:
                jerk = new_cmds[Derivatives.jerk][free_variable.position_name]
                state[free_variable.name].jerk = jerk
        x_vel = state[self.x_vel_name].velocity
        y_vel = state[self.y_vel_name].velocity
        rot_vel = state[self.rot_vel_name].velocity
        delta = state[self.yaw_name].position
        state[self.x_name].velocity = (np.cos(delta) * x_vel - np.sin(delta) * y_vel)
        state[self.x_name].position += state[self.x_name].velocity * dt
        state[self.y_name].velocity = (np.sin(delta) * x_vel + np.cos(delta) * y_vel)
        state[self.y_name].position += state[self.y_name].velocity * dt
        state[self.yaw_name].velocity = rot_vel
        state[self.yaw_name].position += rot_vel * dt

    def set_initial_state(self):
        pass

    def update_derivative_limits(self, derivative: Derivatives, new_limit: w.symbol_expr_float):
        pass

    def update_derivative_weight(self, derivative: Derivatives, new_weight: w.symbol_expr_float):
        pass

    def update_limits(self, linear_limits: derivative_joint_map, angular_limits: derivative_joint_map):
        for free_variable in self._all_symbols():
            free_variable.lower_limits = {}
            free_variable.upper_limits = {}

        for order, linear_limit in linear_limits.items():
            self.x_vel.set_upper_limit(order, linear_limit[self.x_vel_name])
            self.y_vel.set_upper_limit(order, linear_limit[self.y_vel_name])

            self.x_vel.set_lower_limit(order, -linear_limit[self.x_vel_name])
            self.y_vel.set_lower_limit(order, -linear_limit[self.y_vel_name])

        for order, angular_limit in angular_limits.items():
            self.yaw_vel.set_upper_limit(order, angular_limit[self.rot_vel_name])
            self.yaw_vel.set_lower_limit(order, -angular_limit[self.rot_vel_name])

    def update_weights(self, weights: derivative_joint_map):
        # self.delete_weights()
        for order, weight in weights.items():
            try:
                for free_variable in self._all_symbols():
                    free_variable.quadratic_weights[order] = weight[self.name]
            except KeyError:
                # can't do "if in", because the dict may be a defaultdict
                pass

    def get_limit_expressions(self, order: int) -> Tuple[
        Optional[Union[w.Symbol, float]], Optional[Union[w.Symbol, float]]]:
        return None, None

    def has_free_variables(self) -> bool:
        return True

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return [self.x_vel, self.y_vel, self.yaw_vel]

    def _all_symbols(self) -> List[FreeVariable]:
        return self.free_variable_list + self.translation_variables + self.orientation_variables


class DiffDrive(ActuatedJoint):
    pass