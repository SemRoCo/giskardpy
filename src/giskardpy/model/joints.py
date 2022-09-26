import abc
from abc import ABC
from typing import Dict, Tuple, Optional, List

import numpy as np
import urdf_parser_py.urdf as up
from tf.transformations import quaternion_about_axis, quaternion_multiply

import giskardpy.casadi_wrapper as w
from giskardpy import identifier
from giskardpy.data_types import PrefixName
from giskardpy.god_map import GodMap
from giskardpy.my_types import my_string, expr_symbol, expr_matrix, derivative_joint_map, derivative_map
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.utils.math import axis_angle_from_quaternion, quaternion_from_axis_angle, qv_mult


class Joint(ABC):
    name: my_string
    parent_link_name: my_string
    child_link_name: my_string
    god_map: GodMap

    def __init__(self,
                 name: my_string,
                 parent_link_name: my_string,
                 child_link_name: my_string,
                 god_map: GodMap,
                 parent_T_child: expr_matrix):
        if isinstance(name, str):
            name = PrefixName(name, None)
        self.name = name
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.parent_T_child = parent_T_child
        self.god_map = god_map
        self.create_free_variables()

    @property
    def world(self):
        return self.god_map.get_data(identifier.world)

    @property
    def parent_T_child(self):
        return w.dot(self._parent_T_child, self._joint_transformation())

    @parent_T_child.setter
    def parent_T_child(self, value):
        self._parent_T_child = value

    def create_free_variable(self, name: str, lower_limits: derivative_map, upper_limits: derivative_map):
        return FreeVariable(name=name,
                            god_map=self.god_map,
                            lower_limits=lower_limits,
                            upper_limits=upper_limits)

    @abc.abstractmethod
    def create_free_variables(self):
        """
        """

    @abc.abstractmethod
    def _joint_transformation(self):
        """
        modifies self.parent_T_child using free variables
        """
        return w.eye(4)

    @abc.abstractmethod
    def update_state(self, new_cmds: Dict[int, Dict[str, float]], dt: float):
        """
        updates the world state
        :param new_cmds: result from the qp solver
        :param dt: delta time
        """

    @abc.abstractmethod
    def update_limits(self, linear_limits: derivative_joint_map, angular_limits: derivative_joint_map):
        """
        :param linear_limits:
        :param angular_limits:
        """

    @abc.abstractmethod
    def update_weights(self, weights: derivative_joint_map):
        """
        :param weights: maps derivative to weight. e.g. {1:0.001, 2:0, 3:0.001}
        :return:
        """

    @abc.abstractmethod
    def get_limit_expressions(self, order: int) -> Optional[Tuple[expr_symbol, expr_symbol]]:
        """
        """

    @abc.abstractmethod
    def has_free_variables(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def free_variable_list(self) -> List[FreeVariable]:
        pass

    def __str__(self):
        return f'{self.name}: {self.parent_link_name}<-{self.child_link_name}'

    def __repr__(self):
        return str(self)


class DependentJoint(Joint, ABC):
    @abc.abstractmethod
    def connect_to_existing_free_variables(self):
        """

        """


class FixedJoint(Joint):

    def __init__(self, name: my_string, parent_link_name: my_string, child_link_name: my_string,
                 god_map: GodMap, parent_T_child: Optional[expr_matrix] = None):
        if parent_T_child is None:
            parent_T_child = w.eye(4)
        super().__init__(name, parent_link_name, child_link_name, god_map, parent_T_child)

    def get_limit_expressions(self, order: int) -> Optional[Tuple[expr_symbol, expr_symbol]]:
        return None

    def update_limits(self, linear_limits, angular_limits):
        pass

    def update_weights(self, weights: Dict[int, float]):
        pass

    def _joint_transformation(self):
        return w.eye(4)

    def update_state(self, new_cmds: Dict[int, Dict[str, float]], dt: float):
        pass

    def has_free_variables(self) -> bool:
        return False

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return []

    def create_free_variables(self):
        pass


class URDFJoint(Joint, ABC):
    urdf_joint: up.Joint

    def __init__(self, urdf_joint: up.Joint, prefix: str, god_map: GodMap):
        self.urdf_joint = urdf_joint
        joint_name = PrefixName(urdf_joint.name, prefix)
        parent_link_name = PrefixName(urdf_joint.parent, prefix)
        child_link_name = PrefixName(urdf_joint.child, prefix)
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
        parent_T_child = w.dot(w.translation3(*translation_offset),
                               w.rotation_matrix_from_rpy(*rotation_offset))
        super().__init__(name=joint_name, parent_link_name=parent_link_name, child_link_name=child_link_name,
                         god_map=god_map, parent_T_child=parent_T_child)

    @classmethod
    def from_urdf(cls, urdf_joint: up.Joint, prefix: my_string, god_map: GodMap):
        if urdf_joint.type == 'fixed':
            joint_class = FixedURDFJoint
        elif urdf_joint.mimic is not None:
            if urdf_joint.type == 'prismatic':
                joint_class = MimicPrismaticURDFJoint
            elif urdf_joint.type == 'revolute':
                joint_class = MimicRevoluteURDFJoint
            elif urdf_joint.type == 'continuous':
                joint_class = MimicContinuousURDFJoint
            else:
                raise NotImplementedError(
                    f'Joint type \'{urdf_joint.type}\' of \'{urdf_joint.name}\' is not implemented.')
        else:
            if urdf_joint.type == 'prismatic':
                joint_class = PrismaticURDFJoint
            elif urdf_joint.type == 'revolute':
                joint_class = RevoluteURDFJoint
            elif urdf_joint.type == 'continuous':
                joint_class = ContinuousURDFJoint
            else:
                raise NotImplementedError(
                    f'Joint type \'{urdf_joint.type}\' of \'{urdf_joint.name}\' is not implemented.')

        return joint_class(urdf_joint=urdf_joint, prefix=prefix, god_map=god_map)

    def urdf_hard_limits(self):
        lower_limits = {}
        upper_limits = {}
        if not self.urdf_joint.type == 'continuous':
            try:
                lower_limits[0] = self.urdf_joint.limit.lower
                upper_limits[0] = self.urdf_joint.limit.upper
            except AttributeError:
                pass
        try:
            lower_limits[1] = -self.urdf_joint.limit.velocity
            upper_limits[1] = self.urdf_joint.limit.velocity
        except AttributeError:
            pass
        return lower_limits, upper_limits

    def urdf_soft_limits(self):
        lower_limits = {}
        upper_limits = {}
        if not self.urdf_joint.type == 'continuous':
            try:
                lower_limits[0] = max(self.urdf_joint.safety_controller.soft_lower_limit, self.urdf_joint.limit.lower)
                upper_limits[0] = min(self.urdf_joint.safety_controller.soft_upper_limit, self.urdf_joint.limit.upper)
            except AttributeError:
                try:
                    lower_limits[0] = self.urdf_joint.limit.lower
                    upper_limits[0] = self.urdf_joint.limit.upper
                except AttributeError:
                    pass
        return lower_limits, upper_limits


class FixedURDFJoint(URDFJoint, FixedJoint):
    pass


class OneDofJoint(Joint, ABC):
    free_variable: FreeVariable
    axis: Tuple[float, float, float]

    def __init__(self, name: my_string, parent_link_name: my_string, child_link_name: my_string,
                 parent_T_child: expr_matrix,
                 axis: Tuple[float, float, float], lower_limits, upper_limits, god_map: GodMap):
        self.axis = axis
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        Joint.__init__(self, name, parent_link_name, child_link_name, god_map, parent_T_child)

    def create_free_variables(self):
        self.free_variable = self.create_free_variable(self.name, self.lower_limits, self.upper_limits)

    def update_state(self, new_cmds: Dict[int, Dict[str, float]], dt: float):
        world = self.god_map.unsafe_get_data(identifier.world)
        try:
            vel = new_cmds[0][self.free_variable.position_name]
        except KeyError as e:
            # joint is currently not part of the optimization problem
            return
        world.state[self.name].position += vel * dt
        world.state[self.name].velocity = vel
        if len(new_cmds) >= 2:
            acc = new_cmds[1][self.free_variable.position_name]
            world.state[self.name].acceleration = acc
        if len(new_cmds) >= 3:
            jerk = new_cmds[2][self.free_variable.position_name]
            world.state[self.name].jerk = jerk

    @property
    def position_expression(self) -> expr_symbol:
        return self.free_variable.get_symbol(0)

    def delete_limits(self):
        for i in range(1, len(self.free_variable.lower_limits)):
            del self.free_variable.lower_limits[i]
            del self.free_variable.upper_limits[i]

    def update_weights(self, weights: derivative_joint_map):
        # self.delete_weights()
        for order, weight in weights.items():
            try:
                self.free_variable.quadratic_weights[order] = weight[self.name]
            except KeyError:
                # can't do if in, because the dict may be a defaultdict
                pass

    def get_limit_expressions(self, order: int) -> Optional[Tuple[expr_symbol, expr_symbol]]:
        return self.free_variable.get_lower_limit(order), self.free_variable.get_upper_limit(order)

    def has_free_variables(self) -> bool:
        return True

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return [self.free_variable]


class MimicJoint(DependentJoint, OneDofJoint, ABC):

    def __init__(self, name: my_string, parent_link_name: my_string, child_link_name: my_string,
                 parent_T_child: w.ca.SX, axis, lower_limits, upper_limits,
                 mimed_joint_name: my_string, multiplier: float, offset: float, god_map: GodMap):
        try:
            Joint.__init__(self, name, parent_link_name, child_link_name, god_map, parent_T_child)
        except AttributeError as e:
            pass
        self.mimed_joint_name = mimed_joint_name
        self.multiplier = multiplier
        self.offset = offset
        OneDofJoint.__init__(self, name, parent_link_name, child_link_name, parent_T_child, axis,
                             lower_limits, upper_limits, god_map)

    @property
    def position_expression(self) -> expr_symbol:
        multiplier = 1 if self.multiplier is None else self.multiplier
        offset = 0 if self.offset is None else self.offset
        return self.free_variable.get_symbol(0) * multiplier + offset

    def connect_to_existing_free_variables(self):
        mimed_joint: OneDofJoint = self.god_map.unsafe_get_data(identifier.world).joints[self.mimed_joint_name]
        self.free_variable = mimed_joint.free_variable

    def has_free_variables(self) -> bool:
        return False

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return [self.free_variable]

    def delete_limits(self):
        pass

    def delete_weights(self):
        pass

    def update_state(self, new_cmds: Dict[int, Dict[str, float]], dt: float):
        super().update_state(new_cmds, dt)

    def update_weights(self, weights: Dict[int, Dict[my_string, float]]):
        pass

    def get_limit_expressions(self, order: int) -> Optional[Tuple[expr_symbol, expr_symbol]]:
        return None


class PrismaticJoint(OneDofJoint):
    def _joint_transformation(self):
        translation_axis = w.point3(*self.axis) * self.position_expression
        parent_T_child = w.translation3(translation_axis[0], translation_axis[1], translation_axis[2])
        return parent_T_child
        # self.parent_T_child = w.dot(self.parent_T_child, parent_P_child)

    def update_limits(self, linear_limits, angular_limits):
        self.delete_limits()
        for order, linear_limit in linear_limits.items():
            self.free_variable.set_upper_limit(order, linear_limit[self.name])
            self.free_variable.set_lower_limit(order, -linear_limit[self.name])


class RevoluteJoint(OneDofJoint):
    def _joint_transformation(self):
        rotation_axis = w.vector3(*self.axis)
        parent_R_child = w.rotation_matrix_from_axis_angle(rotation_axis, self.position_expression)
        return parent_R_child
        # self.parent_T_child = w.dot(self.parent_T_child, parent_R_child)

    def update_limits(self, linear_limits, angular_limits):
        self.delete_limits()
        for order, angular_limit in angular_limits.items():
            self.free_variable.set_upper_limit(order, angular_limit[self.name])
            self.free_variable.set_lower_limit(order, -angular_limit[self.name])


class ContinuousJoint(RevoluteJoint):
    pass


class OneDofURDFJoint(OneDofJoint, URDFJoint, ABC):
    def __init__(self, urdf_joint: up.Joint, prefix: str, god_map: GodMap):
        try:
            URDFJoint.__init__(self, urdf_joint, prefix, god_map)
        except AttributeError:
            # to be expected, because the next init will set the attributes
            pass
        lower_limits, upper_limits = self.urdf_hard_limits()
        OneDofJoint.__init__(self,
                             name=self.name,
                             parent_link_name=self.parent_link_name,
                             child_link_name=self.child_link_name,
                             parent_T_child=self._parent_T_child,
                             axis=self.urdf_joint.axis,
                             lower_limits=lower_limits,
                             upper_limits=upper_limits,
                             god_map=self.god_map)
        soft_lower_limits, soft_upper_limits = self.urdf_soft_limits()
        if 0 in soft_lower_limits:
            self.free_variable.set_lower_limit(0, soft_lower_limits[0])
        if 0 in soft_upper_limits:
            self.free_variable.set_upper_limit(0, soft_upper_limits[0])


class PrismaticURDFJoint(OneDofURDFJoint, PrismaticJoint):
    pass


class RevoluteURDFJoint(OneDofURDFJoint, RevoluteJoint):
    pass


class ContinuousURDFJoint(OneDofURDFJoint, ContinuousJoint):
    pass


class MimicURDFJoint(MimicJoint, OneDofURDFJoint, ABC):
    def __init__(self, urdf_joint: up.Joint, prefix: str, god_map: GodMap):
        try:
            URDFJoint.__init__(self, urdf_joint, prefix, god_map)
        except AttributeError:
            # to be expected, because the next init will set the attributes
            pass
        lower_limits, upper_limits = self.urdf_hard_limits()
        MimicJoint.__init__(self,
                            name=self.name,
                            parent_link_name=self.parent_link_name,
                            child_link_name=self.child_link_name,
                            parent_T_child=self._parent_T_child,
                            mimed_joint_name=PrefixName(self.urdf_joint.mimic.joint, prefix),
                            multiplier=self.urdf_joint.mimic.multiplier,
                            offset=self.urdf_joint.mimic.offset,
                            god_map=self.god_map,
                            axis=self.urdf_joint.axis,
                            lower_limits=lower_limits,
                            upper_limits=upper_limits)
        # OneDofJoint.__init__(self,
        #                      name=self.name,
        #                      parent_link_name=self.parent_link_name,
        #                      child_link_name=self.child_link_name,
        #                      parent_T_child=self.parent_T_child,
        #                      axis=self.urdf_joint.axis,
        #                      lower_limits=lower_limits,
        #                      upper_limits=upper_limits,
        #                      god_map=self.god_map)


class MimicPrismaticURDFJoint(MimicURDFJoint, PrismaticJoint):
    pass


class MimicRevoluteURDFJoint(MimicURDFJoint, RevoluteJoint):
    pass


class MimicContinuousURDFJoint(MimicURDFJoint, ContinuousJoint):
    pass


class OmniDrive(Joint):
    def __init__(self,
                 god_map: GodMap,
                 parent_link_name: my_string,
                 child_link_name: my_string,
                 name: Optional[my_string] = 'brumbrum',
                 translation_velocity_limit: Optional[float] = 0.5,
                 rotation_velocity_limit: Optional[float] = 0.6,
                 translation_acceleration_limit: Optional[float] = None,
                 rotation_acceleration_limit: Optional[float] = None,
                 translation_jerk_limit: Optional[float] = 5,
                 rotation_jerk_limit: Optional[float] = 10,
                 **kwargs):
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.translation_acceleration_limit = translation_acceleration_limit
        self.rotation_acceleration_limit = rotation_acceleration_limit
        self.translation_jerk_limit = translation_jerk_limit
        self.rotation_jerk_limit = rotation_jerk_limit
        self.translation_names = ['odom_x', 'odom_y', 'odom_z']
        # self.orientation_names = ['odom_qx', 'odom_qy', 'odom_qz', 'odom_qw']
        self.orientation_names = ['roll', 'pitch', 'yaw']
        # self.rot_name = 'odom_rot'
        self.x_vel_name = 'odom_x_vel'
        self.y_vel_name = 'odom_y_vel'
        self.rot_vel_name = 'odom_yaw_vel'
        self.translation_variables: List[FreeVariable] = []
        self.orientation_variables: List[FreeVariable] = []
        super().__init__(name, parent_link_name, child_link_name, god_map, w.eye(4))

    def create_free_variables(self):
        translation_upper_limits = {}
        if self.translation_velocity_limit is not None:
            translation_upper_limits[1] = self.translation_velocity_limit
        if self.translation_acceleration_limit is not None:
            translation_upper_limits[2] = self.translation_acceleration_limit
        if self.translation_jerk_limit is not None:
            translation_upper_limits[3] = self.translation_jerk_limit
        translation_lower_limits = {k: -v for k, v in translation_upper_limits.items()}

        rotation_upper_limits = {}
        if self.rotation_velocity_limit is not None:
            rotation_upper_limits[1] = self.rotation_velocity_limit
        if self.rotation_acceleration_limit is not None:
            rotation_upper_limits[2] = self.rotation_acceleration_limit
        if self.rotation_jerk_limit is not None:
            rotation_upper_limits[3] = self.rotation_jerk_limit
        rotation_lower_limits = {k: -v for k, v in rotation_upper_limits.items()}

        for translation_variable_name in self.translation_names:
            self.translation_variables.append(self.create_free_variable(name=translation_variable_name,
                                                                        lower_limits=translation_lower_limits,
                                                                        upper_limits=translation_upper_limits))

        for orientation_variable_name in self.orientation_names:
            self.orientation_variables.append(self.create_free_variable(name=orientation_variable_name,
                                                                        lower_limits=rotation_lower_limits,
                                                                        upper_limits=rotation_upper_limits))
        self.x_vel = self.create_free_variable(self.x_vel_name,
                                               translation_lower_limits,
                                               translation_upper_limits)
        self.y_vel = self.create_free_variable(self.y_vel_name,
                                               translation_lower_limits,
                                               translation_upper_limits)
        self.yaw = self.create_free_variable(self.yaw_name,
                                             rotation_lower_limits,
                                             rotation_upper_limits)
        self.yaw_vel = self.create_free_variable(self.rot_vel_name,
                                                 rotation_lower_limits,
                                                 rotation_upper_limits)

    def _joint_transformation(self):
        odom_T_base_footprint = w.frame_from_x_y_rot(self.x.get_symbol(0),
                                                     self.y.get_symbol(0),
                                                     self.yaw.get_symbol(0))
        base_footprint_T_base_footprint_vel = w.frame_from_x_y_rot(self.x_vel.get_symbol(0),
                                                                   self.y_vel.get_symbol(0),
                                                                   self.yaw_vel.get_symbol(0))
        base_footprint_vel_T_base_footprint = w.frame_rpy(x=0,
                                                          y=0,
                                                          z=self.translation_variables[2].get_symbol(0),
                                                          roll=self.orientation_variables[0].get_symbol(0),
                                                          pitch=self.orientation_variables[1].get_symbol(0),
                                                          yaw=0)
        return w.dot(odom_T_base_footprint, base_footprint_T_base_footprint_vel, base_footprint_vel_T_base_footprint)

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
                vel = new_cmds[0][free_variable.position_name]
            except KeyError as e:
                # joint is currently not part of the optimization problem
                continue
            state[free_variable.name].velocity = vel
            if len(new_cmds) >= 2:
                acc = new_cmds[1][free_variable.position_name]
                state[free_variable.name].acceleration = acc
            if len(new_cmds) >= 3:
                jerk = new_cmds[2][free_variable.position_name]
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

    def get_limit_expressions(self, order: int) -> Optional[Tuple[expr_symbol, expr_symbol]]:
        pass

    def has_free_variables(self) -> bool:
        return True

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return [self.x_vel, self.y_vel, self.yaw_vel]

    def _all_symbols(self) -> List[FreeVariable]:
        return self.free_variable_list + self.translation_variables + self.orientation_variables


class DiffDrive(Joint):
    x: FreeVariable
    y: FreeVariable
    rot: FreeVariable
    x_vel: FreeVariable
    rot_vel: FreeVariable

    def __init__(self,
                 god_map: GodMap,
                 parent_link_name: my_string,
                 child_link_name: my_string,
                 name: Optional[my_string] = 'brumbrum',
                 translation_velocity_limit: Optional[float] = 0.5,
                 rotation_velocity_limit: Optional[float] = 0.6,
                 translation_acceleration_limit: Optional[float] = None,
                 rotation_acceleration_limit: Optional[float] = None,
                 translation_jerk_limit: Optional[float] = 5,
                 rotation_jerk_limit: Optional[float] = 10,
                 x_name: Optional[str] = 'odom_x',
                 y_name: Optional[str] = 'odom_y',
                 rot_name: Optional[str] = 'odom_rot',
                 x_vel_name: Optional[str] = 'base_footprint_x_vel',
                 rot_vel_name: Optional[str] = 'base_footprint_rot_vel',
                 **kwargs):
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.translation_acceleration_limit = translation_acceleration_limit
        self.rotation_acceleration_limit = rotation_acceleration_limit
        self.translation_jerk_limit = translation_jerk_limit
        self.rotation_jerk_limit = rotation_jerk_limit
        self.x_name = x_name
        self.y_name = y_name
        self.rot_name = rot_name
        self.x_vel_name = x_vel_name
        self.rot_vel_name = rot_vel_name
        super().__init__(name, parent_link_name, child_link_name, god_map, w.eye(4))

    def create_free_variables(self):
        translation_upper_limits = {}
        if self.translation_velocity_limit is not None:
            translation_upper_limits[1] = self.translation_velocity_limit
        if self.translation_acceleration_limit is not None:
            translation_upper_limits[2] = self.translation_acceleration_limit
        if self.translation_jerk_limit is not None:
            translation_upper_limits[3] = self.translation_jerk_limit
        translation_lower_limits = {k: -v for k, v in translation_upper_limits.items()}

        rotation_upper_limits = {}
        if self.rotation_velocity_limit is not None:
            rotation_upper_limits[1] = self.rotation_velocity_limit
        if self.rotation_acceleration_limit is not None:
            rotation_upper_limits[2] = self.rotation_acceleration_limit
        if self.rotation_jerk_limit is not None:
            rotation_upper_limits[3] = self.rotation_jerk_limit
        rotation_lower_limits = {k: -v for k, v in rotation_upper_limits.items()}

        self.x = self.create_free_variable(self.x_name,
                                           translation_lower_limits,
                                           translation_upper_limits)
        self.y = self.create_free_variable(self.y_name,
                                           translation_lower_limits,
                                           translation_upper_limits)
        self.rot = self.create_free_variable(self.rot_name,
                                             rotation_lower_limits,
                                             rotation_upper_limits)
        self.x_vel = self.create_free_variable(self.x_vel_name,
                                               translation_lower_limits,
                                               translation_upper_limits)
        self.rot_vel = self.create_free_variable(self.rot_vel_name,
                                                 rotation_lower_limits,
                                                 rotation_upper_limits)

    def _joint_transformation(self):
        odom_T_base_footprint = w.frame_from_x_y_rot(self.x.get_symbol(0),
                                                     self.y.get_symbol(0),
                                                     self.rot.get_symbol(0))
        base_footprint_T_base_footprint_vel = w.frame_from_x_y_rot(self.x_vel.get_symbol(0),
                                                                   0,
                                                                   self.rot_vel.get_symbol(0))
        return w.dot(odom_T_base_footprint, base_footprint_T_base_footprint_vel)

    def update_state(self, new_cmds: derivative_joint_map, dt: float):
        world = self.god_map.unsafe_get_data(identifier.world)
        for free_variable in self.free_variable_list:
            try:
                vel = new_cmds[0][free_variable.position_name]
            except KeyError as e:
                # joint is currently not part of the optimization problem
                continue
            world.state[free_variable.name].velocity = vel
            if len(new_cmds) >= 2:
                acc = new_cmds[1][free_variable.position_name]
                world.state[free_variable.name].acceleration = acc
            if len(new_cmds) >= 3:
                jerk = new_cmds[2][free_variable.position_name]
                world.state[free_variable.name].jerk = jerk
        x = world.state[self.x_vel_name].velocity
        rot = world.state[self.rot_vel_name].velocity
        delta = world.state[self.rot_name].position
        world.state[self.x_name].velocity = (np.cos(delta) * x)
        world.state[self.x_name].position += world.state[self.x_name].velocity * dt
        world.state[self.y_name].velocity = (np.sin(delta) * x)
        world.state[self.y_name].position += world.state[self.y_name].velocity * dt
        world.state[self.rot_name].velocity = rot
        world.state[self.rot_name].position += rot * dt

    def update_limits(self, linear_limits: derivative_joint_map, angular_limits: derivative_joint_map):
        for free_variable in self._all_symbols():
            free_variable.lower_limits = {}
            free_variable.upper_limits = {}

        for order, linear_limit in linear_limits.items():
            # self.x.set_upper_limit(order, linear_limit[self.x_name])
            # self.y.set_upper_limit(order, linear_limit[self.y_name])
            self.x_vel.set_upper_limit(order, linear_limit[self.x_vel_name])

            # self.x.set_lower_limit(order, -linear_limit[self.x_name])
            # self.y.set_lower_limit(order, -linear_limit[self.y_name])
            self.x_vel.set_lower_limit(order, -linear_limit[self.x_vel_name])

        for order, angular_limit in angular_limits.items():
            # self.rot.set_upper_limit(order, angular_limit[self.rot_name])
            # self.rot.set_lower_limit(order, -angular_limit[self.rot_name])
            self.rot_vel.set_upper_limit(order, angular_limit[self.rot_vel_name])
            self.rot_vel.set_lower_limit(order, -angular_limit[self.rot_vel_name])

    def update_weights(self, weights: derivative_joint_map):
        # self.delete_weights()
        for order, weight in weights.items():
            try:
                for free_variable in self._all_symbols():
                    free_variable.quadratic_weights[order] = weight[self.name]
            except KeyError:
                # can't do if in, because the dict may be a defaultdict
                pass

    def get_limit_expressions(self, order: int) -> Optional[Tuple[expr_symbol, expr_symbol]]:
        pass

    def has_free_variables(self) -> bool:
        return True

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return [self.x_vel, self.rot_vel]

    def _all_symbols(self) -> List[FreeVariable]:
        return self.free_variable_list + [self.x, self.y, self.rot]
