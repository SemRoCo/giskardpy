import abc
from abc import ABC
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import urdf_parser_py.urdf as up

import giskardpy.casadi_wrapper as w
from giskardpy import identifier
from giskardpy.my_types import PrefixName, Derivatives
from giskardpy.my_types import my_string, derivative_joint_map, derivative_map
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.utils.utils import blackboard_god_map


class Joint(ABC):
    def __init__(self,
                 name: my_string,
                 parent_link_name: my_string,
                 child_link_name: my_string,
                 parent_T_child: w.TransMatrix):
        if isinstance(name, str):
            name = PrefixName(name, None)
        self.name: my_string = name
        self.parent_link_name: my_string = parent_link_name
        self.child_link_name: my_string = child_link_name
        self.parent_T_child: w.TransMatrix = w.TransMatrix(parent_T_child)
        self.create_free_variables()

    @property
    def god_map(self):
        return blackboard_god_map()

    @property
    def world(self):
        return self.god_map.get_data(identifier.world)

    @property
    def parent_T_child(self) -> w.TransMatrix:
        return self._parent_T_child.dot(self._joint_transformation())

    @parent_T_child.setter
    def parent_T_child(self, value: w.TransMatrix):
        self._parent_T_child = value

    def create_free_variable(self, name: my_string, lower_limits: derivative_map, upper_limits: derivative_map):
        return FreeVariable(name=name,
                            god_map=self.god_map,
                            lower_limits=lower_limits,
                            upper_limits=upper_limits)

    @abc.abstractmethod
    def create_free_variables(self):
        """
        """

    @abc.abstractmethod
    def _joint_transformation(self) -> w.TransMatrix:
        """
        modifies self.parent_T_child using free variables
        """
        return w.TransMatrix()

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
    def get_limit_expressions(self, order: int) -> Optional[Tuple[w.Expression, w.Expression]]:
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
                 parent_T_child: Optional[w.TransMatrix] = None):
        if parent_T_child is None:
            parent_T_child = w.eye(4)
        super().__init__(name, parent_link_name, child_link_name, parent_T_child)

    def get_limit_expressions(self, order: int) -> Optional[Tuple[w.Expression, w.Expression]]:
        return None

    def update_limits(self, linear_limits, angular_limits):
        pass

    def update_weights(self, weights: Dict[int, float]):
        pass

    def _joint_transformation(self) -> w.TransMatrix:
        return w.TransMatrix()

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

    def __init__(self, urdf_joint: up.Joint, prefix: str):
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
        parent_T_child = w.TransMatrix.from_xyz_rpy(x=translation_offset[0],
                                                    y=translation_offset[1],
                                                    z=translation_offset[2],
                                                    roll=rotation_offset[0],
                                                    pitch=rotation_offset[1],
                                                    yaw=rotation_offset[2])
        super().__init__(name=joint_name, parent_link_name=parent_link_name, child_link_name=child_link_name,
                         parent_T_child=parent_T_child)

    @classmethod
    def from_urdf(cls, urdf_joint: up.Joint, prefix: my_string):
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
                # if 'caster_rotation' in urdf_joint.name:
                #     joint_class = PR2CasterJoint
                # else:
                joint_class = ContinuousURDFJoint
            else:
                raise NotImplementedError(
                    f'Joint type \'{urdf_joint.type}\' of \'{urdf_joint.name}\' is not implemented.')

        return joint_class(urdf_joint=urdf_joint, prefix=prefix)

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
                 parent_T_child: w.TransMatrix,
                 axis: Tuple[float, float, float], lower_limits, upper_limits):
        self.axis = axis
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        Joint.__init__(self, name, parent_link_name, child_link_name, parent_T_child)

    def create_free_variables(self):
        self.free_variable = self.create_free_variable(self.name, self.lower_limits, self.upper_limits)

    def update_state(self, new_cmds: derivative_joint_map, dt: float):
        world = self.god_map.unsafe_get_data(identifier.world)
        try:
            vel = new_cmds[Derivatives.velocity][self.free_variable.position_name]
        except KeyError as e:
            # joint is currently not part of the optimization problem
            return
        world.state[self.name].position += vel * dt
        world.state[self.name].velocity = vel
        if len(new_cmds) >= 2:
            acc = new_cmds[Derivatives.acceleration][self.free_variable.position_name]
            world.state[self.name].acceleration = acc
        if len(new_cmds) >= 3:
            jerk = new_cmds[Derivatives.jerk][self.free_variable.position_name]
            world.state[self.name].jerk = jerk

    @property
    def position_expression(self) -> Union[w.Symbol, float]:
        return self.free_variable.get_symbol(Derivatives.position)

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

    def get_limit_expressions(self, order: Derivatives) -> Optional[Tuple[w.Expression, w.Expression]]:
        return self.free_variable.get_lower_limit(order), self.free_variable.get_upper_limit(order)

    def has_free_variables(self) -> bool:
        return True

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return [self.free_variable]


class MimicJoint(DependentJoint, OneDofJoint, ABC):

    def __init__(self, name: my_string, parent_link_name: my_string, child_link_name: my_string,
                 parent_T_child: w.TransMatrix, axis, lower_limits, upper_limits,
                 mimed_joint_name: my_string, multiplier: float, offset: float):
        try:
            Joint.__init__(self, name, parent_link_name, child_link_name, parent_T_child)
        except AttributeError as e:
            pass
        self.mimed_joint_name = mimed_joint_name
        self.multiplier = multiplier
        self.offset = offset
        OneDofJoint.__init__(self, name, parent_link_name, child_link_name, parent_T_child, axis,
                             lower_limits, upper_limits)

    @property
    def position_expression(self) -> Union[w.Symbol, float]:
        multiplier = 1 if self.multiplier is None else self.multiplier
        offset = 0 if self.offset is None else self.offset
        return self.free_variable.get_symbol(Derivatives.position) * multiplier + offset

    def connect_to_existing_free_variables(self):
        mimed_joint: OneDofJoint = self.god_map.unsafe_get_data(identifier.world)._joints[self.mimed_joint_name]
        self.free_variable = mimed_joint.free_variable

    def has_free_variables(self) -> bool:
        return False

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return [self.free_variable]

    def delete_weights(self):
        pass

    def update_state(self, new_cmds: Dict[int, Dict[str, float]], dt: float):
        super().update_state(new_cmds, dt)

    def update_weights(self, weights: Dict[int, Dict[my_string, float]]):
        pass


class PrismaticJoint(OneDofJoint):
    def _joint_transformation(self) -> w.TransMatrix:
        translation_axis = w.Point3(self.axis) * self.position_expression
        parent_T_child = w.TransMatrix.from_xyz_rpy(x=translation_axis[0],
                                                    y=translation_axis[1],
                                                    z=translation_axis[2])
        return parent_T_child

    def update_limits(self, linear_limits, angular_limits):
        self.delete_limits()
        for order, linear_limit in linear_limits.items():
            self.free_variable.set_upper_limit(order, linear_limit[self.name])
            self.free_variable.set_lower_limit(order, -linear_limit[self.name])


class RevoluteJoint(OneDofJoint):
    def _joint_transformation(self) -> w.TransMatrix:
        rotation_axis = w.Vector3(self.axis)
        parent_R_child = w.RotationMatrix.from_axis_angle(rotation_axis, self.position_expression)
        return w.TransMatrix(parent_R_child)
        # self.parent_T_child = w.dot(self.parent_T_child, parent_R_child)

    def update_limits(self, linear_limits, angular_limits):
        self.delete_limits()
        for order, angular_limit in angular_limits.items():
            self.free_variable.set_upper_limit(order, angular_limit[self.name])
            self.free_variable.set_lower_limit(order, -angular_limit[self.name])


class ContinuousJoint(RevoluteJoint):
    pass


class OneDofURDFJoint(OneDofJoint, URDFJoint, ABC):
    def __init__(self, urdf_joint: up.Joint, prefix: str):
        try:
            URDFJoint.__init__(self, urdf_joint, prefix)
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
                             upper_limits=upper_limits)
        soft_lower_limits, soft_upper_limits = self.urdf_soft_limits()
        if 0 in soft_lower_limits:
            self.free_variable.set_lower_limit(Derivatives.position, soft_lower_limits[0])
        if 0 in soft_upper_limits:
            self.free_variable.set_upper_limit(Derivatives.position, soft_upper_limits[0])


class PrismaticURDFJoint(OneDofURDFJoint, PrismaticJoint):
    pass


class RevoluteURDFJoint(OneDofURDFJoint, RevoluteJoint):
    pass


class ContinuousURDFJoint(OneDofURDFJoint, ContinuousJoint):
    pass


class MimicURDFJoint(MimicJoint, OneDofURDFJoint, ABC):
    def __init__(self, urdf_joint: up.Joint, prefix: str):
        try:
            URDFJoint.__init__(self, urdf_joint, prefix)
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
                            axis=self.urdf_joint.axis,
                            lower_limits=lower_limits,
                            upper_limits=upper_limits)


class MimicPrismaticURDFJoint(MimicURDFJoint, PrismaticJoint):
    pass


class MimicRevoluteURDFJoint(MimicURDFJoint, RevoluteJoint):
    pass


class MimicContinuousURDFJoint(MimicURDFJoint, ContinuousJoint):
    pass


class OmniDrive(Joint):
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
        name = PrefixName(name, group_name)
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
        super().__init__(name, parent_link_name, child_link_name, w.TransMatrix())

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
            self.translation_variables.append(self.create_free_variable(name=translation_variable_name,
                                                                        lower_limits=translation_lower_limits,
                                                                        upper_limits=translation_upper_limits))

        for orientation_variable_name in self.orientation_names:
            self.orientation_variables.append(self.create_free_variable(name=orientation_variable_name,
                                                                        lower_limits=rotation_lower_limits,
                                                                        upper_limits=rotation_upper_limits))
        self.yaw = self.orientation_variables[-1]
        self.x_vel = self.create_free_variable(self.x_vel_name,
                                               translation_lower_limits,
                                               translation_upper_limits)
        self.y_vel = self.create_free_variable(self.y_vel_name,
                                               translation_lower_limits,
                                               translation_upper_limits)
        self.yaw_vel = self.create_free_variable(self.rot_vel_name,
                                                 rotation_lower_limits,
                                                 rotation_upper_limits)

    @property
    def position_variable_names(self):
        return [self.x_name, self.y_name, self.yaw_name]

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


class PR2CasterJoint(OneDofURDFJoint, MimicJoint):
    def __init__(self, urdf_joint: up.Joint, prefix: str):
        super().__init__(urdf_joint, prefix)
        self.mimiced_joint_name = 'brumbrum'

    def create_free_variables(self):
        pass

    @staticmethod
    def pointVel2D(pos_x, pos_y, vel_x, vel_y, vel_z):
        new_vel_x = vel_x - pos_y * vel_z
        new_vel_y = vel_y + pos_x * vel_z
        return new_vel_x, new_vel_y

    def _joint_transformation(self):
        try:
            x_vel = self.x_vel.get_symbol(Derivatives.velocity)
            y_vel = self.y_vel.get_symbol(Derivatives.velocity)
            yaw_vel = self.yaw_vel.get_symbol(Derivatives.velocity)
        except:
            x_vel = 0
            y_vel = 0
            yaw_vel = 0

        # caster_link = self.world.joints[self.name].child_link_name
        parent_P_child = self._parent_T_child.to_position()
        new_vel_x, new_vel_y = self.pointVel2D(parent_P_child[0],
                                               parent_P_child[1],
                                               x_vel,
                                               y_vel,
                                               yaw_vel)
        steer_angle_desired = w.if_else(condition=w.logic_and(w.ca.eq(x_vel, 0),
                                                              w.ca.eq(y_vel, 0),
                                                              w.ca.eq(yaw_vel, 0)),
                                        if_result=0,
                                        else_result=np.arctan2(new_vel_y, new_vel_x))

        rotation_axis = w.Vector3(self.axis)
        parent_R_child = w.RotationMatrix.from_axis_angle(rotation_axis, steer_angle_desired)
        return parent_R_child

    def update_state(self, new_cmds: Dict[int, Dict[str, float]], dt: float):
        pass
        # self.world.state[joint_name].position = steer_angle_desired

    def update_limits(self, linear_limits: derivative_joint_map, angular_limits: derivative_joint_map):
        pass

    def update_weights(self, weights: derivative_joint_map):
        pass

    def get_limit_expressions(self, order: int) -> Optional[Tuple[w.Expression, w.Expression]]:
        pass

    def has_free_variables(self) -> bool:
        pass

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return []

    def connect_to_existing_free_variables(self):
        self.brumbrum: OmniDrive = self.world._joints[self.mimiced_joint_name]
        self.x_vel = self.brumbrum.x_vel
        self.y_vel = self.brumbrum.y_vel
        self.yaw_vel = self.brumbrum.yaw_vel


# class OmniDriveWithCaster(Joint):
#     def __init__(self,
#                  parent_link_name: my_string,
#                  child_link_name: my_string,
#                  name: Optional[my_string] = 'brumbrum',
#                  translation_velocity_limit: Optional[float] = 0.5,
#                  rotation_velocity_limit: Optional[float] = 0.6,
#                  translation_acceleration_limit: Optional[float] = None,
#                  rotation_acceleration_limit: Optional[float] = None,
#                  translation_jerk_limit: Optional[float] = 5,
#                  rotation_jerk_limit: Optional[float] = 10,
#                  **kwargs):
#         self.translation_velocity_limit = translation_velocity_limit
#         self.rotation_velocity_limit = rotation_velocity_limit
#         self.translation_acceleration_limit = translation_acceleration_limit
#         self.rotation_acceleration_limit = rotation_acceleration_limit
#         self.translation_jerk_limit = translation_jerk_limit
#         self.rotation_jerk_limit = rotation_jerk_limit
#         self.translation_names = ['odom_x', 'odom_y', 'odom_z']
#         # self.orientation_names = ['odom_qx', 'odom_qy', 'odom_qz', 'odom_qw']
#         self.orientation_names = ['roll', 'pitch', 'yaw']
#         # self.rot_name = 'odom_rot'
#         self.x_vel_name = 'odom_x_vel'
#         self.y_vel_name = 'odom_y_vel'
#         self.rot_vel_name = 'odom_yaw_vel'
#         self.translation_variables: List[FreeVariable] = []
#         self.orientation_variables: List[FreeVariable] = []
#         super().__init__(name, parent_link_name, child_link_name, w.eye(4))
#         self.caster_joints = ['fl_caster_rotation_joint',
#                               'fr_caster_rotation_joint',
#                               'bl_caster_rotation_joint',
#                               'br_caster_rotation_joint']
#
#     def create_free_variables(self):
#         translation_upper_limits = {}
#         if self.translation_velocity_limit is not None:
#             translation_upper_limits[1] = self.translation_velocity_limit
#         if self.translation_acceleration_limit is not None:
#             translation_upper_limits[2] = self.translation_acceleration_limit
#         if self.translation_jerk_limit is not None:
#             translation_upper_limits[3] = self.translation_jerk_limit
#         translation_lower_limits = {k: -v for k, v in translation_upper_limits.items()}
#
#         rotation_upper_limits = {}
#         if self.rotation_velocity_limit is not None:
#             rotation_upper_limits[1] = self.rotation_velocity_limit
#         if self.rotation_acceleration_limit is not None:
#             rotation_upper_limits[2] = self.rotation_acceleration_limit
#         if self.rotation_jerk_limit is not None:
#             rotation_upper_limits[3] = self.rotation_jerk_limit
#         rotation_lower_limits = {k: -v for k, v in rotation_upper_limits.items()}
#
#         for translation_variable_name in self.translation_names:
#             self.translation_variables.append(self.create_free_variable(name=translation_variable_name,
#                                                                         lower_limits=translation_lower_limits,
#                                                                         upper_limits=translation_upper_limits))
#
#         for orientation_variable_name in self.orientation_names:
#             self.orientation_variables.append(self.create_free_variable(name=orientation_variable_name,
#                                                                         lower_limits=rotation_lower_limits,
#                                                                         upper_limits=rotation_upper_limits))
#         self.yaw = self.orientation_variables[-1]
#         self.x_vel = self.create_free_variable(self.x_vel_name,
#                                                translation_lower_limits,
#                                                translation_upper_limits)
#         self.y_vel = self.create_free_variable(self.y_vel_name,
#                                                translation_lower_limits,
#                                                translation_upper_limits)
#         self.yaw_vel = self.create_free_variable(self.rot_vel_name,
#                                                  rotation_lower_limits,
#                                                  rotation_upper_limits)
#
#     def _joint_transformation(self):
#         odom_T_base_footprint = w.frame_from_x_y_rot(self.x.get_symbol(0),
#                                                      self.y.get_symbol(0),
#                                                      self.yaw.get_symbol(0))
#         base_footprint_T_base_footprint_vel = w.frame_from_x_y_rot(self.x_vel.get_symbol(0),
#                                                                    self.y_vel.get_symbol(0),
#                                                                    self.yaw_vel.get_symbol(0))
#         base_footprint_vel_T_base_footprint = w.frame_rpy(x=0,
#                                                           y=0,
#                                                           z=self.translation_variables[2].get_symbol(0),
#                                                           roll=self.orientation_variables[0].get_symbol(0),
#                                                           pitch=self.orientation_variables[1].get_symbol(0),
#                                                           yaw=0)
#         return w.dot(odom_T_base_footprint, base_footprint_T_base_footprint_vel, base_footprint_vel_T_base_footprint)
#
#     @property
#     def x(self):
#         return self.translation_variables[0]
#
#     @property
#     def y(self):
#         return self.translation_variables[1]
#
#     @property
#     def z(self):
#         return self.translation_variables[2]
#
#     @property
#     def x_name(self):
#         return self.translation_names[0]
#
#     @property
#     def y_name(self):
#         return self.translation_names[1]
#
#     @property
#     def z_name(self):
#         return self.translation_names[2]
#
#     @property
#     def roll_name(self):
#         return self.orientation_names[0]
#
#     @property
#     def pitch_name(self):
#         return self.orientation_names[1]
#
#     @property
#     def yaw_name(self):
#         return self.orientation_names[2]
#
#     def pointVel2D(self, pos_x, pos_y, vel_x, vel_y, vel_z):
#         new_vel_x = vel_x - pos_y * vel_z
#         new_vel_y = vel_y + pos_x * vel_z
#         return new_vel_x, new_vel_y
#
#     def computeDesiredCasterSteer(self, joint_name, vel_x, vel_y, vel_z):
#         caster_link = self.world.joints[joint_name].child_link_name
#         base_footprint_T_caster = self.world.compute_fk_pose(self.child_link_name, caster_link)
#         new_vel_x, new_vel_y = self.pointVel2D(base_footprint_T_caster.pose.position.x,
#                                                base_footprint_T_caster.pose.position.y,
#                                                vel_x,
#                                                vel_y,
#                                                vel_z)
#         steer_angle_desired = np.arctan2(new_vel_y, new_vel_x)
#         self.world.state[joint_name].position = steer_angle_desired
#
#     def update_state(self, new_cmds: derivative_joint_map, dt: float):
#         state = self.world.state
#         for free_variable in self.free_variable_list:
#             try:
#                 vel = new_cmds[0][free_variable.position_name]
#             except KeyError as e:
#                 # joint is currently not part of the optimization problem
#                 continue
#             state[free_variable.name].velocity = vel
#             if len(new_cmds) >= 2:
#                 acc = new_cmds[1][free_variable.position_name]
#                 state[free_variable.name].acceleration = acc
#             if len(new_cmds) >= 3:
#                 jerk = new_cmds[2][free_variable.position_name]
#                 state[free_variable.name].jerk = jerk
#         x_vel = state[self.x_vel_name].velocity
#         y_vel = state[self.y_vel_name].velocity
#         rot_vel = state[self.rot_vel_name].velocity
#         delta = state[self.yaw_name].position
#         state[self.x_name].velocity = (np.cos(delta) * x_vel - np.sin(delta) * y_vel)
#         state[self.x_name].position += state[self.x_name].velocity * dt
#         state[self.y_name].velocity = (np.sin(delta) * x_vel + np.cos(delta) * y_vel)
#         state[self.y_name].position += state[self.y_name].velocity * dt
#         state[self.yaw_name].velocity = rot_vel
#         state[self.yaw_name].position += rot_vel * dt
#         for caster_joint in self.caster_joints:
#             self.computeDesiredCasterSteer(caster_joint, x_vel, y_vel, rot_vel)
#
#     def update_limits(self, linear_limits: derivative_joint_map, angular_limits: derivative_joint_map):
#         for free_variable in self._all_symbols():
#             free_variable.lower_limits = {}
#             free_variable.upper_limits = {}
#
#         for order, linear_limit in linear_limits.items():
#             self.x_vel.set_upper_limit(order, linear_limit[self.x_vel_name])
#             self.y_vel.set_upper_limit(order, linear_limit[self.y_vel_name])
#
#             self.x_vel.set_lower_limit(order, -linear_limit[self.x_vel_name])
#             self.y_vel.set_lower_limit(order, -linear_limit[self.y_vel_name])
#
#         for order, angular_limit in angular_limits.items():
#             self.yaw_vel.set_upper_limit(order, angular_limit[self.rot_vel_name])
#             self.yaw_vel.set_lower_limit(order, -angular_limit[self.rot_vel_name])
#
#     def update_weights(self, weights: derivative_joint_map):
#         # self.delete_weights()
#         for order, weight in weights.items():
#             try:
#                 for free_variable in self._all_symbols():
#                     free_variable.quadratic_weights[order] = weight[self.name]
#             except KeyError:
#                 # can't do "if in", because the dict may be a defaultdict
#                 pass
#
#     def get_limit_expressions(self, order: int) -> Optional[Tuple[expr_symbol, expr_symbol]]:
#         pass
#
#     def has_free_variables(self) -> bool:
#         return True
#
#     @property
#     def free_variable_list(self) -> List[FreeVariable]:
#         return [self.x_vel, self.y_vel, self.yaw_vel]
#
#     def _all_symbols(self) -> List[FreeVariable]:
#         return self.free_variable_list + self.translation_variables + self.orientation_variables


# class QuasiOmniDrive(Joint):
#     def __init__(self,
#                  parent_link_name: my_string,
#                  child_link_name: my_string,
#                  name: Optional[my_string] = 'brumbrum',
#                  translation_velocity_limit: Optional[float] = 0.5,
#                  rotation_velocity_limit: Optional[float] = 0.6,
#                  translation_acceleration_limit: Optional[float] = None,
#                  rotation_acceleration_limit: Optional[float] = None,
#                  translation_jerk_limit: Optional[float] = 5,
#                  rotation_jerk_limit: Optional[float] = 10,
#                  **kwargs):
#         self.translation_velocity_limit = translation_velocity_limit
#         self.rotation_velocity_limit = rotation_velocity_limit
#         self.translation_acceleration_limit = translation_acceleration_limit
#         self.rotation_acceleration_limit = rotation_acceleration_limit
#         self.translation_jerk_limit = translation_jerk_limit
#         self.rotation_jerk_limit = rotation_jerk_limit
#         self.translation_names = ['odom_x', 'odom_y', 'odom_z']
#         self.orientation_names = ['roll', 'pitch', 'yaw']
#         self.crosscap_r_name = 'crosscap_r'
#         self.crosscap_alpha_name = 'crosscap_alpha'
#         # self.rot_name = 'odom_rot'
#         self.x_vel_name = 'odom_x_vel'
#         self.yaw1_vel_name = 'odom_yaw1_vel'
#         # self.yaw2_vel_name = 'odom_yaw2_vel'
#         self.translation_variables: List[FreeVariable] = []
#         self.orientation_variables: List[FreeVariable] = []
#         self.crosscap_variables: List[FreeVariable] = []
#         super().__init__(name, parent_link_name, child_link_name, w.eye(4))
#
#     def create_free_variables(self):
#         translation_upper_limits = {}
#         if self.translation_velocity_limit is not None:
#             translation_upper_limits[1] = self.translation_velocity_limit
#         if self.translation_acceleration_limit is not None:
#             translation_upper_limits[2] = self.translation_acceleration_limit
#         if self.translation_jerk_limit is not None:
#             translation_upper_limits[3] = self.translation_jerk_limit
#         translation_lower_limits = {k: -v for k, v in translation_upper_limits.items()}
#
#         rotation_upper_limits = {}
#         if self.rotation_velocity_limit is not None:
#             rotation_upper_limits[1] = self.rotation_velocity_limit
#         if self.rotation_acceleration_limit is not None:
#             rotation_upper_limits[2] = self.rotation_acceleration_limit
#         if self.rotation_jerk_limit is not None:
#             rotation_upper_limits[3] = self.rotation_jerk_limit
#         rotation_lower_limits = {k: -v for k, v in rotation_upper_limits.items()}
#
#         for translation_variable_name in self.translation_names:
#             self.translation_variables.append(self.create_free_variable(name=translation_variable_name,
#                                                                         lower_limits=translation_lower_limits,
#                                                                         upper_limits=translation_upper_limits))
#
#         for orientation_variable_name in self.orientation_names:
#             self.orientation_variables.append(self.create_free_variable(name=orientation_variable_name,
#                                                                         lower_limits=rotation_lower_limits,
#                                                                         upper_limits=rotation_upper_limits))
#         self.yaw = self.orientation_variables[2]
#         self.x_vel = self.create_free_variable(self.x_vel_name,
#                                                translation_lower_limits,
#                                                translation_upper_limits)
#         self.yaw1_vel = self.create_free_variable(self.yaw1_vel_name,
#                                                   translation_lower_limits,
#                                                   translation_upper_limits)
#         self.crosscap_r = self.create_free_variable(name=self.crosscap_r_name,
#                                                     lower_limits={
#                                                         0: -1,
#                                                         1: -10,
#                                                         2: -10,
#                                                         3: -100
#                                                     },
#                                                     upper_limits={
#                                                         0: 1,
#                                                         1: 10,
#                                                         2: 10,
#                                                         3: 100
#                                                     })
#         self.crosscap_alpha = self.create_free_variable(name=self.crosscap_alpha_name,
#                                                         lower_limits={
#                                                             1: -10,
#                                                             2: -10,
#                                                             3: -100
#                                                         },
#                                                         upper_limits={
#                                                             1: 10,
#                                                             2: 10,
#                                                             3: 100
#                                                         })
#
#     def _joint_transformation(self):
#         r = self.crosscap_r.get_symbol(0)
#         alpha = self.crosscap_alpha.get_symbol(0)
#         x = r * w.cos(alpha)
#         y = r * w.sin(alpha)
#         self.god_map.set_data(identifier.hack, 0)
#         hack = self.god_map.to_symbol(identifier.hack)
#         x_vel = self.x_vel.get_symbol(0)
#         # x = w.if_less(w.abs(r), 0.01,
#         #                  if_result=1,
#         #                  else_result=r * w.cos(alpha))
#         # y = w.if_less(w.abs(r), 0.01,
#         #                  if_result=0,
#         #                  else_result=r * w.sin(alpha))
#         # v = w.scale(v, 1)
#         # center_V_p = w.Matrix([x, y, 0, 0])
#         # center_V_x = w.Matrix([1, 0, 0, 0])
#         # center_V_z = w.Matrix([0, 0, 1, 0])
#         # c_V_goal = w.cross(center_V_p, center_V_z)
#         # angle = w.angle_between_vector(center_V_x, c_V_goal)
#         rotation_scale = 1 - w.abs(r)
#         translation_scale = w.abs(r)
#         factor = 0.1
#
#         r_sign = w.sign(r)
#         r_sign = w.if_eq_zero(r_sign, 1, r_sign)
#         r2 = w.if_less(w.abs(r), 0.01, if_result=r + 0.01 * r_sign, else_result=r)
#
#         odom_T_b1 = w.frame_from_x_y_rot(self.x.get_symbol(0),
#                                          self.y.get_symbol(0),
#                                          self.yaw.get_symbol(0))
#         b1_T_v1 = w.frame_from_x_y_rot(factor * r2 * w.cos(alpha) * hack + r * w.cos(alpha) * x_vel,
#                                        factor * r2 * w.sin(alpha) * hack + r * w.sin(alpha) * x_vel,
#                                        self.yaw1_vel.get_symbol(0))
#         # v1_T_b2 = w.frame_from_x_y_rot(0,
#         #                                0,
#         #                                self.yaw2.get_symbol(0))
#         # b2_T_v2 = w.frame_from_x_y_rot(0,
#         #                                0,
#         #                                self.yaw2_vel.get_symbol(0))
#         v2_T_base_footprint = w.frame_rpy(x=0,
#                                           y=0,
#                                           z=self.translation_variables[2].get_symbol(0),
#                                           roll=self.orientation_variables[0].get_symbol(0),
#                                           pitch=self.orientation_variables[1].get_symbol(0),
#                                           yaw=0)
#         return w.dot(odom_T_b1, b1_T_v1, v2_T_base_footprint)
#
#     @property
#     def x(self):
#         return self.translation_variables[0]
#
#     @property
#     def y(self):
#         return self.translation_variables[1]
#
#     @property
#     def z(self):
#         return self.translation_variables[2]
#
#     @property
#     def x_name(self):
#         return self.translation_names[0]
#
#     @property
#     def y_name(self):
#         return self.translation_names[1]
#
#     @property
#     def z_name(self):
#         return self.translation_names[2]
#
#     @property
#     def roll_name(self):
#         return self.orientation_names[0]
#
#     @property
#     def pitch_name(self):
#         return self.orientation_names[1]
#
#     @property
#     def yaw1_name(self):
#         return self.orientation_names[2]
#
#     def update_state(self, new_cmds: derivative_joint_map, dt: float):
#         state = self.world.state
#         for free_variable in self.free_variable_list:
#             try:
#                 vel = new_cmds[0][free_variable.position_name]
#             except KeyError as e:
#                 # joint is currently not part of the optimization problem
#                 continue
#             state[free_variable.name].velocity = vel
#             if len(new_cmds) >= 2:
#                 acc = new_cmds[1][free_variable.position_name]
#                 state[free_variable.name].acceleration = acc
#             if len(new_cmds) >= 3:
#                 jerk = new_cmds[2][free_variable.position_name]
#                 state[free_variable.name].jerk = jerk
#         x_vel = state[self.x_vel_name].velocity
#         yaw1_vel = state[self.yaw1_vel_name].velocity
#         # yaw2_vel = state[self.yaw2_vel_name].velocity
#         yaw = state[self.yaw1_name].position
#
#         r = state[self.crosscap_r_name].position
#         alpha = state[self.crosscap_alpha_name].position
#         x = r * w.cos(alpha)
#         y = r * w.sin(alpha)
#         v = np.array([x, y, 0])
#         rotation_scale = 1 - w.abs(r)
#         # v_norm = np.linalg.norm(v)
#         # if v_norm > 0.001:
#         #     v /= v_norm
#         # else:
#         #     v = np.array([0,0,0])
#         v *= x_vel
#
#         state[self.crosscap_r_name].position += state[self.crosscap_r_name].velocity * dt
#         state[self.crosscap_alpha_name].position += state[self.crosscap_alpha_name].velocity * dt
#
#         state[self.x_name].velocity = (np.cos(yaw) * v[0] - np.sin(yaw) * v[1])
#         state[self.x_name].position += state[self.x_name].velocity * dt
#         state[self.y_name].velocity = (np.sin(yaw) * v[0] + np.cos(yaw) * v[1])
#         state[self.y_name].position += state[self.y_name].velocity * dt
#         state[self.yaw1_name].velocity = rotation_scale * yaw1_vel
#         state[self.yaw1_name].position += state[self.yaw1_name].velocity * dt
#         # state[self.yaw2_name].position += yaw2_vel * dt
#
#     def update_limits(self, linear_limits: derivative_joint_map, angular_limits: derivative_joint_map):
#         for free_variable in self._all_symbols():
#             free_variable.lower_limits = {}
#             free_variable.upper_limits = {}
#
#         for order, linear_limit in linear_limits.items():
#             self.x_vel.set_upper_limit(order, linear_limit[self.x_vel_name])
#             # self.y_vel.set_upper_limit(order, linear_limit[self.y_vel_name])
#
#             self.x_vel.set_lower_limit(order, -linear_limit[self.x_vel_name])
#             # self.y_vel.set_lower_limit(order, -linear_limit[self.y_vel_name])
#
#         for order, angular_limit in angular_limits.items():
#             self.yaw1_vel.set_upper_limit(order, angular_limit[self.yaw1_vel_name])
#             self.yaw1_vel.set_lower_limit(order, -angular_limit[self.yaw1_vel_name])
#             # self.yaw2_vel.set_upper_limit(order, angular_limit[self.yaw2_vel_name])
#             # self.yaw2_vel.set_lower_limit(order, -angular_limit[self.yaw2_vel_name])
#
#     def update_weights(self, weights: derivative_joint_map):
#         # self.delete_weights()
#         for order, weight in weights.items():
#             try:
#                 for free_variable in self._all_symbols():
#                     free_variable.quadratic_weights[order] = weight[self.name]
#             except KeyError:
#                 # can't do "if in", because the dict may be a defaultdict
#                 pass
#
#     def get_limit_expressions(self, order: int) -> Optional[Tuple[expr_symbol, expr_symbol]]:
#         pass
#
#     def has_free_variables(self) -> bool:
#         return True
#
#     @property
#     def free_variable_list(self) -> List[FreeVariable]:
#         return [self.x_vel, self.yaw1_vel, self.crosscap_r, self.crosscap_alpha]
#
#     def _all_symbols(self) -> List[FreeVariable]:
#         return self.free_variable_list + self.translation_variables + self.orientation_variables


class DiffDrive(Joint):
    x: FreeVariable
    y: FreeVariable
    yaw: FreeVariable
    x_vel: FreeVariable
    yaw_vel: FreeVariable

    def __init__(self,
                 parent_link_name: my_string,
                 child_link_name: my_string,
                 group_name: Optional[str] = None,
                 name: Optional[my_string] = 'brumbrum',
                 translation_velocity_limit: Optional[float] = 0.5,
                 rotation_velocity_limit: Optional[float] = 0.6,
                 translation_acceleration_limit: Optional[float] = None,
                 rotation_acceleration_limit: Optional[float] = None,
                 translation_jerk_limit: Optional[float] = 5,
                 rotation_jerk_limit: Optional[float] = 10,
                 x_name: Optional[str] = 'odom_x',
                 y_name: Optional[str] = 'odom_y',
                 yaw_name: Optional[str] = 'odom_yaw',
                 x_vel_name: Optional[str] = 'base_footprint_x_vel',
                 rot_vel_name: Optional[str] = 'base_footprint_rot_vel',
                 **kwargs):
        name = PrefixName(name, group_name)
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.translation_acceleration_limit = translation_acceleration_limit
        self.rotation_acceleration_limit = rotation_acceleration_limit
        self.translation_jerk_limit = translation_jerk_limit
        self.rotation_jerk_limit = rotation_jerk_limit
        self.x_name = PrefixName(x_name, group_name)
        self.y_name = PrefixName(y_name, group_name)
        self.yaw_name = PrefixName(yaw_name, group_name)
        self.x_vel_name = PrefixName(x_vel_name, group_name)
        self.rot_vel_name = PrefixName(rot_vel_name, group_name)
        super().__init__(name, parent_link_name, child_link_name, w.TransMatrix())

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

        self.x = self.create_free_variable(self.x_name,
                                           translation_lower_limits,
                                           translation_upper_limits)
        self.y = self.create_free_variable(self.y_name,
                                           translation_lower_limits,
                                           translation_upper_limits)
        self.yaw = self.create_free_variable(self.yaw_name,
                                             rotation_lower_limits,
                                             rotation_upper_limits)
        self.x_vel = self.create_free_variable(self.x_vel_name,
                                               translation_lower_limits,
                                               translation_upper_limits)
        self.yaw_vel = self.create_free_variable(self.rot_vel_name,
                                                 rotation_lower_limits,
                                                 rotation_upper_limits)

    def _joint_transformation(self):
        odom_T_bf = w.TransMatrix.from_xyz_rpy(x=self.x.get_symbol(Derivatives.position),
                                               y=self.y.get_symbol(Derivatives.position),
                                               yaw=self.yaw.get_symbol(Derivatives.position))
        bf_T_bf_vel = w.TransMatrix.from_xyz_rpy(x=self.x_vel.get_symbol(Derivatives.position),
                                                 y=0,
                                                 yaw=self.yaw_vel.get_symbol(Derivatives.position))
        return w.dot(odom_T_bf, bf_T_bf_vel)

    def update_state(self, new_cmds: derivative_joint_map, dt: float):
        world = self.god_map.unsafe_get_data(identifier.world)
        for free_variable in self.free_variable_list:
            try:
                vel = new_cmds[Derivatives.velocity][free_variable.position_name]
            except KeyError as e:
                # joint is currently not part of the optimization problem
                continue
            world.state[free_variable.name].velocity = vel
            if len(new_cmds) >= 2:
                acc = new_cmds[Derivatives.acceleration][free_variable.position_name]
                world.state[free_variable.name].acceleration = acc
            if len(new_cmds) >= 3:
                jerk = new_cmds[Derivatives.jerk][free_variable.position_name]
                world.state[free_variable.name].jerk = jerk
        x = world.state[self.x_vel_name].velocity
        rot = world.state[self.rot_vel_name].velocity
        delta = world.state[self.yaw_name].position
        world.state[self.x_name].velocity = (np.cos(delta) * x)
        world.state[self.x_name].position += world.state[self.x_name].velocity * dt
        world.state[self.y_name].velocity = (np.sin(delta) * x)
        world.state[self.y_name].position += world.state[self.y_name].velocity * dt
        world.state[self.yaw_name].velocity = rot
        world.state[self.yaw_name].position += rot * dt

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
            self.yaw_vel.set_upper_limit(order, angular_limit[self.rot_vel_name])
            self.yaw_vel.set_lower_limit(order, -angular_limit[self.rot_vel_name])

    def update_weights(self, weights: derivative_joint_map):
        # self.delete_weights()
        for order, weight in weights.items():
            try:
                for free_variable in self._all_symbols():
                    free_variable.quadratic_weights[order] = weight[self.name]
            except KeyError:
                # can't do if in, because the dict may be a defaultdict
                pass

    def get_limit_expressions(self, order: int) -> Optional[Tuple[Union[w.Symbol, float], Union[w.Symbol, float]]]:
        pass

    def has_free_variables(self) -> bool:
        return True

    @property
    def free_variable_list(self) -> List[FreeVariable]:
        return [self.x_vel, self.yaw_vel]

    def _all_symbols(self) -> List[FreeVariable]:
        return self.free_variable_list + [self.x, self.y, self.yaw]
