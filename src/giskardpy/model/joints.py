import numpy as np

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.data_types import PrefixName
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.utils.utils import limits_from_urdf_joint


class Joint(object):
    def __init__(self, name, parent_link_name, child_link_name, parent_T_child=None,
                 translation_offset=None,
                 rotation_offset=None):
        assert isinstance(name, PrefixName)
        self.name = name  # type: PrefixName
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name

        if translation_offset is None:
            translation_offset = [0, 0, 0]
        if rotation_offset is None:
            rotation_offset = [0, 0, 0]
        if parent_T_child is None:
            self.parent_T_child = w.dot(w.translation3(*translation_offset),
                                        w.rotation_matrix_from_rpy(*rotation_offset))
        else:
            self.parent_T_child = parent_T_child

    def __repr__(self):
        return str(self.name)

    def has_free_variables(self):
        return False

    @classmethod
    def from_urdf(cls, urdf_joint, prefix, parent_link_name, child_link_name, god_map):
        joint_name = PrefixName(urdf_joint.name, prefix)
        if urdf_joint.origin is not None:
            translation_offset = urdf_joint.origin.xyz
            rotation_offset = urdf_joint.origin.rpy
        else:
            translation_offset = None
            rotation_offset = None

        if urdf_joint.mimic is not None:
            if urdf_joint.type == 'prismatic':
                joint = MimicedPrismaticJoint(name=joint_name,
                                              parent_link_name=parent_link_name,
                                              child_link_name=child_link_name,
                                              translation_offset=translation_offset,
                                              rotation_offset=rotation_offset,
                                              god_map=god_map,
                                              axis=urdf_joint.axis,
                                              multiplier=urdf_joint.mimic.multiplier,
                                              offset=urdf_joint.mimic.offset,
                                              mimed_joint_name=PrefixName(urdf_joint.mimic.joint, prefix))
            elif urdf_joint.type == 'revolute':
                joint = MimicedRevoluteJoint(name=joint_name,
                                             parent_link_name=parent_link_name,
                                             child_link_name=child_link_name,
                                             translation_offset=translation_offset,
                                             rotation_offset=rotation_offset,
                                             god_map=god_map,
                                             axis=urdf_joint.axis,
                                             multiplier=urdf_joint.mimic.multiplier,
                                             offset=urdf_joint.mimic.offset,
                                             mimed_joint_name=PrefixName(urdf_joint.mimic.joint, prefix))
            elif urdf_joint.type == 'continuous':
                joint = MimicedContinuousJoint(name=joint_name,
                                               parent_link_name=parent_link_name,
                                               child_link_name=child_link_name,
                                               translation_offset=translation_offset,
                                               rotation_offset=rotation_offset,
                                               god_map=god_map,
                                               axis=urdf_joint.axis,
                                               multiplier=urdf_joint.mimic.multiplier,
                                               offset=urdf_joint.mimic.offset,
                                               mimed_joint_name=PrefixName(urdf_joint.mimic.joint, prefix))
            else:
                raise NotImplementedError('Joint type \'{}\' of \'{}\' is not implemented.'.format(urdf_joint.name,
                                                                                                   urdf_joint.type))
        else:
            if urdf_joint.type == 'fixed':
                joint = FixedJoint(name=joint_name,
                                   parent_link_name=parent_link_name,
                                   child_link_name=child_link_name,
                                   translation_offset=translation_offset,
                                   rotation_offset=rotation_offset)
            elif urdf_joint.type == 'revolute':
                joint = RevoluteJoint(name=joint_name,
                                      parent_link_name=parent_link_name,
                                      child_link_name=child_link_name,
                                      translation_offset=translation_offset,
                                      rotation_offset=rotation_offset,
                                      god_map=god_map,
                                      axis=urdf_joint.axis)
            elif urdf_joint.type == 'prismatic':
                joint = PrismaticJoint(name=joint_name,
                                       parent_link_name=parent_link_name,
                                       child_link_name=child_link_name,
                                       translation_offset=translation_offset,
                                       rotation_offset=rotation_offset,
                                       god_map=god_map,
                                       axis=urdf_joint.axis)
            elif urdf_joint.type == 'continuous':
                joint = ContinuousJoint(name=joint_name,
                                        parent_link_name=parent_link_name,
                                        child_link_name=child_link_name,
                                        translation_offset=translation_offset,
                                        rotation_offset=rotation_offset,
                                        god_map=god_map,
                                        axis=urdf_joint.axis)
            else:
                raise NotImplementedError('Joint type \'{}\' of \'{}\' is not implemented.'.format(urdf_joint.name,
                                                                                                   urdf_joint.type))

        if isinstance(joint, OneDofJoint):
            if not isinstance(joint, MimicJoint):
                lower_limits, upper_limits = limits_from_urdf_joint(urdf_joint)
                joint.create_free_variables(where_am_i=identifier.joint_states,
                                            lower_limits=lower_limits,
                                            upper_limits=upper_limits)
        return joint


class FixedJoint(Joint):
    pass


class MovableJoint(Joint):
    def __init__(self, name, parent_link_name, child_link_name, god_map, parent_T_child=None,
                 translation_offset=None, rotation_offset=None):
        """
        :type name: str
        :type parent_link_name: str
        :type child_link_name: str
        :type god_map: giskardpy.god_map.GodMap
        :type parent_T_child:
        :type translation_offset:
        :type rotation_offset:
        """
        super(MovableJoint, self).__init__(name=name,
                                           parent_link_name=parent_link_name,
                                           child_link_name=child_link_name,
                                           parent_T_child=parent_T_child,
                                           translation_offset=translation_offset,
                                           rotation_offset=rotation_offset)
        self.god_map = god_map
        self._world = self.god_map.unsafe_get_data(identifier.world)

    @property
    def world(self):
        """
        :rtype: giskardpy.model.world.WorldTree
        """
        return self._world

    @property
    def free_variables(self):
        return self._free_variables

    @property
    def free_variable_list(self):
        return list(self._free_variables.values())

    def create_free_variables(self, **kwargs):
        self._free_variables = kwargs
        self.update_parent_T_child()

    def has_free_variables(self):
        return len(self.free_variables) > 0

    def update_state(self, new_cmds, dt):
        pass

    def update_parent_T_child(self):
        pass

    def update_limits(self, linear_limits, angular_limits):
        raise NotImplementedError()

    def update_weights(self, weights):
        raise NotImplementedError()

    def get_limit_expressions(self, order):
        raise NotImplementedError()


class OneDofJoint(MovableJoint):
    @property
    def free_variable(self):
        """
        :rtype: FreeVariable
        """
        return self._free_variables['position']

    @property
    def position_symbol(self):
        return self._free_variables['position'].get_symbol(0)

    @property
    def position_limits(self):
        return self.get_limit_expressions(0)

    @property
    def velocity_limit(self):
        return self.get_limit_expressions(1)[1]

    def get_limit_expressions(self, order):
        return self.free_variable.get_lower_limit(order), self.free_variable.get_upper_limit(order)

    def create_free_variables(self, where_am_i, lower_limits, upper_limits, horizon_function=None):
        free_variable = FreeVariable(
            symbols={
                0: self.god_map.to_symbol(where_am_i + [self.name, 'position']),
                1: self.god_map.to_symbol(where_am_i + [self.name, 'velocity']),
                2: self.god_map.to_symbol(where_am_i + [self.name, 'acceleration']),
                3: self.god_map.to_symbol(where_am_i + [self.name, 'jerk']),
            },
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            quadratic_weights={},
            horizon_functions={1: 0.1})
        super(OneDofJoint, self).create_free_variables(position=free_variable)

    def update_state(self, new_cmds, dt):
        try:
            vel = new_cmds[0][self.free_variable.name]
        except KeyError as e:
            # joint is currently not active
            return
        self.world.state[self.name].position += vel * dt
        self.world.state[self.name].velocity = vel
        if len(new_cmds) >= 2:
            acc = new_cmds[1][self.free_variable.name]
            self.world.state[self.name].acceleration = acc
        if len(new_cmds) >= 3:
            jerk = new_cmds[2][self.free_variable.name]
            self.world.state[self.name].jerk = jerk

    def delete_limits(self):
        self.free_variable.lower_limits = {}
        self.free_variable.upper_limits = {}

    def delete_weights(self):
        self.free_variable.quadratic_weights = {}

    def update_weights(self, weights):
        self.delete_weights()
        for order, weight in weights.items():
            self.free_variable.quadratic_weights[order] = weight[self.name]


class RevoluteJoint(OneDofJoint):
    def __init__(self, name, parent_link_name, child_link_name, axis, god_map, parent_T_child=None,
                 translation_offset=None,
                 rotation_offset=None):
        super(RevoluteJoint, self).__init__(name=name,
                                            parent_link_name=parent_link_name,
                                            child_link_name=child_link_name,
                                            god_map=god_map,
                                            parent_T_child=parent_T_child,
                                            translation_offset=translation_offset,
                                            rotation_offset=rotation_offset)
        self.axis = np.array(axis)

    def update_parent_T_child(self):
        expr = self.position_symbol
        self.parent_T_child = w.dot(self.parent_T_child,
                                    w.rotation_matrix_from_axis_angle(w.vector3(*self.axis), expr))

    def update_limits(self, linear_limits, angular_limits):
        self.delete_limits()
        for order, angular_limit in angular_limits.items():
            self.free_variable.set_upper_limit(order, angular_limit[self.name])
            self.free_variable.set_lower_limit(order, -angular_limit[self.name])


class ContinuousJoint(OneDofJoint):
    def __init__(self, name, parent_link_name, child_link_name, axis, god_map, parent_T_child=None,
                 translation_offset=None,
                 rotation_offset=None):
        super(ContinuousJoint, self).__init__(name=name,
                                              parent_link_name=parent_link_name,
                                              child_link_name=child_link_name,
                                              god_map=god_map,
                                              parent_T_child=parent_T_child,
                                              translation_offset=translation_offset,
                                              rotation_offset=rotation_offset)
        self.axis = np.array(axis)

    def update_parent_T_child(self):
        expr = self.position_symbol
        self.parent_T_child = w.dot(self.parent_T_child,
                                    w.rotation_matrix_from_axis_angle(w.vector3(*self.axis), expr))

    def update_limits(self, linear_limits, angular_limits):
        self.delete_limits()
        for order, angular_limit in angular_limits.items():
            self.free_variable.set_upper_limit(order, angular_limit[self.name])
            self.free_variable.set_lower_limit(order, -angular_limit[self.name])


class PrismaticJoint(OneDofJoint):
    def __init__(self, name, parent_link_name, child_link_name, axis, god_map, parent_T_child=None,
                 translation_offset=None,
                 rotation_offset=None):
        super(PrismaticJoint, self).__init__(name=name,
                                             parent_link_name=parent_link_name,
                                             child_link_name=child_link_name,
                                             god_map=god_map,
                                             parent_T_child=parent_T_child,
                                             translation_offset=translation_offset,
                                             rotation_offset=rotation_offset)
        self.axis = np.array(axis)

    def update_parent_T_child(self):
        expr = self.position_symbol
        translation_axis = (w.point3(*self.axis) * expr)
        self.parent_T_child = w.dot(self.parent_T_child, w.translation3(translation_axis[0],
                                                                        translation_axis[1],
                                                                        translation_axis[2]))

    def update_limits(self, linear_limits, angular_limits):
        self.delete_limits()
        for order, linear_limit in linear_limits.items():
            self.free_variable.set_upper_limit(order, linear_limit[self.name])
            self.free_variable.set_lower_limit(order, -linear_limit[self.name])


class MimicJoint(OneDofJoint):
    def __init__(self, name, parent_link_name, child_link_name, god_map, mimed_joint_name, multiplier, offset,
                 parent_T_child=None, translation_offset=None, rotation_offset=None):
        OneDofJoint.__init__(self,
                             name=name,
                             parent_link_name=parent_link_name,
                             child_link_name=child_link_name,
                             god_map=god_map,
                             parent_T_child=parent_T_child,
                             translation_offset=translation_offset,
                             rotation_offset=rotation_offset)
        self.mimed_joint_name = mimed_joint_name
        self.multiplier = multiplier
        self.offset = offset

    def _apply_mimic(self, expr):
        multiplier = 1 if self.multiplier is None else self.multiplier
        offset = 0 if self.offset is None else self.offset
        return expr * multiplier + offset

    @property
    def position_symbol(self):
        mimed_free_variable = self._free_variables[0].get_symbol(0)
        return self._apply_mimic(mimed_free_variable)

    @property
    def position_limits(self):
        lower_limit = self._apply_mimic(self.free_variable.get_lower_limit(0))
        upper_limit = self._apply_mimic(self.free_variable.get_upper_limit(0))
        return lower_limit, upper_limit

    @property
    def velocity_limit(self):
        return self._apply_mimic(self.free_variable.get_upper_limit(1))

    @property
    def free_variables(self):
        return []

    def delete_limits(self):
        """
        This will get deleted over references to the mimed joint.
        """
        pass

    def delete_weights(self):
        """
        This will get deleted over references to the mimed joint.
        """
        pass

    def update_limits(self, linear_limits, angular_limits, order):
        pass


class MimicedPrismaticJoint(MimicJoint, PrismaticJoint):
    def __init__(self, name, parent_link_name, child_link_name, god_map, axis, mimed_joint_name, multiplier, offset,
                 parent_T_child=None, translation_offset=None, rotation_offset=None):
        PrismaticJoint.__init__(self,
                                name=name,
                                parent_link_name=parent_link_name,
                                child_link_name=child_link_name,
                                axis=axis,
                                god_map=god_map,
                                parent_T_child=parent_T_child,
                                translation_offset=translation_offset,
                                rotation_offset=rotation_offset)
        MimicJoint.__init__(self,
                            name=name,
                            parent_link_name=parent_link_name,
                            child_link_name=child_link_name,
                            god_map=god_map,
                            mimed_joint_name=mimed_joint_name,
                            multiplier=multiplier,
                            offset=offset,
                            parent_T_child=parent_T_child,
                            translation_offset=translation_offset,
                            rotation_offset=rotation_offset)

    def update_parent_T_child(self):
        PrismaticJoint.update_parent_T_child(self)


class MimicedRevoluteJoint(MimicJoint, RevoluteJoint):
    def __init__(self, name, parent_link_name, child_link_name, god_map, axis, mimed_joint_name, multiplier, offset,
                 parent_T_child=None, translation_offset=None, rotation_offset=None):
        RevoluteJoint.__init__(self,
                               name=name,
                               parent_link_name=parent_link_name,
                               child_link_name=child_link_name,
                               axis=axis,
                               god_map=god_map,
                               parent_T_child=parent_T_child,
                               translation_offset=translation_offset,
                               rotation_offset=rotation_offset)
        MimicJoint.__init__(self,
                            name=name,
                            parent_link_name=parent_link_name,
                            child_link_name=child_link_name,
                            god_map=god_map,
                            mimed_joint_name=mimed_joint_name,
                            multiplier=multiplier,
                            offset=offset,
                            parent_T_child=parent_T_child,
                            translation_offset=translation_offset,
                            rotation_offset=rotation_offset)

    def update_parent_T_child(self):
        RevoluteJoint.update_parent_T_child(self)


class MimicedContinuousJoint(MimicJoint, ContinuousJoint):
    def __init__(self, name, parent_link_name, child_link_name, god_map, axis, mimed_joint_name, multiplier, offset,
                 parent_T_child=None, translation_offset=None, rotation_offset=None):
        ContinuousJoint.__init__(self,
                                 name=name,
                                 parent_link_name=parent_link_name,
                                 child_link_name=child_link_name,
                                 axis=axis,
                                 god_map=god_map,
                                 parent_T_child=parent_T_child,
                                 translation_offset=translation_offset,
                                 rotation_offset=rotation_offset)
        MimicJoint.__init__(self,
                            name=name,
                            parent_link_name=parent_link_name,
                            child_link_name=child_link_name,
                            god_map=god_map,
                            mimed_joint_name=mimed_joint_name,
                            multiplier=multiplier,
                            offset=offset,
                            parent_T_child=parent_T_child,
                            translation_offset=translation_offset,
                            rotation_offset=rotation_offset)

    def update_parent_T_child(self):
        ContinuousJoint.update_parent_T_child(self)


class DiffDriveJoint(MovableJoint):
    trans_s = 'trans'
    rot_s = 'rot'

    def __init__(self, name, parent_link_name, child_link_name, god_map, translation_axis=None,
                 rotation_axis=None, parent_T_child=None):
        super().__init__(name, parent_link_name, child_link_name, god_map, parent_T_child,
                         translation_offset=[0, 0, 0],
                         rotation_offset=[0, 0, 0])
        if translation_axis is None:
            translation_axis = [1, 0, 0]
        self.translation_axis = translation_axis
        if rotation_axis is None:
            rotation_axis = [0, 0, 1]
        self.rotation_axis = rotation_axis

    def create_free_variables(self, where_am_i, trans_lower_limits, trans_upper_limits, rot_lower_limits,
                              rot_upper_limits, horizon_function=None):
        names = ['{}/x'.format(self.name),
                 '{}/y'.format(self.name),
                 '{}/z'.format(self.name)]
        variables = []
        for name in names:
            variables.append(FreeVariable(
                symbols={
                    0: self.god_map.to_symbol(where_am_i + [name, 'position']),
                    1: self.god_map.to_symbol(where_am_i + [name, 'velocity']),
                    2: self.god_map.to_symbol(where_am_i + [name, 'acceleration']),
                    3: self.god_map.to_symbol(where_am_i + [name, 'jerk']),
                },
                lower_limits={},
                upper_limits={},
                quadratic_weights={},
                horizon_functions={1: 0.1}))
        x, y, z = variables
        trans = FreeVariable(
            symbols={
                0: self.god_map.to_symbol(where_am_i + [self.trans_name, 'position']),
                1: self.god_map.to_symbol(where_am_i + [self.trans_name, 'velocity']),
                2: self.god_map.to_symbol(where_am_i + [self.trans_name, 'acceleration']),
                3: self.god_map.to_symbol(where_am_i + [self.trans_name, 'jerk']),
            },
            lower_limits=trans_lower_limits,
            upper_limits=trans_upper_limits,
            quadratic_weights={},
            horizon_functions={1: 0.1})
        rot = FreeVariable(
            symbols={
                0: self.god_map.to_symbol(where_am_i + [self.rot_name, 'position']),
                1: self.god_map.to_symbol(where_am_i + [self.rot_name, 'velocity']),
                2: self.god_map.to_symbol(where_am_i + [self.rot_name, 'acceleration']),
                3: self.god_map.to_symbol(where_am_i + [self.rot_name, 'jerk']),
            },
            lower_limits=rot_lower_limits,
            upper_limits=rot_upper_limits,
            quadratic_weights={},
            horizon_functions={1: 0.1})
        super().create_free_variables(x=x, y=y, z=z, trans=trans, rot=rot)

    @property
    def x(self):
        return self._free_variables['x'].get_symbol(0)

    def state_name(self, thing):
        return '{}/{}'.format(self.name, thing)

    @property
    def y(self):
        return self._free_variables['y'].get_symbol(0)

    @property
    def z(self):
        return self._free_variables['z'].get_symbol(0)

    @property
    def trans(self):
        return self._free_variables[self.trans_s].get_symbol(0)

    @property
    def rot(self):
        return self._free_variables[self.rot_s].get_symbol(0)

    @property
    def trans_name(self):
        return self.state_name(self.trans_s)

    @property
    def trans_name_long(self):
        return self._free_variables[self.trans_s].name

    @property
    def rot_name(self):
        return self.state_name(self.rot_s)

    @property
    def rot_name_long(self):
        return self._free_variables[self.rot_s].name

    @property
    def free_variable_list(self):
        return [self._free_variables[self.trans_s], self._free_variables[self.rot_s]]

    def update_parent_T_child(self):
        odom_T_x = w.translation3(self.x, 0, 0)
        x_T_y = w.translation3(0, self.y, 0)
        y_T_z = w.rotation_matrix_from_axis_angle(w.vector3(0, 0, 1), self.z)
        z_T_rot = w.rotation_matrix_from_axis_angle(w.vector3(*self.rotation_axis), self.rot)
        translation_axis = w.point3(*self.translation_axis) * (self.trans)
        rot_T_base = w.translation3(translation_axis[0],
                                    translation_axis[1],
                                    translation_axis[2])
        self.parent_T_child = w.dot(self.parent_T_child, odom_T_x, x_T_y, y_T_z, z_T_rot, rot_T_base)

    def update_state(self, new_cmds, dt):
        try:
            trans_vel = new_cmds[0][self.trans_name_long]
            rot_vel = new_cmds[0][self.rot_name_long]
        except KeyError as e:
            # joint is currently not active
            return
        self.world.state[self.trans_name].position = 0
        self.world.state[self.trans_name].velocity = trans_vel
        self.world.state[self.trans_name].acceleration = new_cmds[1][self.trans_name_long]
        self.world.state[self.trans_name].jerk = new_cmds[2][self.trans_name_long]

        self.world.state[self.rot_name].position = 0
        self.world.state[self.rot_name].velocity = rot_vel
        self.world.state[self.rot_name].acceleration = new_cmds[1][self.rot_name_long]
        self.world.state[self.rot_name].jerk = new_cmds[2][self.rot_name_long]

        delta = self.world.state[self.state_name('z')].position
        self.world.state[self.state_name('x')].position += np.cos(delta) * trans_vel * dt
        self.world.state[self.state_name('y')].position += np.sin(delta) * trans_vel * dt
        self.world.state[self.state_name('z')].position += rot_vel * dt
        pass

    def delete_limits(self):
        self.trans.lower_limits = {}
        self.trans.upper_limits = {}
        self.rot.lower_limits = {}
        self.rot.upper_limits = {}

    def delete_weights(self):
        self.trans.quadratic_weights = {}
        self.rot.quadratic_weights = {}

    def update_limits(self, linear_limits, angular_limits):
        self.delete_limits()
        for order, linear_limit in linear_limits.items():
            angular_limit = angular_limits[order]
            self._free_variables[self.trans_s].set_upper_limit(order, linear_limit[self.trans_name])
            self._free_variables[self.trans_s].set_lower_limit(order, -linear_limit[self.trans_name])
            self._free_variables[self.rot_s].set_upper_limit(order, angular_limit[self.rot_name])
            self._free_variables[self.rot_s].set_lower_limit(order, -angular_limit[self.rot_name])

    def update_weights(self, weights):
        self.delete_weights()
        for order, weight in weights.items():
            self._free_variables[self.trans_s].quadratic_weights[order] = weight[self.trans_name]
            self._free_variables[self.rot_s].quadratic_weights[order] = weight[self.rot_name]

    def get_limit_expressions(self, order):
        pass


class DiffDriveWheelsJoint(MovableJoint):
    l_wheel_s = 'l_wheel'
    r_wheel_s = 'r_wheel'
    wheel_dist = 0.404
    wheel_radius = 0.198

    def __init__(self, name, parent_link_name, child_link_name, god_map, translation_axis=None,
                 rotation_axis=None, parent_T_child=None):
        super().__init__(name, parent_link_name, child_link_name, god_map, parent_T_child,
                         translation_offset=[0, 0, 0],
                         rotation_offset=[0, 0, 0])
        if translation_axis is None:
            translation_axis = [1, 0, 0]
        self.translation_axis = translation_axis
        if rotation_axis is None:
            rotation_axis = [0, 0, 1]
        self.rotation_axis = rotation_axis

    def create_free_variables(self, where_am_i, trans_lower_limits, trans_upper_limits, rot_lower_limits,
                              rot_upper_limits, horizon_function=None):
        names = ['{}/x'.format(self.name),
                 '{}/y'.format(self.name),
                 '{}/z'.format(self.name)]
        variables = []
        for name in names:
            variables.append(FreeVariable(
                symbols={
                    0: self.god_map.to_symbol(where_am_i + [name, 'position']),
                    1: self.god_map.to_symbol(where_am_i + [name, 'velocity']),
                    2: self.god_map.to_symbol(where_am_i + [name, 'acceleration']),
                    3: self.god_map.to_symbol(where_am_i + [name, 'jerk']),
                },
                lower_limits={},
                upper_limits={},
                quadratic_weights={},
                horizon_functions={1: 0.1}))
        x, y, z = variables
        trans = FreeVariable(
            symbols={
                0: self.god_map.to_symbol(where_am_i + [self.l_wheel_name, 'position']),
                1: self.god_map.to_symbol(where_am_i + [self.l_wheel_name, 'velocity']),
                2: self.god_map.to_symbol(where_am_i + [self.l_wheel_name, 'acceleration']),
                3: self.god_map.to_symbol(where_am_i + [self.l_wheel_name, 'jerk']),
            },
            lower_limits=trans_lower_limits,
            upper_limits=trans_upper_limits,
            quadratic_weights={},
            horizon_functions={1: 0.1})
        rot = FreeVariable(
            symbols={
                0: self.god_map.to_symbol(where_am_i + [self.r_wheel_name, 'position']),
                1: self.god_map.to_symbol(where_am_i + [self.r_wheel_name, 'velocity']),
                2: self.god_map.to_symbol(where_am_i + [self.r_wheel_name, 'acceleration']),
                3: self.god_map.to_symbol(where_am_i + [self.r_wheel_name, 'jerk']),
            },
            lower_limits=rot_lower_limits,
            upper_limits=rot_upper_limits,
            quadratic_weights={},
            horizon_functions={1: 0.1})
        super().create_free_variables(x=x, y=y, z=z, l_wheel=trans, r_wheel=rot)

    @property
    def x(self):
        return self._free_variables['x'].get_symbol(0)

    def state_name(self, thing):
        return '{}/{}'.format(self.name, thing)

    @property
    def y(self):
        return self._free_variables['y'].get_symbol(0)

    @property
    def z(self):
        return self._free_variables['z'].get_symbol(0)

    @property
    def l_wheel(self):
        return self._free_variables[self.l_wheel_s].get_symbol(0)

    @property
    def r_wheel(self):
        return self._free_variables[self.r_wheel_s].get_symbol(0)

    @property
    def l_wheel_name(self):
        return self.state_name(self.l_wheel_s)

    @property
    def l_wheel_name_long(self):
        return self._free_variables[self.l_wheel_s].name

    @property
    def r_wheel_name(self):
        return self.state_name(self.r_wheel_s)

    @property
    def r_wheel_name_long(self):
        return self._free_variables[self.r_wheel_s].name

    @property
    def free_variable_list(self):
        return [self._free_variables[self.l_wheel_s], self._free_variables[self.r_wheel_s]]

    def update_parent_T_child(self):
        rot = self.wheel_radius / self.wheel_dist * (self.r_wheel - self.l_wheel)
        # trans = wheel_radius / 2 * (self.r_wheel + self.l_wheel)
        trans_r = self.wheel_radius / 2 * (self.r_wheel)
        trans_l = self.wheel_radius / 2 * (self.l_wheel)
        # self.world.state[self.state_name('x')].position += np.cos(delta) * trans_vel * dt
        # self.world.state[self.state_name('y')].position += np.sin(delta) * trans_vel * dt
        # self.world.state[self.state_name('z')].position += rot_vel * dt
        odom_T_x = w.translation3(self.x, 0, 0)
        x_T_y = w.translation3(0, self.y, 0)
        y_T_z = w.rotation_matrix_from_axis_angle(w.vector3(0, 0, 1), self.z)
        z_T_rot = w.rotation_matrix_from_axis_angle(w.vector3(*self.rotation_axis), rot)
        rot_T_base = w.translation3(w.cos(self.z+0.01) * trans_r,
                                    w.sin(self.z+0.01) * trans_r,
                                    0)
        rot_T_base2 = w.translation3(w.cos(self.z-0.01) * trans_l,
                                     w.sin(self.z-0.01) * trans_l,
                                     0)
        self.parent_T_child = w.dot(self.parent_T_child, odom_T_x, x_T_y, y_T_z, z_T_rot, rot_T_base, rot_T_base2)

    def update_state(self, new_cmds, dt):
        try:
            l_wheel_vel = new_cmds[0][self.l_wheel_name_long]
            r_wheel_vel = new_cmds[0][self.r_wheel_name_long]
        except KeyError as e:
            # joint is currently not active
            return
        self.world.state[self.l_wheel_name].position = 0
        self.world.state[self.l_wheel_name].velocity = l_wheel_vel
        self.world.state[self.l_wheel_name].acceleration = new_cmds[1][self.l_wheel_name_long]
        self.world.state[self.l_wheel_name].jerk = new_cmds[2][self.l_wheel_name_long]

        self.world.state[self.r_wheel_name].position = 0
        self.world.state[self.r_wheel_name].velocity = r_wheel_vel
        self.world.state[self.r_wheel_name].acceleration = new_cmds[1][self.r_wheel_name_long]
        self.world.state[self.r_wheel_name].jerk = new_cmds[2][self.r_wheel_name_long]

        rot_vel = self.wheel_radius / self.wheel_dist * (r_wheel_vel - l_wheel_vel)
        trans_vel = self.wheel_radius / 2 * (r_wheel_vel + l_wheel_vel)

        delta = self.world.state[self.state_name('z')].position
        self.world.state[self.state_name('x')].position += np.cos(delta) * trans_vel * dt
        self.world.state[self.state_name('y')].position += np.sin(delta) * trans_vel * dt
        self.world.state[self.state_name('z')].position += rot_vel * dt
        pass

    def delete_limits(self):
        self.l_wheel.lower_limits = {}
        self.l_wheel.upper_limits = {}
        self.r_wheel.lower_limits = {}
        self.r_wheel.upper_limits = {}

    def delete_weights(self):
        self.l_wheel.quadratic_weights = {}
        self.r_wheel.quadratic_weights = {}

    def update_limits(self, linear_limits, angular_limits):
        self.delete_limits()
        for order, linear_limit in linear_limits.items():
            angular_limit = angular_limits[order]
            self._free_variables[self.l_wheel_s].set_upper_limit(order, linear_limit[self.l_wheel_name])
            self._free_variables[self.l_wheel_s].set_lower_limit(order, -linear_limit[self.l_wheel_name])
            self._free_variables[self.r_wheel_s].set_upper_limit(order, angular_limit[self.r_wheel_name])
            self._free_variables[self.r_wheel_s].set_lower_limit(order, -angular_limit[self.r_wheel_name])

    def update_weights(self, weights):
        self.delete_weights()
        for order, weight in weights.items():
            self._free_variables[self.l_wheel_s].quadratic_weights[order] = 1000 * weight[self.l_wheel_name]
            self._free_variables[self.r_wheel_s].quadratic_weights[order] = 1000 * weight[self.r_wheel_name]

    def get_limit_expressions(self, order):
        pass
