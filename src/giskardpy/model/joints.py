import numpy as np

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.data_types import PrefixName
from giskardpy.qp.free_variable import FreeVariable


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
            if urdf_joint.type == 'revolute':
                joint = RevoluteJoint(name=joint_name,
                                      parent_link_name=parent_link_name,
                                      child_link_name=child_link_name,
                                      translation_offset=translation_offset,
                                      rotation_offset=rotation_offset,
                                      god_map=god_map,
                                      axis=urdf_joint.axis)
            elif urdf_joint.type == 'prismatic':
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
            elif urdf_joint.type == 'continuous':
                joint = ContinuousJoint(name=joint_name,
                                        parent_link_name=parent_link_name,
                                        child_link_name=child_link_name,
                                        translation_offset=translation_offset,
                                        rotation_offset=rotation_offset,
                                        god_map=god_map,
                                        axis=urdf_joint.axis)
            else:
                raise NotImplementedError('Joint of type {} is not supported'.format(urdf_joint.type))
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
                raise NotImplementedError('Joint of type {} is not supported'.format(urdf_joint.type))

        if isinstance(joint, MovableJoint):
            if isinstance(joint, MimicJoint):
                joint.set_free_variables([])
            else:
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

                free_variable = FreeVariable(symbols={
                    0: god_map.to_symbol(identifier.joint_states + [joint.name, 'position']),
                    1: god_map.to_symbol(identifier.joint_states + [joint.name, 'velocity']),
                    2: god_map.to_symbol(identifier.joint_states + [joint.name, 'acceleration']),
                    3: god_map.to_symbol(identifier.joint_states + [joint.name, 'jerk']),
                },
                    lower_limits=lower_limits,
                    upper_limits=upper_limits,
                    quadratic_weights={},
                    horizon_functions={1: 0.1})
                joint.set_free_variables([free_variable])
        return joint


class FixedJoint(Joint):
    pass


class MovableJoint(Joint):
    def __init__(self, name, parent_link_name, child_link_name, god_map, parent_T_child=None,
                 translation_offset=None, rotation_offset=None):
        """
        :type name: PrefixName
        :type parent_link_name: Link
        :type child_link_name: Link
        :type free_variable: FreeVariable
        :type translation_offset: list
        :type rotation_offset: list
        :type free_variable: giskardpy.qp.free_variable.FreeVariable
        """
        super(MovableJoint, self).__init__(name=name,
                                           parent_link_name=parent_link_name,
                                           child_link_name=child_link_name,
                                           parent_T_child=parent_T_child,
                                           translation_offset=translation_offset,
                                           rotation_offset=rotation_offset)
        self.god_map = god_map
        self._free_variables = []

    @property
    def free_variables(self):
        return self._free_variables

    def set_free_variables(self, free_variables):
        self._free_variables.extend(free_variables)

    def has_free_variables(self):
        return len(self.free_variables) > 0


class OneDofJoint(MovableJoint):
    @property
    def free_variable(self):
        """
        :rtype: FreeVariable
        """
        return self.free_variables[0]

    @property
    def position_symbol(self):
        return self.free_variable.get_symbol(0)

    @property
    def position_limits(self):
        return self.free_variable.get_lower_limit(0), self.free_variable.get_upper_limit(0)

    @property
    def velocity_limit(self):
        return self.free_variable.get_upper_limit(1)

    def delete_limits(self):
        self.free_variable.lower_limits = {}
        self.free_variable.upper_limits = {}

    def delete_weights(self):
        self.free_variable.quadratic_weights = {}


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

    def set_free_variables(self, free_variables):
        super(RevoluteJoint, self).set_free_variables(free_variables)
        self.parent_T_child = w.dot(self.parent_T_child,
                                    w.rotation_matrix_from_axis_angle(w.vector3(*self.axis),
                                                                      self.free_variable.get_symbol(0)))


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

    def set_free_variables(self, free_variables):
        super(ContinuousJoint, self).set_free_variables(free_variables)
        self.parent_T_child = w.dot(self.parent_T_child,
                                    w.rotation_matrix_from_axis_angle(w.vector3(*self.axis),
                                                                      self.free_variable.get_symbol(0)))


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

    def set_free_variables(self, free_variables):
        super(PrismaticJoint, self).set_free_variables(free_variables)
        translation_axis = (w.point3(*self.axis) * self.free_variable.get_symbol(0))
        self.parent_T_child = w.dot(self.parent_T_child, w.translation3(translation_axis[0],
                                                                        translation_axis[1],
                                                                        translation_axis[2]))


class MimicJoint(MovableJoint):
    pass


class MimicedPrismaticJoint(PrismaticJoint, MimicJoint):
    def __init__(self, name, parent_link_name, child_link_name, god_map, axis, mimed_joint_name, multiplier, offset,
                 parent_T_child=None, translation_offset=None, rotation_offset=None):
        super(MimicedPrismaticJoint, self).__init__(name=name,
                                                    parent_link_name=parent_link_name,
                                                    child_link_name=child_link_name,
                                                    god_map=god_map,
                                                    axis=axis,
                                                    parent_T_child=parent_T_child,
                                                    translation_offset=translation_offset,
                                                    rotation_offset=rotation_offset)
        self.axis = axis
        self.mimed_joint_name = mimed_joint_name
        self.multiplier = multiplier
        self.offset = offset

    @property
    def position_symbol(self):
        mimed_free_variable = self.god_map.to_symbol(identifier.joint_states + [self.mimed_joint_name, 'position'])
        multiplier = 1 if self.multiplier is None else self.multiplier
        offset = 0 if self.offset is None else self.offset
        return mimed_free_variable * multiplier + offset

    def set_free_variables(self, free_variables):
        expr = self.position_symbol
        translation_axis = (w.point3(*self.axis) * expr)
        self.parent_T_child = w.dot(self.parent_T_child, w.translation3(translation_axis[0],
                                                                        translation_axis[1],
                                                                        translation_axis[2]))
