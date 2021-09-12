from giskardpy import casadi_wrapper as w


class FixedJoint(object):
    def __init__(self, name, parent, child, parent_T_child=None, translation_offset=None, rotation_offset=None):
        self.name = name
        self.parent = parent
        self.child = child # type: giskardpy.model.world.Link

        if translation_offset is None:
            translation_offset = [0, 0, 0]
        if rotation_offset is None:
            rotation_offset = [0, 0, 0]
        if parent_T_child is None:
            self.parent_T_child = w.dot(w.translation3(*translation_offset), w.rotation_matrix_from_rpy(*rotation_offset))
        else:
            self.parent_T_child = parent_T_child

    def __repr__(self):
        return self.name

    @property
    def parent_link_name(self):
        return self.parent.name

    @property
    def child_link_name(self):
        return self.child.name


class MovableJoint(FixedJoint):
    def __init__(self, name, parent, child, free_variable, parent_T_child=None, translation_offset=None, rotation_offset=None):
        """
        :type name: str
        :type parent: Link
        :type child: Link
        :type translation_offset: list
        :type rotation_offset: list
        :type free_variable: giskardpy.qp.free_variable.FreeVariable
        """
        super(MovableJoint, self).__init__(name, parent, child, parent_T_child, translation_offset, rotation_offset)
        self.free_variable = free_variable

    @property
    def position_limits(self):
        return self.free_variable.get_lower_limit(0), self.free_variable.get_upper_limit(0)

    @property
    def velocity_limit(self):
        return self.free_variable.get_upper_limit(1)


class RevoluteJoint(MovableJoint):
    def __init__(self, name, parent, child, axis, free_variable, parent_T_child=None, translation_offset=None, rotation_offset=None):
        super(RevoluteJoint, self).__init__(name, parent, child, free_variable, parent_T_child, translation_offset, rotation_offset)
        self.parent_T_child = w.dot(self.parent_T_child,
                                    w.rotation_matrix_from_axis_angle(w.vector3(*axis), self.free_variable.get_symbol(0)))


class ContinuousJoint(MovableJoint):
    def __init__(self, name, parent, child, axis, free_variable, parent_T_child=None, translation_offset=None, rotation_offset=None):
        super(ContinuousJoint, self).__init__(name, parent, child, free_variable, parent_T_child, translation_offset, rotation_offset)
        self.parent_T_child = w.dot(self.parent_T_child,
                                    w.rotation_matrix_from_axis_angle(w.vector3(*axis), self.free_variable.get_symbol(0)))


class PrismaticJoint(MovableJoint):
    def __init__(self, name, parent, child, axis, free_variable, parent_T_child=None, translation_offset=None, rotation_offset=None):
        super(PrismaticJoint, self).__init__(name, parent, child, free_variable, parent_T_child, translation_offset, rotation_offset)
        translation_axis = (w.point3(*axis) * self.free_variable.get_symbol(0))
        self.parent_T_child = w.dot(self.parent_T_child, w.translation3(translation_axis[0],
                                                                        translation_axis[1],
                                                                        translation_axis[2]))


class MimicJoint(MovableJoint):
    def __init__(self, name, parent, child, free_variable, parent_T_child=None, translation_offset=None, rotation_offset=None):
        super(MimicJoint, self).__init__(name, parent, child, free_variable, parent_T_child, translation_offset, rotation_offset)
