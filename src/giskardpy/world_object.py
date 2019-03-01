from giskardpy.urdf_object import URDFObject


class WorldObject(URDFObject):
    def __init__(self, urdf_object):
        super(WorldObject, self).__init__(urdf_object)

    def set_base_pose(self):
        pass

    def get_base_pose(self):
        pass

    def set_joint_state(self, joint_state):
        pass

    def get_joint_state(self):
        pass

    def get_self_collision_matrix(self):
        pass