from giskardpy.urdf_object import URDFObject


class WorldObject(URDFObject):
    def __init__(self, urdf, base_pose=None):
        super(WorldObject, self).__init__(urdf)
        self.set_base_pose(base_pose)

    @classmethod
    def from_urdf_file(cls, urdf_file, *args, **kwargs):
        return super(WorldObject, cls).from_urdf_file(urdf_file, *args, **kwargs)

    @classmethod
    def from_world_body(cls, world_body, *args, **kwargs):
        return super(WorldObject, cls).from_world_body(world_body, *args, **kwargs)

    @classmethod
    def from_parts(cls, robot_name, links, joints, *args, **kwargs):
        return super(WorldObject, cls).from_parts(robot_name, links, joints, *args, **kwargs)

    def set_base_pose(self, pose):
        """
        :param pose:
        :return:
        """
        pass

    def get_base_pose(self):
        pass

    def set_joint_state(self, joint_state):
        pass

    def get_joint_state(self):
        pass

    def get_self_collision_matrix(self):
        pass