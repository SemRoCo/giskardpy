class World(object):
    def __init__(self):
        self.objects = {}
        self.robot = None
        self.robot_name = u'robby'

    def add_object(self, name, urdf_object):
        if not self.has_object(name):
            self.objects[name] = urdf_object
        else:
            raise KeyError(u'object with that name already exists')

    def remove_object(self, name):
        if self.has_object(name):
            del(self.objects[name])

    def remove_all_objects(self):
        self.objects = {}

    def get_objects(self):
        return self.objects

    def get_object_names(self):
        """
        :rtype: list
        """
        return list(self.objects.keys())

    def get_object(self, name):
        return self.objects[name]

    def has_object(self, name):
        """
        Checks for objects with the same name.
        :type name: str
        :rtype: bool
        """
        return name in self.objects

    def get_robot(self):
        """
        :rtype: WorldObject
        """
        return self.robot

    def add_robot(self, urdf_object, controlled_joints, base_pose):
        self.robot = urdf_object

    def remove_robot(self):
        self.robot = None

    def has_robot(self):
        """
        :rtype: bool
        """
        return self.robot is not None

    def hard_reset(self):
        """
        removes everything
        """
        self.soft_reset()
        self.remove_robot()

    def soft_reset(self):
        """
        keeps robot and other important objects like ground plane
        """
        self.remove_all_objects()

    def check_collisions(self, cut_off_distances):
        pass

class WorldObject(object):
    def __init__(self, urdf_object):
        # TODO why not inherit from urdf object?
        self.urdf_object = urdf_object

    def get_pose(self):
        pass

    def get_configuration(self):
        pass

    def get_self_collision_matrix(self):
        pass

    def get_urdf_object(self):
        """
        :rtype: giskardpy.urdf_object.NewURDFObject
        """
        return self.urdf_object

    def set_pose(self):
        pass

    def set_configuration(self):
        pass