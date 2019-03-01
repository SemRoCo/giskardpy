from geometry_msgs.msg import PoseStamped

from giskardpy.exceptions import RobotExistsException, DuplicateNameException
from giskardpy.symengine_robot import Robot
from giskardpy.urdf_object import URDFObject
from giskardpy.world_object import WorldObject


class World(object):
    def __init__(self):
        self.objects = {}
        self.robot = None # type: Robot

    # General ----------------------------------------------------------------------------------------------------------

    def soft_reset(self):
        """
        keeps robot and other important objects like ground plane
        """
        self.remove_all_objects()

    def hard_reset(self):
        """
        removes everything
        """
        self.soft_reset()
        self.remove_robot()

    def check_collisions(self, cut_off_distances):
        pass

    # Objects ----------------------------------------------------------------------------------------------------------

    def add_object(self, object_):
        """
        :type object_: URDFObject
        """
        if not isinstance(object_, URDFObject):
            object_ = WorldObject(object_.get_urdf())
        if self.has_robot() and self.get_robot().get_name() == object_.get_name():
            raise DuplicateNameException(u'object and robot have the same name')
        if self.has_object(object_.get_name()):
            raise DuplicateNameException(u'object with that name already exists')
        self.objects[object_.get_name()] = object_

    def get_object(self, name):
        """
        :type name: str
        :rtype: WorldObject
        """
        return self.objects[name]

    def get_objects(self):
        return self.objects

    def get_object_names(self):
        """
        :rtype: list
        """
        return list(self.objects.keys())

    def has_object(self, name):
        """
        Checks for objects with the same name.
        :type name: str
        :rtype: bool
        """
        return name in self.get_objects()

    def set_object_joint_state(self, name, joint_state):
        """
        :type name: str
        :param joint_state: joint name -> SingleJointState
        :type joint_state: dict
        """
        self.get_object(name).set_joint_state(joint_state)

    def remove_object(self, name):
        if self.has_object(name):
            del (self.objects[name])

    def remove_all_objects(self):
        self.objects = {}

    # Robot ------------------------------------------------------------------------------------------------------------

    def add_robot(self, robot, controlled_joints=None, base_pose=None):
        """
        :type robot: giskardpy.world_object.WorldObject
        :type controlled_joints: list
        :type base_pose: PoseStamped
        """
        if self.has_robot():
            raise RobotExistsException(u'A robot is already loaded')
        if self.has_object(robot.get_name()):
            raise DuplicateNameException(
                u'can\'t add robot; object with name "{}" already exists'.format(self.robot_name))
        self.robot = robot

    def get_robot(self):
        """
        :rtype: Robot
        """
        return self.robot

    def has_robot(self):
        """
        :rtype: bool
        """
        return self.robot is not None

    def set_robot_joint_state(self, joint_state):
        """
        Set the current joint state readings for a robot in the world.
        :param joint_state: joint name -> SingleJointState
        :type joint_state: dict
        """
        self.robot.set_joint_state(joint_state)

    def remove_robot(self):
        self.robot = None
