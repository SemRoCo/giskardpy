from giskardpy.identifier import robot_description_identifier, default_joint_weight_identifier, js_identifier, \
    robot_identifier, controlled_joints_identifier, controllable_links_identifier, default_joint_vel_identifier
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import PluginBase
from giskardpy.symengine_robot import Robot
from giskardpy.utils import urdfs_equal


class RobotPlugin(PluginBase):

    def __init__(self, default_joint_vel_limit=None, default_joint_weight=None):
        """
        :type robot_description_identifier: str
        :type js_identifier: str
        :type default_joint_vel_limit: float
        """
        super(RobotPlugin, self).__init__()
        # if not isinstance(self.get_robot(), Robot):
        if default_joint_vel_limit is not None:
            self.get_god_map().safe_set_data([default_joint_weight_identifier], default_joint_weight)
        if default_joint_weight is not None:
            self.get_god_map().safe_set_data([default_joint_vel_identifier], default_joint_vel_limit)
            # self.__urdf_updated = True
            # self.controlled_joints = set()
            # self.controllable_links = set()

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

    def initialize(self):
        self.init_robot()
        # if self.__is_urdf_updated():
        #     self.init_robot()
        #     self.__urdf_updated = True
        # else:
        #     self.__urdf_updated = False

    def __is_urdf_updated(self):
        new_urdf = self.god_map.safe_get_data([robot_description_identifier])
        # TODO figure out a better solution which does not require the urdf to be rehashed all the time
        return self.get_robot() is None or not urdfs_equal(self.get_robot().get_urdf(), new_urdf)

    def was_urdf_updated(self):
        return self.__is_urdf_updated()
        # return self.__urdf_updated

    def init_robot(self):
        urdf = self.god_map.safe_get_data([robot_description_identifier])
        default_joint_weight = self.get_god_map().to_symbol([default_joint_weight_identifier])
        default_joint_vel = self.get_god_map().to_symbol([default_joint_vel_identifier])
        self.get_god_map().safe_set_data([robot_identifier], Robot(urdf, default_joint_vel, default_joint_weight))
        current_joints = JointStatesInput(self.god_map.to_symbol,
                                          self.get_robot().get_joint_names_controllable(),
                                          [js_identifier],
                                          [u'position'])
        self.get_robot().parse_urdf(current_joints.joint_map)
        self.update_controlled_joints_and_links()

    def get_controlled_joints(self):
        return self.get_god_map().safe_get_data([controlled_joints_identifier])

    def get_controlled_links(self):
        return self.get_god_map().safe_get_data([controllable_links_identifier])

    def get_robot(self):
        """
        :rtype: Robot
        """
        return self.get_god_map().safe_get_data([robot_identifier])

    def update_controlled_joints_and_links(self):
        """
        Gets controlled joints from god map and uses this to calculate the controllable link, which are written to
        the god map.
        """
        self.controllable_links = set()
        for joint_name in self.get_controlled_joints():
            self.controllable_links.update(self.get_robot().get_sub_tree_link_names_with_collision(joint_name))
        self.god_map.safe_set_data([controllable_links_identifier], self.controllable_links)




class RobotKinPlugin(RobotPlugin):
    def __init__(self):
        super(RobotKinPlugin, self).__init__(0, 0)
