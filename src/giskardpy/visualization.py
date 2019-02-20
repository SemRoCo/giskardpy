from plugin import GiskardBehavior
import py_trees
from urdf_parser_py.urdf import URDF, Box, Mesh, Cylinder, Sphere
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from symengine_robot import hacky_urdf_parser_fix, Robot


class VisualizationBehavior(GiskardBehavior):
    def __init__(self, name, enable_visualization):
        super(VisualizationBehavior, self).__init__(name)
        self.enable_visualization = enable_visualization

    def setup(self, timeout):
        self.robot_d = rospy.get_param(u'/robot_description') #self.get_god_map().get_data(['robot_description'])
        self.robot = Robot(self.robot_d, 1.0)#URDF.from_xml_string(hacky_urdf_parser_fix(self.robot_d))
        self.publisher = rospy.Publisher('giskard_visualization', MarkerArray, queue_size=1)
        self.num_markers = len(self.robot._urdf_robot.links)
        self.robot_base = self.robot._urdf_robot.get_root()
        return True


    def update(self):
        if not self.enable_visualization:
            return py_trees.common.Status.SUCCESS

        self.fk_dict = self.get_god_map().get_data(['fk'])
        markers = []
        for index, link in enumerate(self.robot._urdf_robot.links):
            if not self.robot.has_link_visuals(link.name):
                continue

            marker = Marker()
            link_type = type(link.visual.geometry)

            if link_type == Mesh:
                marker.type = Marker.MESH_RESOURCE
                marker.mesh_resource = link.visuals[0].geometry.filename
                marker.scale.x = 1.0
                marker.scale.z = 1.0
                marker.scale.y = 1.0
                marker.mesh_use_embedded_materials = True
            elif link_type == Box:
                marker.type = Marker.CUBE
                marker.scale.x = link.visuals[0].geometry.size[0]
                marker.scale.y = link.visuals[0].geometry.size[1]
                marker.scale.z = link.visuals[0].geometry.size[2]
            elif link_type == Cylinder:
                marker.type = Marker.CYLINDER
                marker.scale.x = link.visuals[0].geometry.radius
                marker.scale.y = link.visuals[0].geometry.radius
                marker.scale.z = link.visuals[0].geometry.length
            elif link_type == Sphere:
                marker.type = Marker.SPHERE
                marker.scale.x = link.visuals[0].geometry.radius
                marker.scale.y = link.visuals[0].geometry.radius
                marker.scale.z = link.visuals[0].geometry.radius
            else:
                continue

            link_in_base = self.fk_dict[self.robot_base, link.name]
            marker.header.frame_id = self.robot_base
            marker.action = Marker.ADD
            marker.id = index
            marker.ns = "planning_visualization"
            marker.header.stamp = rospy.Time()
            marker.pose = link_in_base.pose
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            markers.append(marker)

        self.publisher.publish(markers)
        return py_trees.common.Status.SUCCESS

