from plugin import GiskardBehavior
import py_trees
#import roslib; roslib.load_manifest('urdfdom_py')
from urdf_parser_py.urdf import URDF
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from symengine_robot import hacky_urdf_parser_fix


class VisualizationBehavior(GiskardBehavior):
    def __init__(self, name):
        super(VisualizationBehavior, self).__init__(name)

    def setup(self, timeout):
        self.robot_d = rospy.get_param(u'/robot_description')
        self.robot = URDF.from_xml_string(hacky_urdf_parser_fix(self.robot_d))
        self.publisher = rospy.Publisher('giskard_visualization', MarkerArray, queue_size=1)
        self.num_markers = len(self.robot.links)
        self.robot_base = self.robot.get_root()
        self.enable_visualization = False
        if rospy.has_param(u'/giskard/enable_visualization'):
            self.enable_visualization = rospy.get_param(u'/giskard/enable_visualization')
        return True


    def update(self):
        if not self.enable_visualization:
            return py_trees.common.Status.FAILURE

        self.fk_dict = self.get_god_map().get_data(['fk'])
        markers = []
        for index, link in enumerate(self.robot.links):
            link_in_base = self.fk_dict[self.robot_base, link.name]
            marker = Marker()
            marker.type = Marker.MESH_RESOURCE
            marker.header.frame_id = 'base_footprint'
            marker.action = Marker.ADD
            marker.id = index
            marker.ns = "giskard_visualization"
            marker.header.stamp = rospy.Time()
            if len(link.visuals) <= 0:
                continue

            try:
                marker.mesh_resource = link.visuals[0].geometry.filename
            except Exception as e:
                continue
            marker.pose.position.x = link_in_base.pose.position.x
            marker.pose.position.y = link_in_base.pose.position.y
            marker.pose.position.z = link_in_base.pose.position.z
            marker.pose.orientation.x = link_in_base.pose.orientation.x
            marker.pose.orientation.y = link_in_base.pose.orientation.y
            marker.pose.orientation.z = link_in_base.pose.orientation.z
            marker.pose.orientation.w = link_in_base.pose.orientation.w
            marker.scale.x = 1.0
            marker.scale.z = 1.0
            marker.scale.y = 1.0
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.mesh_use_embedded_materials = True
            markers.append(marker)

        self.publisher.publish(markers)
        return py_trees.common.Status.FAILURE

