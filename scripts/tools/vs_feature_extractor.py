import rospy
from visualization_msgs.msg import MarkerArray, Marker
from scipy.spatial.transform import Rotation as R
import numpy as np

class VSFeatureExtractor:
    def __init__(self, object_name: str):
        rospy.init_node('vs_feature_extractor')
        rospy.Subscriber('/mujoco/visualization_marker_array', MarkerArray, self.callback)
        self.pub = rospy.Publisher('vs_feature', Marker, queue_size=1)
        self.object_name = object_name
        self.TYPE_CUBE = 1
        self.TYPE_SPHERE = 2

    def callback(self, msg: MarkerArray):
        for m in msg.markers:
            assert isinstance(m, Marker)
            if m.ns == self.object_name and m.type == self.TYPE_CUBE and m.id == 226:
                # initialize new marker to visualize the location of the top edge of the cup
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.type = self.TYPE_CUBE
                marker.ns = 'vs_feature'
                marker.id = 1
                marker.action = 0
                marker.pose = m.pose
                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                marker.color.g = 1
                marker.color.a = 1

                # calculate a position on the top edge  of the wall of the cup
                # calculate rotation matrix from quaternion of the Marker Pose
                r = R.from_quat([m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w])
                # use rotation matrix to calculate z-axis vector of the Marker frame in world coordinates
                root_V_z = np.dot(r.as_matrix(), np.array([0, 0, 1]))
                # create position vector of the Marker origin in world coordinates
                root_P_m = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
                # calculate upper edge of the box represented by the marker. Do this by adding half the size of the box
                # to the origin along the z-axis of the marker
                root_P_marker = root_P_m + root_V_z * m.scale.z/2
                # write data into a new marker
                marker.pose.position.x = root_P_marker[0]
                marker.pose.position.y = root_P_marker[1]
                marker.pose.position.z = root_P_marker[2]
                # publish new marker
                self.pub.publish(marker)


if __name__ == '__main__':
    vs_feature_extractor = VSFeatureExtractor('sync_create_cup2')
    rospy.spin()
