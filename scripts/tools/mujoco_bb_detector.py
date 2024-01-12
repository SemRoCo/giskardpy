import rospy
from visualization_msgs.msg import MarkerArray, Marker
import threading
import time
from scipy.spatial.transform import Rotation
import numpy as np

# TODO: Debug, it does not work well with the cup in the pr2 demo
class MujocoBBDetector:
    def __init__(self, sub_topic: str, pub_topic: str):
        self.objects = {}
        rospy.Subscriber(sub_topic, MarkerArray, self.callback)
        self.pub = rospy.Publisher(pub_topic, MarkerArray, queue_size=1)
        # timer = threading.Timer(1 / 100, self.publish)
        # timer.daemon = True
        # timer.start()

    def callback(self, data: MarkerArray):
        for marker in data.markers:
            assert isinstance(marker, Marker)
            ns = marker.ns
            if 'ball' in ns:
                continue
            if ns not in self.objects.keys():
                self.objects[ns] = [(marker.pose, marker.scale)]
            else:
                self.objects[ns].append((marker.pose, marker.scale))

        marker_array = MarkerArray()
        for key in self.objects.keys():
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            object_data = self.objects[key]
            px = 0
            py = 0
            pz = 0
            counter = 0
            for pose, scale in object_data:
                # Calculate 8 vertices using the scale parameter
                x = scale.x / 2
                y = scale.y / 2
                z = scale.z / 2
                vertices = [[x, y, z],
                            [-x, y, z],
                            [x, -y, z],
                            [x, y, -z],
                            [-x, -y, z],
                            [x, -y, -z],
                            [-x, y, -z],
                            [-x, -y, -z]]
                # Calculate root_T_marker using rotation and position
                rotation = Rotation.from_quat(
                    [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
                root_T_marker = np.zeros((4, 4))
                root_T_marker[:3, :3] = rotation.as_matrix()[:3, :3]
                root_T_marker[:, 3] = [pose.position.x, pose.position.y, pose.position.z, 1]
                # transform each vertice and compare with the below min/max

                # Calculate the size of the bounding box
                rot_scale = np.dot(rotation.as_matrix(), [scale.x, scale.y, scale.z])
                bbox_size_x = scale.x #rot_scale[0]
                bbox_size_y = scale.y #rot_scale[1]
                bbox_size_z = scale.z #rot_scale[2]

                # Calculate the center of the bounding box
                bbox_center_x = pose.position.x
                bbox_center_y = pose.position.y
                bbox_center_z = pose.position.z

                px += pose.position.x
                py += pose.position.y
                pz += pose.position.z
                counter += 1

                # Update min and max coordinates in each dimension
                min_x = min(min_x, bbox_center_x - bbox_size_x / 2)
                min_y = min(min_y, bbox_center_y - bbox_size_y / 2)
                min_z = min(min_z, bbox_center_z - bbox_size_z / 2)
                max_x = max(max_x, bbox_center_x + bbox_size_x / 2)
                max_y = max(max_y, bbox_center_y + bbox_size_y / 2)
                max_z = max(max_z, bbox_center_z + bbox_size_z / 2)

            # Calculate the size of the overall bounding box
            bbox_size_x = max_x - min_x
            bbox_size_y = max_y - min_y
            bbox_size_z = max_z - min_z

            # Calculate the center of the overall bounding box
            bbox_center_x = (min_x + max_x) / 2
            bbox_center_y = (min_y + max_y) / 2
            bbox_center_z = (min_z + max_z) / 2
            # Print or use the bounding box information as needed
            print("Bounding Box Size (X, Y, Z):", bbox_size_x, bbox_size_y, bbox_size_z)
            print("Bounding Box Center (X, Y, Z):", bbox_center_x, bbox_center_y, bbox_center_z)
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = 1
            marker.ns = key
            marker.id = 1
            marker.action = 0
            marker.pose.orientation = pose.orientation
            # marker.pose.orientation = object_data[0][0].orientation

            marker.pose.position.x = px / counter#bbox_center_x
            marker.pose.position.y = py / counter#bbox_center_y
            marker.pose.position.z = pz / counter#bbox_center_z
            marker.scale.x = bbox_size_x
            marker.scale.y = bbox_size_y
            marker.scale.z = bbox_size_z
            marker.color.g = 1
            marker.color.a = 0.5
            marker_array.markers.append(marker)
        self.pub.publish(marker_array)
        self.objects.clear()


if __name__ == '__main__':
    rospy.init_node('mujocoBBdetector')
    MujocoBBDetector('/mujoco/visualization_marker_array', 'mujoco_object_bb')
    rospy.spin()
