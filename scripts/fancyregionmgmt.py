#!/usr/bin/env python3
import actionlib
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import robokudo_msgs.msg
import geometry_msgs.msg


class RegionManager:
    def __init__(self):
        self.region_names = ['shelf_a', 'shelf_b', 'shelf_c', 'shelf_d']
        self.region_free = [None, None, None, None]

        # in map coordinates
        self.region_points = [geometry_msgs.msg.Point(), geometry_msgs.msg.Point(),
                              geometry_msgs.msg.Point(), geometry_msgs.msg.Point(), ]

        region_std_place_y = 0.2
        region_std_place_z = 0.79

        self.region_points[0].x = 8.15
        self.region_points[0].y = region_std_place_y
        self.region_points[0].z = region_std_place_z

        self.region_points[1].x = 7.97
        self.region_points[1].y = region_std_place_y
        self.region_points[1].z = region_std_place_z

        self.region_points[2].x = 7.79
        self.region_points[2].y = region_std_place_y
        self.region_points[2].z = region_std_place_z

        self.region_points[3].x = 7.61
        self.region_points[3].y = region_std_place_y
        self.region_points[3].z = region_std_place_z

        # Region vis marker
        self.marker_pub = rospy.Publisher('region_placing_visualization_marker', Marker, queue_size=10)

    def publish_marker_for_regions(self, region_idx):
        # Create a Marker message
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "region_placing"
        marker.id = region_idx
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Define points
        marker.points.append(self.region_points[region_idx])

        # Set the scale of the marker
        marker.scale.x = 0.1  # Point width
        marker.scale.y = 0.1  # Point height

        # Set the color of the marker
        marker.color.a = 1.0
        if self.region_free[region_idx]:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif not self.region_free[region_idx]:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0

        # Publish the marker
        self.marker_pub.publish(marker)

    def publish_all_markers_for_regions(self):
        for (idx, _) in enumerate(self.region_free):
            self.publish_marker_for_regions(idx)

    def update_region_free(self, region_idx: int, result: robokudo_msgs.msg.QueryResult, state):
        if state != actionlib.GoalStatus.SUCCEEDED:
            print(f"Received error/abort while region query: {state}")
            self.region_free[region_idx] = False
            return

        if result.description == 'free':
            self.region_free[region_idx] = True
        else:
            self.region_free[region_idx] = True

    def position_with_most_trues_final(self, lst):
        max_count = -1
        position = -1

        for i in range(len(lst)):
            surrounding_count = 0
            if i > 0:
                surrounding_count += lst[i - 1]
            if i < len(lst) - 1:
                surrounding_count += lst[i + 1]

            # Adjust the scoring to prioritize positions that are true and have the maximum number of 'true's around them.
            count = surrounding_count + (2 if lst[i] else 0)

            if count > max_count:
                max_count = count
                position = i

        return position

