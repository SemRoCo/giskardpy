import rospy
from copy import deepcopy, copy

from collections import defaultdict
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy.plugin import Plugin
from giskardpy.pybullet_world import PyBulletWorld


class PyBulletPlugin(Plugin):
    def __init__(self, js_identifier='js', collision_identifier='collision', closest_point_identifier='cpi'):
        self.js_identifier = js_identifier
        self.collision_identifier = collision_identifier
        self.closest_point_identifier = closest_point_identifier
        self.robot_name = 'pr2'
        self.world = PyBulletWorld()
        self.started = False
        super(PyBulletPlugin, self).__init__()

    def get_readings(self):
        collisions = self.world.check_collision()
        # self.make_collision_markers(collisions)
        closest_point = {}
        for (link1, link2), collision_info in collisions.items():
            if link1 in closest_point:
                closest_point[link1] = min(closest_point[link1], collision_info, key=lambda x: x.contact_distance)
            else:
                closest_point[link1] = collision_info
        return {self.collision_identifier: collisions,
                self.closest_point_identifier: closest_point}

    def update(self):
        js = self.god_map.get_data(self.js_identifier)
        self.world.set_joint_state(self.robot_name, js)

    def start(self, god_map):
        super(PyBulletPlugin, self).start(god_map)
        if not self.started:
            self.collision_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=1)
            self.world.activate_viewer()
            #TODO get robot description from databus
            urdf = rospy.get_param('robot_description')
            self.world.spawn_urdf_str_robot(self.robot_name, urdf)
            self.started = True

    def stop(self):
        pass
        # self.world.deactivate_viewer()

    def get_replacement_parallel_universe(self):
        return self

    def default_marker(self, position, i):
        m = Marker()
        m.header.frame_id = 'base_footprint'
        m.action = Marker.ADD
        m.type = Marker.SPHERE
        m.pose.position = Point(*position)
        m.id = i
        m.ns = 'pybullet collisions'
        m.scale = Vector3(0.03, 0.03, 0.03)
        m.color = ColorRGBA(1, 0, 0, 1)
        return m

    # @profile
    def make_collision_markers(self, collisions):
        ma = MarkerArray()
        if len(collisions) > 0:
            # TODO visualize only specific contacts
            i = 0
            for ((link1, link2), collision_info) in collisions.items():
                if link1 == 'l_gripper_palm_link' and link2 == 'r_gripper_palm_link' or \
                    link2 == 'l_gripper_palm_link' and link1 == 'r_gripper_palm_link':
                    ma.markers.append(self.default_marker(collision_info.position_on_a, i))
                    ma.markers.append(self.default_marker(collision_info.position_on_b, -i))
                    i += 1
                # ma.markers.extend([self.default_marker(collision_info.position_on_a, i) for i, collision_info in enumerate(collisions.values())])
                # ma.markers.extend([self.default_marker(collision_info.position_on_b, i) for i, collision_info in enumerate(collisions.values())])
            #     m = Marker()
            #     m.header.frame_id = 'base_footprint'
            #     m.action = Marker.ADD
            #     m.type = Marker.SPHERE
            #     m.pose.position = Point(*collision_info.position_on_a)
            #     m.id = i
            #     m.ns = 'pybullet collisions'
            #     m.scale = Vector3(0.03,0.03,0.03)
            #     m.color = ColorRGBA(1,0,0,1)
            #     ma.markers.append(m)
            #     m = copy(m)
            #     m.pose.position = Point(*collision_info.position_on_b)
            #     ma.markers.append(m)
        else:
            m = Marker()
            m.action = Marker.DELETEALL
            ma.markers.append(m)
        self.collision_pub.publish(ma)

