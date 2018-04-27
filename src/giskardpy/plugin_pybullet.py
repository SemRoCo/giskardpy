import rospy
from copy import deepcopy, copy

from collections import defaultdict
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy.plugin import Plugin
from giskardpy.pybullet_world import PyBulletWorld, ContactInfo
from giskardpy.utils import keydefaultdict


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
        self.make_collision_markers(collisions)
        closest_point = keydefaultdict(lambda key:ContactInfo(None, None, None, None, None, (0,0,0), (10,10,10),
                                                              None, None, None))
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
            self.collision_pub = rospy.Publisher('visualization_marker', Marker, queue_size=1)
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

    # @profile
    def make_collision_markers(self, collisions):
        m = Marker()
        m.header.frame_id = 'base_footprint'
        m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.id = 1337
        m.ns = 'pybullet collisions'
        m.scale = Vector3(0.003, 0, 0)
        # m.color = ColorRGBA(1, 0, 0, 1)
        if len(collisions) > 0:
            # TODO visualize only specific contacts
            for ((link1, link2), collision_info) in collisions.items():
                if collision_info.contact_distance is not None:
                    if collision_info.contact_distance < 0.05:
                        m.points.append(Point(*collision_info.position_on_a))
                        m.points.append(Point(*collision_info.position_on_b))
                        m.colors.append(ColorRGBA(1,0,0,1))
                        m.colors.append(ColorRGBA(1,0,0,1))
                    elif collision_info.contact_distance < 0.1:
                        m.points.append(Point(*collision_info.position_on_a))
                        m.points.append(Point(*collision_info.position_on_b))
                        m.colors.append(ColorRGBA(0,1,0,1))
                        m.colors.append(ColorRGBA(0,1,0,1))
        else:
            m = Marker()
            m.action = Marker.DELETEALL
        self.collision_pub.publish(m)
        # rospy.sleep(0.05)

