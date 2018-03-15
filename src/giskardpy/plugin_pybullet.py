import rospy
from copy import deepcopy
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy.plugin import Plugin
from giskardpy.pybullet_world import PyBulletWorld


class PyBulletPlugin(Plugin):
    def __init__(self, js_identifier='js', collision_identifier='collision'):
        self.js_identifier = js_identifier
        self.collision_identifier = collision_identifier
        self.robot_name = 'pr2'
        self.world = PyBulletWorld()
        self.started = False
        super(PyBulletPlugin, self).__init__()

    def get_readings(self):
        collisions = self.world.check_collision()
        self.make_collision_markers(collisions)
        return {self.collision_identifier: collisions}

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

    def make_collision_markers(self, collisions):
        ma = MarkerArray()
        if len(collisions) > 0:
            for i, ((link1, link2), collision_info) in enumerate(collisions.items()):
                m = Marker()
                m.header.frame_id = 'base_footprint'
                m.action = Marker.ADD
                m.type = Marker.SPHERE
                m.pose.position = Point(*collision_info.position_on_a)
                m.id = i
                m.ns = 'pybullet collisions'
                m.scale = Vector3(0.03,0.03,0.03)
                m.color = ColorRGBA(1,0,0,1)
                ma.markers.append(m)
                m = deepcopy(m)
                m.pose.position = Point(*collision_info.position_on_b)
                ma.markers.append(m)
        else:
            m = Marker()
            m.action = Marker.DELETEALL
            ma.markers.append(m)
        self.collision_pub.publish(ma)

