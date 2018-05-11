from rospkg import RosPack

import rospy

from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.object import WorldObject, to_urdf_string, VisualProperty, BoxShape, CollisionProperty, to_marker, \
    MeshShape
from giskardpy.plugin import Plugin
from giskardpy.pybullet_world import PyBulletWorld, ContactInfo
import giskardpy.trajectory as g
from giskardpy.utils import keydefaultdict


class PyBulletPlugin(Plugin):
    def __init__(self, js_identifier, collision_identifier, closest_point_identifier, gui=False,
                 marker=False):
        self.js_identifier = js_identifier
        self.collision_identifier = collision_identifier
        self.closest_point_identifier = closest_point_identifier
        self.robot_name = 'pr2'
        self.marker = marker
        self.world = PyBulletWorld(gui=gui)
        self.srv = rospy.Service('kitchen', Trigger, self.cb)
        self.marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.viz_gui = rospy.Service('~enable_gui', SetBool, self.enable_gui_cb)
        self.viz_gui = rospy.Service('~enable_marker', SetBool, self.enable_marker_cb)
        super(PyBulletPlugin, self).__init__()

    def enable_marker_cb(self, setbool):
        """
        :param setbool:
        :type setbool: std_srvs.srv._SetBool.SetBoolRequest
        :return:
        :rtype: SetBoolResponse
        """
        self.marker = setbool.data
        return SetBoolResponse()

    def enable_gui_cb(self, setbool):
        """
        :param setbool:
        :type setbool: std_srvs.srv._SetBool.SetBoolRequest
        :return:
        :rtype: SetBoolResponse
        """
        if setbool.data:
            self.world.muh(True)
        else:
            self.world.deactivate_viewer()
            self.world.muh(False)
        return SetBoolResponse()

    def cb(self, trigger):
        rospy.loginfo('loading kitchen')
        # self.world.spawn_object_from_urdf('kitchen', '{}/urdf/muh.urdf'.format(RosPack().get_path('iai_kitchen')))
        # self.world.spawn_object_from_urdf('shelf', '{}/urdf/shelves.urdf'.format(RosPack().get_path('iai_shelves')),
        #                                   [1.3,1,0])
        # my_obj = WorldObject(name='my_box',
        #                      visual_props=[VisualProperty(geometry=BoxShape(0.5, 1.5, 1.5),
        #                                                   origin=g.Transform(g.Point(.8,0,0),
        #                                                                      g.Quaternion(0,0,0,1)))],
        #                      collision_props=[CollisionProperty(geometry=BoxShape(0.5, 1.5, 1.5),
        #                                                         origin=g.Transform(g.Point(.8, 0, 0),
        #                                                                            g.Quaternion(0, 0, 0, 1)))])
        my_obj = WorldObject(name='my_sink',
                             visual_props=[VisualProperty(geometry=MeshShape('package://iai_kitchen/meshes/misc/Sink2.dae'),
                                                          origin=g.Transform(g.Point(.8,0,.6),
                                                                             g.Quaternion(0,0,0,1)))],
                             collision_props=[CollisionProperty(geometry=MeshShape('package://iai_kitchen/meshes/misc/Sink2.dae'),
                                                                origin=g.Transform(g.Point(.8, 0, .6),
                                                                                   g.Quaternion(0, 0, 0, 1)))])
        urdf_string = to_urdf_string(my_obj)
        self.world.spawn_object_from_urdf_str('box', urdf_string)
        self.marker_pub.publish(to_marker(my_obj))
        rospy.sleep(0.2)
        return TriggerResponse()

    def get_readings(self):
        collisions = self.world.check_collision()
        if self.marker:
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

    def start_once(self):
        self.collision_pub = rospy.Publisher('visualization_marker', Marker, queue_size=1)
        self.world.activate_viewer()
        #TODO get robot description from databus
        urdf = rospy.get_param('robot_description')
        self.world.spawn_urdf_str_robot(self.robot_name, urdf)

    def stop(self):
        pass
        # self.world.deactivate_viewer()

    def copy(self):
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
            m.action = Marker.DELETE
        self.collision_pub.publish(m)
        # rospy.sleep(0.05)

