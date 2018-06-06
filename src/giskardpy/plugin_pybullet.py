from collections import defaultdict
from itertools import product
from rospkg import RosPack
from time import time
import numpy as np
import rospy

from geometry_msgs.msg import Point, Vector3, PoseStamped, PointStamped
from giskard_msgs.msg import CollisionEntry, WorldBody
from multiprocessing import Lock
from std_msgs.msg import ColorRGBA
from std_srvs.srv import SetBool, SetBoolResponse
from sensor_msgs.msg import JointState
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.exceptions import CorruptShapeException, UnknownBodyException, DuplicateObjectNameException, \
    IntersectingCollisionException, UnsupportedOptionException
from giskardpy.object import WorldObject, to_urdf_string, VisualProperty, BoxShape, CollisionProperty, to_marker, \
    MeshShape, from_msg, from_pose_msg
from giskardpy.plugin import Plugin
from giskardpy.pybullet_world import PyBulletWorld, ContactInfo
from giskardpy.symengine_wrappers import euclidean_distance
from giskardpy.tfwrapper import transform_pose, lookup_transform, transform_point, transform_vector
from giskardpy.trajectory import ClosestPointInfo
from giskardpy.utils import keydefaultdict, to_joint_state_dict, to_point_stamped, to_vector3_stamped, to_list


class PyBulletPlugin(Plugin):
    def __init__(self, js_identifier, collision_identifier, closest_point_identifier, collision_goal_identifier,
                 map_frame, root_link, default_collision_avoidance_distance, path_to_data_folder='', gui=False,
                 marker=False):
        self.collision_goal_identifier = collision_goal_identifier
        self.path_to_data_folder = path_to_data_folder
        self.js_identifier = js_identifier
        self.collision_identifier = collision_identifier
        self.closest_point_identifier = closest_point_identifier
        self.default_collision_avoidance_distance = default_collision_avoidance_distance
        self.map_frame = map_frame
        self.robot_root = root_link
        self.robot_name = 'pr2'
        self.global_reference_frame_name = 'map'
        self.marker = marker
        self.gui = gui
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects
        super(PyBulletPlugin, self).__init__()

    def copy(self):
        cp = self.__class__(js_identifier=self.js_identifier,
                            collision_identifier=self.collision_identifier,
                            closest_point_identifier=self.closest_point_identifier,
                            collision_goal_identifier=self.collision_goal_identifier,
                            map_frame=self.map_frame,
                            root_link=self.robot_root,
                            path_to_data_folder=self.path_to_data_folder,
                            gui=self.gui,
                            default_collision_avoidance_distance=self.default_collision_avoidance_distance)
        cp.world = self.world
        cp.marker = self.marker
        # cp.srv = self.srv
        # cp.viz_gui = self.viz_gui
        cp.collision_pub = self.collision_pub
        return cp

    def start_once(self):
        self.world = PyBulletWorld(gui=self.gui, path_to_data_folder=self.path_to_data_folder)
        self.srv = rospy.Service('~update_world', UpdateWorld, self.update_world_cb)
        self.viz_gui = rospy.Service('~enable_marker', SetBool, self.enable_marker_cb)
        self.collision_pub = rospy.Publisher('visualization_marker', Marker, queue_size=1)
        self.world.activate_viewer()
        # TODO get robot description from god map
        urdf = rospy.get_param('robot_description')
        self.world.spawn_robot_from_urdf(self.robot_name, urdf)

    def enable_marker_cb(self, setbool):
        """
        :param setbool:
        :type setbool: std_srvs.srv._SetBool.SetBoolRequest
        :return:
        :rtype: SetBoolResponse
        """
        self.marker = setbool.data
        return SetBoolResponse()

    def update_world_cb(self, req):
        """
        Callback function of the ROS service to update the internal giskard world.
        :param req: Service request as received from the service client.
        :type req: UpdateWorldRequest
        :return: Service response, reporting back any runtime errors that occurred.
        :rtype UpdateWorldResponse
        """
        with self.lock:
            try:
                if req.operation is UpdateWorldRequest.ADD:
                    # Check that no object with this name already exists.
                    if self.world.has_object(req.body.name) or self.world.get_robot().has_attached_object(
                            req.body.name):
                        DuplicateObjectNameException('Cannot spawn object "{}" because an object with such a '
                                                     'name already exists'.format(req.body.name))

                    # CASE: Spawn rigidly attached object
                    if req.rigidly_attached:
                        # Catch unsupported sub-case of rigidly attaching a robot to a robot
                        if req.body.type is WorldBody.URDF_BODY:
                            raise UnsupportedOptionException('Attaching URDF bodies to robots is not supported.')
                        self.world.get_robot().attach_object(from_msg(req.body), req.pose.header.frame_id,
                                                             from_pose_msg(req.pose.pose))

                    # CASE: Spawn a 'free' object
                    else:
                        urdf_string = req.body.urdf if req.body.type is WorldBody.URDF_BODY else to_urdf_string(
                            from_msg(req.body))
                        pose = from_pose_msg(transform_pose(self.global_reference_frame_name, req.pose).pose)
                        self.world.spawn_object_from_urdf(req.body.name, urdf_string, pose)
                        # SUB-CASE: If it is an articulated object, open up a joint state subscriber
                        if req.body.joint_state_topic:
                            callback = (lambda (msg): self.object_js_cb(req.body.name, msg))
                            self.object_js_subs[req.body.name] = \
                                rospy.Subscriber(req.body.joint_state_topic, JointState, callback, queue_size=1)

                elif req.operation is UpdateWorldRequest.REMOVE:
                    if self.world.has_object(req.body.name):
                        self.world.delete_object(req.body.name)
                        if self.object_js_subs.has_key(req.body.name):
                            self.object_js_subs[req.body.name].unregister()
                            del (self.object_js_subs[req.body.name])
                            del (self.object_joint_states[req.body.name])
                    elif self.world.get_robot().has_attached_object(req.body.name):
                        self.world.get_robot().detach_object(req.body.name)
                    else:
                        raise UnknownBodyException('Cannot delete unknown object {}'.format(req.body.name))
                elif req.operation is UpdateWorldRequest.ALTER:
                    # TODO: implement me
                    pass
                elif req.operation is UpdateWorldRequest.REMOVE_ALL:
                    self.world.delete_all_objects()
                    self.world.get_robot().detach_all_objects()
                    for object_name in self.object_js_subs.keys():
                        self.object_js_subs[object_name].unregister()
                        del (self.object_js_subs[object_name])
                        del (self.object_joint_states[object_name])
                else:
                    return UpdateWorldResponse(UpdateWorldResponse.INVALID_OPERATION,
                                               "Received invalid operation code: {}".format(req.operation))
                return UpdateWorldResponse()
            except CorruptShapeException as e:
                return UpdateWorldResponse(UpdateWorldResponse.CORRUPT_SHAPE_ERROR, e.message)
            except UnknownBodyException as e:
                return UpdateWorldResponse(UpdateWorldResponse.MISSING_BODY_ERROR, e.message)
            except DuplicateObjectNameException as e:
                return UpdateWorldResponse(UpdateWorldResponse.DUPLICATE_BODY_ERROR, e.message)
            except UnsupportedOptionException as e:
                return UpdateWorldResponse(UpdateWorldResponse.UNSUPPORTED_OPTIONS, e.message)

    def update(self):
        with self.lock:
            js = self.god_map.get_data([self.js_identifier])
            self.world.set_joint_state(js)
            for object_name, object_joint_state in self.object_joint_states.items():
                self.world.get_object(object_name).set_joint_state(object_joint_state)
            p = lookup_transform(self.map_frame, self.robot_root)
            self.world.get_robot().set_base_pose(position=[p.pose.position.x,
                                                           p.pose.position.y,
                                                           p.pose.position.z],
                                                 orientation=[p.pose.orientation.x,
                                                              p.pose.orientation.y,
                                                              p.pose.orientation.z,
                                                              p.pose.orientation.w])

            collision_goals = self.god_map.get_data([self.collision_goal_identifier])
            if collision_goals is None:
                collision_goals = []
            allowed_collisions = set()
            distances = defaultdict(lambda: self.default_collision_avoidance_distance)
            for collision_entry in collision_goals:  # type: CollisionEntry
                if collision_entry.body_b == '' and \
                        collision_entry.type not in [CollisionEntry.ALLOW_ALL_COLLISIONS,
                                                     CollisionEntry.AVOID_ALL_COLLISIONS]:
                    raise Exception('body_b not set')

                if collision_entry.body_b == '' and collision_entry.link_b != '':
                    raise Exception('body_b is empty but link_b is not')

                if collision_entry.robot_link == '':
                    links_a = self.world.get_robot().get_link_names()
                else:
                    links_a = [collision_entry.robot_link]

                if collision_entry.body_b == '':
                    bodies_b = self.world.get_object_list()
                else:
                    bodies_b = [collision_entry.body_b]
                for body_b in bodies_b:
                    if collision_entry.link_b == '':
                        links_b = self.world.get_object(body_b).get_link_names()
                    else:
                        links_b = [collision_entry.link_b]

                    for link_a, link_b in product(links_a, links_b):
                        key = (link_a, body_b, link_b)
                        if collision_entry.type == CollisionEntry.ALLOW_COLLISION or \
                                collision_entry.type == CollisionEntry.ALLOW_ALL_COLLISIONS:
                            allowed_collisions.add(key)
                        if collision_entry.type == CollisionEntry.AVOID_COLLISION or \
                                collision_entry.type == CollisionEntry.AVOID_ALL_COLLISIONS:
                            distances[key] = collision_entry.min_dist

            collisions = self.world.check_collisions(distances, allowed_collisions)

            closest_point = keydefaultdict(lambda k: ClosestPointInfo((10, 0, 0),
                                                                      (0, 0, 0),
                                                                      1e9,
                                                                      self.default_collision_avoidance_distance,
                                                                      k,
                                                                      '',
                                                                      (1, 0, 0)))
            for key, collision_info in collisions.items():  # type: ((str, str), ContactInfo)
                link1 = key[0]
                a_in_robot_root = to_list(transform_point(self.robot_root,
                                                          to_point_stamped(self.map_frame,
                                                                           collision_info.position_on_a)))
                b_in_robot_root = to_list(transform_point(self.robot_root,
                                                          to_point_stamped(self.map_frame,
                                                                           collision_info.position_on_b)))
                n_in_robot_root = to_list(transform_vector(self.robot_root,
                                                           to_vector3_stamped(self.map_frame,
                                                                              collision_info.contact_normal_on_b)))
                cpi = ClosestPointInfo(a_in_robot_root, b_in_robot_root, collision_info.contact_distance,
                                       distances[key], key[0], key[2], n_in_robot_root)
                # if cpi.contact_distance < 0:
                #     raise IntersectingCollisionException(key)
                if link1 in closest_point:
                    closest_point[link1] = min(closest_point[link1], cpi, key=lambda x: x.contact_distance)
                else:
                    closest_point[link1] = cpi

            if self.marker:
                self.make_cpi_markers(closest_point)

            self.god_map.set_data([self.collision_identifier], None)
            self.god_map.set_data([self.closest_point_identifier], closest_point)

    def stop(self):
        pass
        # self.world.deactivate_viewer()

    def make_cpi_markers(self, collisions):
        m = Marker()
        m.header.frame_id = self.robot_root
        m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.id = 1337
        m.ns = 'pybullet collisions'
        m.scale = Vector3(0.003, 0, 0)
        if len(collisions) > 0:
            for collision_info in collisions.values(): # type: ClosestPointInfo
                red_threshold = collision_info.min_dist
                yellow_threshold = collision_info.min_dist * 2
                green_threshold = collision_info.min_dist * 3

                if collision_info.contact_distance < green_threshold:
                    m.points.append(Point(*collision_info.position_on_a))
                    m.points.append(Point(*collision_info.position_on_b))
                    m.colors.append(ColorRGBA(0, 1, 0, 1))
                    m.colors.append(ColorRGBA(0, 1, 0, 1))
                if collision_info.contact_distance < yellow_threshold:
                    m.colors[-2] = ColorRGBA(1, 1, 0, 1)
                    m.colors[-1] = ColorRGBA(1, 1, 0, 1)
                if collision_info.contact_distance < red_threshold:
                    m.colors[-2] = ColorRGBA(1, 0, 0, 1)
                    m.colors[-1] = ColorRGBA(1, 0, 0, 1)
        else:
            m.action = Marker.DELETE
        self.collision_pub.publish(m)

    def object_js_cb(self, object_name, msg):
        """
        Callback message for ROS Subscriber on JointState to get states of articulated objects into world.
        :param object_name: Name of the object for which the Joint State message is.
        :type object_name: str
        :param msg: Current state of the articulated object that shall be set in the world.
        :type msg: JointState
        :return: Nothing
        """
        self.object_joint_states[object_name] = to_joint_state_dict(msg)
