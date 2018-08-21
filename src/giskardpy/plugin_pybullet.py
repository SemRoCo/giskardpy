import traceback
from itertools import product
import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3
from giskard_msgs.msg import CollisionEntry, WorldBody
from multiprocessing import Lock
from std_msgs.msg import ColorRGBA
from std_srvs.srv import SetBool, SetBoolResponse
from sensor_msgs.msg import JointState
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest
from visualization_msgs.msg import Marker, MarkerArray
from giskardpy.exceptions import CorruptShapeException, UnknownBodyException, \
    UnsupportedOptionException, DuplicateNameException, PhysicsWorldException
from giskardpy.object import to_marker, world_body_to_urdf_object, from_pose_msg
from giskardpy.plugin import Plugin
from giskardpy.pybullet_world import PyBulletWorld, ContactInfo
from giskardpy.tfwrapper import transform_pose, lookup_transform, transform_point, transform_vector
from giskardpy.data_types import ClosestPointInfo
from giskardpy.utils import keydefaultdict, to_joint_state_dict, to_point_stamped, to_vector3_stamped, to_list


class PyBulletPlugin(Plugin):
    def __init__(self, js_identifier, collision_identifier, closest_point_identifier, collision_goal_identifier,
                 controllable_links_identifier, robot_description_identifier,
                 map_frame, root_link, default_collision_avoidance_distance, path_to_data_folder='', gui=False,
                 marker=False, enable_self_collision=True):
        self.collision_goal_identifier = collision_goal_identifier
        self.controllable_links_identifier = controllable_links_identifier
        self.path_to_data_folder = path_to_data_folder
        self.js_identifier = js_identifier
        self.collision_identifier = collision_identifier
        self.closest_point_identifier = closest_point_identifier
        self.default_collision_avoidance_distance = default_collision_avoidance_distance
        self.robot_description_identifier = robot_description_identifier
        self.enable_self_collision = enable_self_collision
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
                            controllable_links_identifier=self.controllable_links_identifier,
                            map_frame=self.map_frame,
                            root_link=self.robot_root,
                            path_to_data_folder=self.path_to_data_folder,
                            gui=self.gui,
                            default_collision_avoidance_distance=self.default_collision_avoidance_distance,
                            robot_description_identifier=self.robot_description_identifier,
                            enable_self_collision=self.enable_self_collision)
        cp.world = self.world
        cp.marker = self.marker
        # cp.srv = self.srv
        # cp.viz_gui = self.viz_gui
        cp.pub_collision_marker = self.pub_collision_marker
        return cp

    def start_once(self):
        self.world = PyBulletWorld(gui=self.gui, path_to_data_folder=self.path_to_data_folder)
        self.srv_update_world = rospy.Service('~update_world', UpdateWorld, self.update_world_cb)
        self.srv_viz_gui = rospy.Service('~enable_marker', SetBool, self.enable_marker_cb)
        self.pub_collision_marker = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=1)
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
        # TODO test me
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
        # TODO block or queue updates while planning
        with self.lock:
            try:
                if req.operation is UpdateWorldRequest.ADD:
                    if req.rigidly_attached:
                        self.attach_object(req)
                    else:
                        self.add_object(req)

                elif req.operation is UpdateWorldRequest.REMOVE:
                    self.remove_object(req.body.name)
                # TODO implement alter
                elif req.operation is UpdateWorldRequest.REMOVE_ALL:
                    self.clear_world()
                else:
                    return UpdateWorldResponse(UpdateWorldResponse.INVALID_OPERATION,
                                               u'Received invalid operation code: {}'.format(req.operation))
                self.publish_object_as_marker(req)
                return UpdateWorldResponse()
            except CorruptShapeException as e:
                return UpdateWorldResponse(UpdateWorldResponse.CORRUPT_SHAPE_ERROR, str(e))
            except UnknownBodyException as e:
                return UpdateWorldResponse(UpdateWorldResponse.MISSING_BODY_ERROR, str(e))
            except DuplicateNameException as e:
                return UpdateWorldResponse(UpdateWorldResponse.DUPLICATE_BODY_ERROR, str(e))
            except UnsupportedOptionException as e:
                return UpdateWorldResponse(UpdateWorldResponse.UNSUPPORTED_OPTIONS, str(e))
            except Exception as e:
                traceback.print_exc()
                return UpdateWorldResponse(UpdateWorldResponse.UNSUPPORTED_OPTIONS, u'{}: {}'.format(e.__class__.__name__,
                                                                                                     str(e)))

    def add_object(self, req):
        """
        :type req: UpdateWorldRequest
        :return:
        """
        world_body = req.body
        global_pose = from_pose_msg(transform_pose(self.global_reference_frame_name, req.pose).pose)
        if world_body.type is WorldBody.URDF_BODY:
            #TODO test me
            self.world.spawn_object_from_urdf_str(world_body.name, world_body.urdf, global_pose)
        else:
            self.world.spawn_urdf_object(world_body_to_urdf_object(world_body), global_pose)

        # SUB-CASE: If it is an articulated object, open up a joint state subscriber
        if world_body.joint_state_topic:
            callback = (lambda msg: self.object_js_cb(world_body.name, msg))
            self.object_js_subs[world_body.name] = \
                rospy.Subscriber(world_body.joint_state_topic, JointState, callback, queue_size=1)


    def attach_object(self, req):
        if req.body.type is WorldBody.URDF_BODY:
            raise UnsupportedOptionException(u'Attaching URDF bodies to robots is not supported.')
        self.world.attach_object(world_body_to_urdf_object(req.body),
                                 req.pose.header.frame_id,
                                 from_pose_msg(req.pose.pose))

    def remove_object(self, name):
        if self.world.has_object(name):
            self.world.delete_object(name)
            if self.object_js_subs.has_key(name):
                self.object_js_subs[name].unregister()
                del (self.object_js_subs[name])
                try:
                    del (self.object_joint_states[name])
                except:
                    pass
        elif self.world.get_robot().has_attached_object(name):
            self.world.get_robot().detach_object(name)
        else:
            raise UnknownBodyException(u'Cannot delete unknown object {}'.format(name))

    def publish_object_as_marker(self, req):
        try:
            ma = to_marker(req)
            self.pub_collision_marker.publish(ma)
        except:
            pass

    def clear_world(self):
        self.pub_collision_marker.publish(MarkerArray([Marker(action=Marker.DELETEALL)]))
        for object_name in self.world.get_object_names():
            if object_name != u'plane': #TODO get rid of this hard coded special case
                self.remove_object(object_name)
        self.world.get_robot().detach_all_objects()

    def update(self):
        with self.lock:
            # TODO only update urdf if it has changed
            self.god_map.set_data([self.robot_description_identifier], self.world.get_robot().get_urdf())

            js = self.god_map.get_data([self.js_identifier])
            if js is not None:
                self.world.set_robot_joint_state(js)
            for object_name, object_joint_state in self.object_joint_states.items():
                self.world.get_object(object_name).set_joint_state(object_joint_state)

            # TODO we can look up the transform outside of this loop
            p = lookup_transform(self.map_frame, self.robot_root)
            self.world.get_robot().set_base_pose(position=[p.pose.position.x,
                                                           p.pose.position.y,
                                                           p.pose.position.z],
                                                 orientation=[p.pose.orientation.x,
                                                              p.pose.orientation.y,
                                                              p.pose.orientation.z,
                                                              p.pose.orientation.w])
            # TODO not necessary to parse collision goals every time
            collision_goals = self.god_map.get_data([self.collision_goal_identifier])
            collision_matrix = self.collision_goals_to_collision_matrix(collision_goals)
            collisions = self.world.check_collisions(collision_matrix,
                                                     self_collision=self.enable_self_collision)

            closest_point = self.collisions_to_closest_point(collisions, collision_matrix)

            if self.marker:
                self.make_cpi_markers(closest_point)

            self.god_map.set_data([self.closest_point_identifier], closest_point)

    def collision_goals_to_collision_matrix(self, collision_goals):
        """
        :param collision_goals: list of CollisionEntry
        :type collision_goals: list
        :return: dict mapping (robot_link, body_b, link_b) -> min allowed distance
        :rtype: dict
        """
        if collision_goals is None:
            collision_goals = []
        min_allowed_distance = dict()
        if len([x for x in collision_goals if x.type in [CollisionEntry.AVOID_ALL_COLLISIONS,
                                                         CollisionEntry.ALLOW_ALL_COLLISIONS]]) == 0:
            # add avoid all collision if there is no other avoid or allow all
            collision_goals.insert(0, CollisionEntry(type=CollisionEntry.AVOID_ALL_COLLISIONS,
                                                     min_dist=self.default_collision_avoidance_distance))

        controllable_links = self.god_map.get_data([self.controllable_links_identifier])

        for collision_entry in collision_goals:  # type: CollisionEntry
            # check if msg got properly filled
            if collision_entry.body_b == u'' and collision_entry.link_b != u'':
                raise PhysicsWorldException(u'body_b is empty but link_b is not')

            # if robot link is empty, use all robot links
            if collision_entry.robot_link == u'':
                robot_links = set(self.world.get_robot().get_link_names())
            elif collision_entry.robot_link in self.world.get_robot().get_link_names():
                # TODO this check is linear but could be constant
                robot_links = {collision_entry.robot_link}
            else:
                raise UnknownBodyException(u'robot_link \'{}\' unknown'.format(collision_entry.robot_link))

            # remove all non controllable links
            # TODO make pybullet robot know which links are controllable
            # TODO on first look controllable links are none
            if controllable_links is not None:
                robot_links.intersection_update(controllable_links)

            # if body_b is empty, use all objects
            if collision_entry.body_b == u'':
                bodies_b = self.world.get_object_names()
            elif self.world.has_object(collision_entry.body_b):
                bodies_b = [collision_entry.body_b]
            else:
                raise UnknownBodyException(u'body_b \'{}\' unknown'.format(collision_entry.body_b))

            for body_b in bodies_b:
                # if link_b is empty, use all links from body_b
                if collision_entry.link_b == u'':  # empty link b means every link from body b
                    links_b = self.world.get_object(body_b).get_link_names()
                elif collision_entry.link_b in self.world.get_object(body_b).get_link_names():
                    links_b = [collision_entry.link_b]
                else:
                    raise UnknownBodyException(u'link_b \'{}\' unknown'.format(collision_entry.link_b))

                for robot_link, link_b in product(robot_links, links_b):
                    key = (robot_link, body_b, link_b)
                    if collision_entry.type == CollisionEntry.ALLOW_COLLISION or \
                            collision_entry.type == CollisionEntry.ALLOW_ALL_COLLISIONS:
                        if key in min_allowed_distance:
                            del min_allowed_distance[key]
                    elif collision_entry.type == CollisionEntry.AVOID_COLLISION or \
                            collision_entry.type == CollisionEntry.AVOID_ALL_COLLISIONS:
                        min_allowed_distance[key] = collision_entry.min_dist

        return min_allowed_distance

    def collisions_to_closest_point(self, collisions, distances):
        """

        :param collisions:
        :param distances:
        :return:
        """
        closest_point = keydefaultdict(lambda k: ClosestPointInfo((10, 0, 0),
                                                                  (0, 0, 0),
                                                                  1e9,
                                                                  self.default_collision_avoidance_distance,
                                                                  k,
                                                                  '',
                                                                  (1, 0, 0)))
        for key, collision_info in collisions.items():  # type: ((str, str), ContactInfo)
            if collision_info is None:
                continue
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
                                   distances[key], key[0], u'{} - {}'.format(key[1], key[2]), n_in_robot_root)
            if link1 in closest_point:
                closest_point[link1] = min(closest_point[link1], cpi, key=lambda x: x.contact_distance)
            else:
                closest_point[link1] = cpi
        return closest_point

    def stop(self):
        self.clear_world()
        self.srv_update_world.shutdown()
        self.srv_viz_gui.shutdown()
        self.pub_collision_marker.unregister()
        self.world.deactivate_viewer()

    def make_cpi_markers(self, collisions):
        m = Marker()
        m.header.frame_id = self.robot_root
        m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.id = 1337
        m.ns = u'pybullet collisions'
        m.scale = Vector3(0.003, 0, 0)
        if len(collisions) > 0:
            for collision_info in collisions.values():  # type: ClosestPointInfo
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
        ma = MarkerArray()
        ma.markers.append(m)
        self.pub_collision_marker.publish(ma)

    def object_js_cb(self, object_name, msg):
        """
        Callback message for ROS Subscriber on JointState to get states of articulated objects into world.
        :param object_name: Name of the object for which the Joint State message is.
        :type object_name: str
        :param msg: Current state of the articulated object that shall be set in the world.
        :type msg: JointState
        """
        self.object_joint_states[object_name] = to_joint_state_dict(msg)
