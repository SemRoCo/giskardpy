import traceback
from copy import copy
from itertools import product

import rospy
from geometry_msgs.msg import Point, Vector3
from giskard_msgs.msg import CollisionEntry, WorldBody
from multiprocessing import Lock

from py_trees import Status
from std_msgs.msg import ColorRGBA
from std_srvs.srv import SetBool, SetBoolResponse
from sensor_msgs.msg import JointState
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest
from visualization_msgs.msg import Marker, MarkerArray
from giskardpy.exceptions import CorruptShapeException, UnknownBodyException, \
    UnsupportedOptionException, DuplicateNameException, PhysicsWorldException
from giskardpy.object import to_marker, world_body_to_urdf_object, from_pose_msg
from giskardpy.plugin import NewPluginBase
from giskardpy.pybullet_world import PyBulletWorld, ContactInfo
from giskardpy.tfwrapper import transform_pose, lookup_transform, transform_point, transform_vector
from giskardpy.data_types import ClosestPointInfo
from giskardpy.utils import keydefaultdict, to_joint_state_dict, to_point_stamped, to_vector3_stamped, msg_to_list


class PybulletPlugin(NewPluginBase):
    def __init__(self, pybullet_identifier, controlled_joints_identifier, path_to_data_folder='', gui=False):
        self.pybullet_identifier = pybullet_identifier
        self.controlled_joints_identifier = controlled_joints_identifier
        self.path_to_data_folder = path_to_data_folder
        self.gui = gui
        # TODO find a non hacky way to get robot name from urdf
        self.robot_name = rospy.get_param(u'robot_description').split('\n', 1)[1].split('" ', 1)[0].split('"')[1]
        super(PybulletPlugin, self).__init__()

    def setup(self):
        self.world = self.get_god_map().safe_get_data([self.pybullet_identifier])
        self.controlled_joints = self.get_god_map().safe_get_data([self.controlled_joints_identifier])
        if self.world is None:
            self.world = PyBulletWorld(enable_gui=self.gui, path_to_data_folder=self.path_to_data_folder)
            self.world.activate_viewer()
            # TODO get robot description from god map
            urdf = rospy.get_param(u'robot_description')
            self.world.spawn_robot_from_urdf(self.robot_name, urdf, self.controlled_joints)
            self.god_map.safe_set_data([self.pybullet_identifier], self.world)


class PyBulletMonitor(PybulletPlugin):
    """
    Syncs pybullet with god map.
    """

    def __init__(self, js_identifier, pybullet_identifier, controlled_joints_identifier, map_frame, root_link,
                 path_to_data_folder='', gui=False):
        self.js_identifier = js_identifier
        self.robot_name = u'robby'
        self.map_frame = map_frame
        self.root_link = root_link
        super(PyBulletMonitor, self).__init__(pybullet_identifier, controlled_joints_identifier, path_to_data_folder, gui)
        self.world = self.god_map.safe_get_data([self.pybullet_identifier])

    def update(self):
        """
        updates robot position in pybullet
        :return:
        """
        js = self.god_map.safe_get_data([self.js_identifier])
        if js is not None:
            self.world.set_robot_joint_state(js)
        p = lookup_transform(self.map_frame, self.root_link)
        self.world.get_robot().set_base_pose(position=[p.pose.position.x,
                                                       p.pose.position.y,
                                                       p.pose.position.z],
                                             orientation=[p.pose.orientation.x,
                                                          p.pose.orientation.y,
                                                          p.pose.orientation.z,
                                                          p.pose.orientation.w])
        return None


class PyBulletUpdatePlugin(PybulletPlugin):
    # TODO reject changes if plugin not active or something
    def __init__(self, pybullet_identifier, controlled_joints_identifier, robot_description_identifier,
                 path_to_data_folder='', gui=False):
        super(PyBulletUpdatePlugin, self).__init__(pybullet_identifier, controlled_joints_identifier, path_to_data_folder, gui)
        self.robot_description_identifier = robot_description_identifier
        self.global_reference_frame_name = u'map'
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects

    def setup(self):
        super(PyBulletUpdatePlugin, self).setup()
        # TODO make service name a parameter
        self.srv_update_world = rospy.Service(u'~update_world', UpdateWorld, self.update_world_cb)
        self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)

    def publish_object_as_marker(self, req):
        try:
            ma = to_marker(req)
            self.pub_collision_marker.publish(ma)
        except:
            pass

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
                elif req.operation is UpdateWorldRequest.ALTER:
                    self.remove_object(req.body.name)
                    self.add_object(req)
                elif req.operation is UpdateWorldRequest.REMOVE_ALL:
                    self.clear_world()
                else:
                    return UpdateWorldResponse(UpdateWorldResponse.INVALID_OPERATION,
                                               u'Received invalid operation code: {}'.format(req.operation))
                self.publish_object_as_marker(req)
                return UpdateWorldResponse()
            except CorruptShapeException as e:
                traceback.print_exc()
                return UpdateWorldResponse(UpdateWorldResponse.CORRUPT_SHAPE_ERROR, str(e))
            except UnknownBodyException as e:
                return UpdateWorldResponse(UpdateWorldResponse.MISSING_BODY_ERROR, str(e))
            except DuplicateNameException as e:
                return UpdateWorldResponse(UpdateWorldResponse.DUPLICATE_BODY_ERROR, str(e))
            except UnsupportedOptionException as e:
                return UpdateWorldResponse(UpdateWorldResponse.UNSUPPORTED_OPTIONS, str(e))
            except Exception as e:
                traceback.print_exc()
                return UpdateWorldResponse(UpdateWorldResponse.UNSUPPORTED_OPTIONS,
                                           u'{}: {}'.format(e.__class__.__name__,
                                                            str(e)))

    def add_object(self, req):
        """
        :type req: UpdateWorldRequest
        """
        world_body = req.body
        global_pose = from_pose_msg(transform_pose(self.global_reference_frame_name, req.pose).pose)
        if world_body.type is WorldBody.URDF_BODY:
            # TODO test me
            self.world.spawn_object_from_urdf_str(world_body.name, world_body.urdf, global_pose)
        else:
            self.world.spawn_urdf_object(world_body_to_urdf_object(world_body), global_pose)

        # SUB-CASE: If it is an articulated object, open up a joint state subscriber
        if world_body.joint_state_topic:
            callback = (lambda msg: self.object_js_cb(world_body.name, msg))
            self.object_js_subs[world_body.name] = \
                rospy.Subscriber(world_body.joint_state_topic, JointState, callback, queue_size=1)

    def attach_object(self, req):
        """
        :type req: UpdateWorldRequest
        """
        if req.body.type is WorldBody.URDF_BODY:
            raise UnsupportedOptionException(u'Attaching URDF bodies to robots is not supported.')
        if req.pose.header.frame_id not in self.world.get_robot().get_link_names():
            raise CorruptShapeException(u'robot link \'{}\' does not exist'.format(req.pose.header.frame_id))
        if self.world.has_object(req.body.name):
            self.world.attach_object(req.body,
                                     req.pose.header.frame_id,
                                     from_pose_msg(req.pose.pose))
        else:
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

    def clear_world(self):
        self.pub_collision_marker.publish(MarkerArray([Marker(action=Marker.DELETEALL)]))
        for object_name in self.world.get_object_names():
            if object_name != u'plane':  # TODO get rid of this hard coded special case
                self.remove_object(object_name)
        self.world.get_robot().detach_all_objects()

    def object_js_cb(self, object_name, msg):
        """
        Callback message for ROS Subscriber on JointState to get states of articulated objects into world.
        :param object_name: Name of the object for which the Joint State message is.
        :type object_name: str
        :param msg: Current state of the articulated object that shall be set in the world.
        :type msg: JointState
        """
        self.object_joint_states[object_name] = to_joint_state_dict(msg)

    def update(self):
        """
        updated urdf in god map and updates pybullet object joint states
        """
        with self.lock:
            # TODO only update urdf if it has changed
            self.god_map.safe_set_data([self.robot_description_identifier], self.world.get_robot().get_urdf())

            for object_name, object_joint_state in self.object_joint_states.items():
                self.world.get_object(object_name).set_joint_state(object_joint_state)

        return super(PyBulletUpdatePlugin, self).update()


class CollisionChecker(PybulletPlugin):
    def __init__(self, collision_goal_identifier, controllable_links_identifier, pybullet_identifier, controlled_joints_identifier,
                 closest_point_identifier, default_collision_avoidance_distance,
                 map_frame, root_link,
                 path_to_data_folder='', gui=False):
        super(CollisionChecker, self).__init__(pybullet_identifier, controlled_joints_identifier, path_to_data_folder, gui)
        self.collision_goal_identifier = collision_goal_identifier
        self.controllable_links_identifier = controllable_links_identifier
        self.closest_point_identifier = closest_point_identifier
        self.default_collision_avoidance_distance = default_collision_avoidance_distance
        self.map_frame = map_frame
        self.robot_root = root_link
        self.robot_name = 'pr2'
        self.global_reference_frame_name = 'map'
        self.gui = gui
        self.marker = True
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects

    def setup(self):
        super(CollisionChecker, self).setup()
        self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.srv_viz_gui = rospy.Service(u'~enable_marker', SetBool, self.enable_marker_cb)
        rospy.sleep(.5)

    def initialize(self):
        collision_goals = self.god_map.safe_get_data([self.collision_goal_identifier])
        self.collision_matrix = self.collision_goals_to_collision_matrix(collision_goals)
        self.god_map.safe_set_data([self.closest_point_identifier], None)
        super(CollisionChecker, self).initialize()

    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        with self.lock:
            # TODO not necessary to parse collision goals every time
            collisions = self.world.check_collisions(self.collision_matrix)

            closest_point = self.collisions_to_closest_point(collisions, self.collision_matrix)

            if self.marker:
                self.publish_cpi_markers(closest_point)

            self.god_map.safe_set_data([self.closest_point_identifier], closest_point)
        return super(CollisionChecker, self).update()

    def enable_marker_cb(self, setbool):
        """
        :type setbool: std_srvs.srv._SetBool.SetBoolRequest
        :rtype: SetBoolResponse
        """
        # TODO test me
        self.marker = setbool.data
        return SetBoolResponse()

    def publish_cpi_markers(self, closest_point_infos):
        """
        Publishes a string for each ClosestPointInfo in the dict. If the distance is below the threshold, the string
        is colored red. If it is below threshold*2 it is yellow. If it is below threshold*3 it is green.
        Otherwise no string will be published.
        :type closest_point_infos: dict
        """
        m = Marker()
        m.header.frame_id = self.robot_root
        m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.id = 1337
        # TODO make namespace parameter
        m.ns = u'pybullet collisions'
        m.scale = Vector3(0.003, 0, 0)
        if len(closest_point_infos) > 0:
            for collision_info in closest_point_infos.values():  # type: ClosestPointInfo
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
            # else:
            #     m.action = Marker.DELETE
        ma = MarkerArray()
        ma.markers.append(m)
        self.pub_collision_marker.publish(ma)

    def allow_collision_with_plane(self):
        # TODO instead of ignoring plane collision by default, figure out that some collision are unavoidable?
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.body_b = u'plane'
        return ce

    def collision_matrix_to_min_dist_dict(self, collision_matrix, robot_name, min_dist):
        return {(link1, robot_name, link2): min_dist for link1, link2 in collision_matrix}

    def collision_goals_to_collision_matrix(self, collision_goals):
        """
        :param collision_goals: list of CollisionEntry
        :type collision_goals: list
        :return: dict mapping (robot_link, body_b, link_b) -> min allowed distance
        :rtype: dict
        """
        # TODO split this into smaller functions
        if collision_goals is None:
            collision_goals = []
        #FIXME
        collision_matrix = self.collision_matrix_to_min_dist_dict(self.world.get_robot().get_self_collision_matrix(),
                                                                  self.world.get_robot().name,
                                                                  self.default_collision_avoidance_distance)
        min_allowed_distance = collision_matrix

        collision_goals.insert(0, self.allow_collision_with_plane()) # FIXME shouldn't this be the last entry?

        if len([x for x in collision_goals if x.type in [CollisionEntry.AVOID_ALL_COLLISIONS,
                                                         CollisionEntry.ALLOW_ALL_COLLISIONS]]) == 0:
            # add avoid all collision if there is no other avoid or allow all
            collision_goals.insert(0, CollisionEntry(type=CollisionEntry.AVOID_ALL_COLLISIONS,
                                                     min_dist=self.default_collision_avoidance_distance))

        controllable_links = self.god_map.safe_get_data([self.controllable_links_identifier])

        for collision_entry in collision_goals:  # type: CollisionEntry
            if collision_entry.type in [CollisionEntry.ALLOW_ALL_COLLISIONS,
                                        CollisionEntry.AVOID_ALL_COLLISIONS]:
                if collision_entry.robot_links != []:
                    rospy.logwarn(u'type==AVOID_ALL_COLLISION but robot_links is set, it will be ignored.')
                    collision_entry.robot_links = []
                if collision_entry.body_b != u'':
                    rospy.logwarn(u'type==AVOID_ALL_COLLISION but body_b is set, it will be ignored.')
                    collision_entry.body_b = u''
                if collision_entry.link_bs != []:
                    rospy.logwarn(u'type==AVOID_ALL_COLLISION but link_bs is set, it will be ignored.')
                    collision_entry.link_bs = []

                if collision_entry.type == CollisionEntry.ALLOW_ALL_COLLISIONS:
                    min_allowed_distance = {}
                    continue
                else:
                    min_allowed_distance = collision_matrix

            # check if msg got properly filled
            if collision_entry.body_b == u'' and collision_entry.link_bs != []:
                raise PhysicsWorldException(u'body_b is empty but link_b is not')

            # if robot link is empty, use all robot links
            if collision_entry.robot_links == []:
                robot_links = set(self.world.get_robot().get_link_names())
            else:
                for robot_link in collision_entry.robot_links:
                    if robot_link not in self.world.get_robot().get_link_names():
                        raise UnknownBodyException(u'robot_link \'{}\' unknown'.format(robot_link))
                robot_links = set(collision_entry.robot_links)

            # remove all non controllable links
            # TODO make pybullet robot know which links are controllable?
            robot_links.intersection_update(controllable_links)

            # if body_b is empty, use all objects
            if collision_entry.body_b == u'':
                bodies_b = self.world.get_object_names()
                # if collision_entry.type == CollisionEntry.AVOID_COLLISION:
                bodies_b.append(self.world.get_robot().name)
            elif self.world.has_object(collision_entry.body_b) or \
                    collision_entry.body_b == self.world.get_robot().name:
                bodies_b = [collision_entry.body_b]
            else:
                raise UnknownBodyException(u'body_b \'{}\' unknown'.format(collision_entry.body_b))

            link_b_was_set = len(collision_entry.link_bs) > 0

            for body_b in bodies_b:
                # if link_b is empty, use all links from body_b
                link_bs = collision_entry.link_bs
                if body_b != self.world.get_robot().name:
                    if link_bs == []:
                        link_bs = self.world.get_object(body_b).get_link_names()
                    elif link_bs != []:
                        for link_b in link_bs:
                            # TODO use sets and intersection to safe time
                            if link_b not in self.world.get_object(body_b).get_link_names():
                                raise UnknownBodyException(u'link_b \'{}\' unknown'.format(link_b))

                for robot_link in robot_links:
                    if not link_b_was_set and body_b == self.world.get_robot().name:
                        link_bs = self.world.get_robot().get_possible_collisions(robot_link)
                    for link_b in link_bs:
                        keys = [(robot_link, body_b, link_b)]
                        if body_b == self.world.get_robot().name:
                            if link_b not in self.world.get_robot().get_possible_collisions(robot_link):
                                continue
                            keys.append((link_b, body_b, robot_link))

                        for key in keys:
                            if collision_entry.type == CollisionEntry.ALLOW_COLLISION:
                                if key in min_allowed_distance:
                                    del min_allowed_distance[key]

                            elif collision_entry.type == CollisionEntry.AVOID_COLLISION or \
                                    collision_entry.type == CollisionEntry.AVOID_ALL_COLLISIONS:
                                min_allowed_distance[key] = collision_entry.min_dist

        return min_allowed_distance

    def collisions_to_closest_point(self, collisions, min_allowed_distance):
        """
        :param collisions: (robot_link, body_b, link_b) -> ContactInfo
        :type collisions: dict
        :param min_allowed_distance: (robot_link, body_b, link_b) -> min allowed distance
        :type min_allowed_distance: dict
        :return: robot_link -> ClosestPointInfo of closest thing
        :rtype: dict
        """
        closest_point = keydefaultdict(lambda k: ClosestPointInfo((10, 0, 0),
                                                                  (0, 0, 0),
                                                                  1e9,
                                                                  self.default_collision_avoidance_distance,
                                                                  k,
                                                                  '',
                                                                  (1, 0, 0)))
        for key, collision_info in collisions.items():  # type: ((str, str, str), ContactInfo)
            if collision_info is None:
                continue
            link1 = key[0]
            a_in_robot_root = msg_to_list(transform_point(self.robot_root,
                                                          to_point_stamped(self.map_frame,
                                                                           collision_info.position_on_a)))
            b_in_robot_root = msg_to_list(transform_point(self.robot_root,
                                                          to_point_stamped(self.map_frame,
                                                                           collision_info.position_on_b)))
            n_in_robot_root = msg_to_list(transform_vector(self.robot_root,
                                                           to_vector3_stamped(self.map_frame,
                                                                              collision_info.contact_normal_on_b)))
            try:
                cpi = ClosestPointInfo(a_in_robot_root, b_in_robot_root, collision_info.contact_distance,
                                       min_allowed_distance[key], key[0], u'{} - {}'.format(key[1], key[2]),
                                       n_in_robot_root)
            except KeyError:
                continue
            if link1 in closest_point:
                closest_point[link1] = min(closest_point[link1], cpi, key=lambda x: x.contact_distance)
            else:
                closest_point[link1] = cpi
        return closest_point
