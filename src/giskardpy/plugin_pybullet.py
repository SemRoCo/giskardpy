import traceback
from copy import copy
from itertools import product
import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3, PoseStamped
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
from giskardpy.identifier import collision_goal_identifier, closest_point_identifier
from giskardpy.plugin import PluginBase
from giskardpy.pybullet_world import PyBulletWorld, ContactInfo
from giskardpy.tfwrapper import transform_pose, lookup_transform, transform_point, transform_vector
from giskardpy.data_types import ClosestPointInfo
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import keydefaultdict, to_joint_state_dict, to_point_stamped, to_vector3_stamped, msg_to_list, \
    make_urdf_world_body
from giskardpy.world_object import WorldObject


# class PybulletPlugin(PluginBase):
#     pass
# def __init__(self):
# self.path_to_data_folder = path_to_data_folder
# self.gui = gui
# self.robot_name = rospy.get_param(u'robot_description').split('\n', 1)[1].split('" ', 1)[0].split('"')[1]
# super(PybulletPlugin, self).__init__()

# def setup(self):
#     self.world = self.get_god_map().safe_get_data([pybullet_identifier])
#     self.controlled_joints = self.get_god_map().safe_get_data([controlled_joints_identifier])
#     if self.world is None:
#         self.world = PyBulletWorld(enable_gui=self.gui, path_to_data_folder=self.path_to_data_folder)
#         self.world.setup()
#         # TODO get robot description from god map
#         urdfs = rospy.get_param(u'robot_description')
#         self.world.add_robot(self.robot_name, urdfs, self.controlled_joints)
#         self.god_map.safe_set_data([pybullet_identifier], self.world)


# class PyBulletMonitor(PybulletPlugin):
#     """
#     Syncs pybullet with god map.
#     """
#     pass
# def __init__(self, map_frame, root_link):
#     self.map_frame = map_frame
#     self.root_link = root_link
#     super(PyBulletMonitor, self).__init__()
#     self.world = self.god_map.safe_get_data(world_identifier)

# def update(self):
#     """
#     updates robot position in pybullet
#     :return:
#     """
# js = self.god_map.safe_get_data([js_identifier])
# if js is not None:
#     self.world.set_robot_joint_state(js)
# p = lookup_transform(self.map_frame, self.root_link)
# self.world.get_robot().set_base_pose(position=[p.pose.position.x,
#                                                p.pose.position.y,
#                                                p.pose.position.z],
#                                      orientation=[p.pose.orientation.x,
#                                                   p.pose.orientation.y,
#                                                   p.pose.orientation.z,
#                                                   p.pose.orientation.w])
# TODO make sure this doesn't cause multi threading problems
# self.god_map.safe_set_data([robot_description_identifier], self.world.get_robot().get_urdf())
# return None


class PyBulletUpdatePlugin(PluginBase):
    # TODO reject changes if plugin not active or something
    def __init__(self):
        super(PyBulletUpdatePlugin, self).__init__()
        self.global_reference_frame_name = u'map'
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects

    def setup(self):
        super(PyBulletUpdatePlugin, self).setup()
        # TODO make service name a parameter
        self.srv_update_world = rospy.Service(u'~update_world', UpdateWorld, self.update_world_cb)
        self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)

    def publish_object_as_marker(self, m):
        """
        :type object_: WorldObject
        """
        try:
            ma = MarkerArray()
            m.ns = u'world' + m.ns
            ma.markers.append(m)
            self.pub_collision_marker.publish(ma)
        except:
            pass

    def delete_markers(self):
        self.pub_collision_marker.publish(MarkerArray([Marker(action=Marker.DELETEALL)]))

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
                        # get object pose
                        self.attach_object(req)
                        # req.operation = UpdateWorldRequest.REMOVE
                        # self.publish_object_as_marker(req)
                        # req.operation = UpdateWorldRequest.ADD
                    else:
                        self.add_object(req)

                elif req.operation is UpdateWorldRequest.REMOVE:
                    self.remove_object(req.body.name)
                elif req.operation is UpdateWorldRequest.ALTER:
                    self.remove_object(req.body.name)
                    self.add_object(req)
                elif req.operation is UpdateWorldRequest.REMOVE_ALL:
                    self.clear_world()
                elif req.operation is UpdateWorldRequest.DETACH:
                    self.detach_object(req)
                else:
                    return UpdateWorldResponse(UpdateWorldResponse.INVALID_OPERATION,
                                               u'Received invalid operation code: {}'.format(req.operation))
                # self.publish_object_as_marker(req)
                return UpdateWorldResponse()
            except CorruptShapeException as e:
                traceback.print_exc()
                return UpdateWorldResponse(UpdateWorldResponse.CORRUPT_SHAPE_ERROR, str(e))
            except UnknownBodyException as e:
                return UpdateWorldResponse(UpdateWorldResponse.MISSING_BODY_ERROR, str(e))
            except KeyError as e:
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
        global_pose = transform_pose(self.global_reference_frame_name, req.pose).pose
        world_object = WorldObject.from_world_body(world_body)
        self.get_world().add_object(world_object)
        self.get_world().set_object_pose(world_body.name, global_pose)
        try:
            m = self.get_world().get_object(world_body.name).as_marker_msg()
            m.header.frame_id = self.global_reference_frame_name
            self.publish_object_as_marker(m)
        except:
            pass
        # SUB-CASE: If it is an articulated object, open up a joint state subscriber
        # FIXME
        # if world_body.joint_state_topic:
        #     callback = (lambda msg: self.object_js_cb(world_body.name, msg))
        #     self.object_js_subs[world_body.name] = \
        #         rospy.Subscriber(world_body.joint_state_topic, JointState, callback, queue_size=1)

    def detach_object(self, req):
        self.get_world().detach(req.body.name)
        try:
            m = self.get_world().get_object(req.body.name).as_marker_msg()
            m.header.frame_id = self.global_reference_frame_name
            self.publish_object_as_marker(m)
        except:
            pass


    def attach_object(self, req):
        """
        :type req: UpdateWorldRequest
        """
        if self.get_world().has_object(req.body.name):
            p = PoseStamped()
            p.header.frame_id = self.global_reference_frame_name
            p.pose = self.get_world().get_object(req.body.name).base_pose
            p = transform_pose(req.pose.header.frame_id, p)
            world_object = self.get_world().get_object(req.body.name)
            self.get_world().attach_existing_obj_to_robot(req.body.name, req.pose.header.frame_id,
                                                          p.pose)
            m = world_object.as_marker_msg()
            m.header.frame_id = p.header.frame_id
            m.pose = p.pose
        else:
            world_object = WorldObject.from_world_body(req.body)
            self.get_world().robot.attach_urdf_object(world_object,
                                                      req.pose.header.frame_id,
                                                      req.pose.pose)
            m = world_object.as_marker_msg()
            m.pose = req.pose.pose
            m.header = req.pose.header
        try:
            m.frame_locked = True
            self.publish_object_as_marker(m)
        except:
            pass

    def remove_object(self, name):
        # FIXME update joint state publisher shit
        m = self.get_world().get_object(name).as_marker_msg()
        m.action = m.DELETE
        self.publish_object_as_marker(m)
        self.get_world().remove_object(name)
        # if self.world.has_object(name):
        #     self.world.remove_object(name)
        #     if self.object_js_subs.has_key(name):
        #         self.object_js_subs[name].unregister()
        #         del (self.object_js_subs[name])
        #         try:
        #             del (self.object_joint_states[name])
        #         except:
        #             pass
        # elif self.world.get_robot().can_attach_object(name):
        #     self.world.get_robot().detach_object(name)
        # else:
        #     raise UnknownBodyException(u'Cannot delete unknown object {}'.format(name))

    def clear_world(self):
        self.delete_markers()
        self.get_world().soft_reset()
        # for object_name in self.world.get_object_names():
        #     if object_name != u'plane' and object_name != u'pybullet_sucks':  # TODO get rid of this hard coded special case
        #         self.remove_object(object_name)
        # self.world.get_robot().detach_all_objects()

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
        updated urdfs in god map and updates pybullet object joint states
        """
        with self.lock:
            pass
            # TODO FIXME
            # for object_name, object_joint_state in self.object_joint_states.items():
            #     self.world.get_object(object_name).set_joint_state(object_joint_state)

        return super(PyBulletUpdatePlugin, self).update()


class CollisionChecker(PluginBase):
    def __init__(self, default_collision_avoidance_distance, map_frame, root_link):
        super(CollisionChecker, self).__init__()
        self.default_min_dist = default_collision_avoidance_distance
        self.map_frame = map_frame
        self.robot_root = root_link
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
        collision_goals = self.get_god_map().safe_get_data(collision_goal_identifier)
        self.collision_matrix = self.get_world().collision_goals_to_collision_matrix(collision_goals,
                                                                                     self.default_min_dist)
        self.get_god_map().safe_set_data(closest_point_identifier, None)
        super(CollisionChecker, self).initialize()

    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        with self.lock:
            # TODO not necessary to parse collision goals every time
            collisions = self.get_world().check_collisions(self.collision_matrix)

            closest_point = self.get_world().collisions_to_closest_point(collisions, self.collision_matrix)

            if self.marker:
                self.publish_cpi_markers(closest_point)

            self.god_map.safe_set_data(closest_point_identifier, closest_point)
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
        return self.allow_collision_with_object(u'plane')

    def allow_collision_with_pybullet_hack(self):
        return self.allow_collision_with_object(u'pybullet_sucks')

    def allow_collision_with_object(self, name):
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.body_b = name
        return ce
