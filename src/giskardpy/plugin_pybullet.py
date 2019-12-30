import traceback
from multiprocessing import Lock

import rospy
from geometry_msgs.msg import PoseStamped
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest
from py_trees import Status
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

import giskardpy.identifier as identifier
from giskardpy import logging
from giskardpy.exceptions import CorruptShapeException, UnknownBodyException, \
    UnsupportedOptionException, DuplicateNameException
from giskardpy.plugin import GiskardBehavior
from giskardpy.tfwrapper import transform_pose
from giskardpy.utils import to_joint_state_dict
from giskardpy.world_object import WorldObject


class WorldUpdatePlugin(GiskardBehavior):
    # TODO reject changes if plugin not active or something
    def __init__(self, name):
        super(WorldUpdatePlugin, self).__init__(name)
        self.map_frame = self.get_god_map().safe_get_data(identifier.map_frame)
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects

    def setup(self, timeout=5.0):
        # TODO make service name a parameter
        self.srv_update_world = rospy.Service(u'~update_world', UpdateWorld, self.update_world_cb)
        self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        return super(WorldUpdatePlugin, self).setup(timeout)

    def update(self):
        """
        updated urdfs in god map and updates pybullet object joint states
        """
        with self.lock:
            pass
            for object_name, object_joint_state in self.object_joint_states.items():
                self.get_world().get_object(object_name).joint_state = object_joint_state

        return Status.SUCCESS

    def object_js_cb(self, object_name, msg):
        """
        Callback message for ROS Subscriber on JointState to get states of articulated objects into world.
        :param object_name: Name of the object for which the Joint State message is.
        :type object_name: str
        :param msg: Current state of the articulated object that shall be set in the world.
        :type msg: JointState
        """
        self.object_joint_states[object_name] = to_joint_state_dict(msg)

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
                if req.operation == UpdateWorldRequest.ADD:
                    if req.rigidly_attached:
                        self.attach_object(req)
                    else:
                        self.add_object(req)

                elif req.operation == UpdateWorldRequest.REMOVE:
                    # why not to detach objects here:
                    #   - during attaching, bodies turn to objects
                    #   - detaching actually requires a joint name
                    #   - you might accidentally detach parts of the robot
                    # if self.get_robot().has_joint(req.body.name):
                    #     self.detach_object(req)
                    self.remove_object(req.body.name)
                elif req.operation == UpdateWorldRequest.ALTER:
                    self.remove_object(req.body.name)
                    self.add_object(req)
                elif req.operation == UpdateWorldRequest.REMOVE_ALL:
                    self.clear_world()
                elif req.operation == UpdateWorldRequest.DETACH:
                    self.detach_object(req)
                else:
                    return UpdateWorldResponse(UpdateWorldResponse.INVALID_OPERATION,
                                               u'Received invalid operation code: {}'.format(req.operation))
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
        global_pose = transform_pose(self.map_frame, req.pose).pose
        world_object = WorldObject.from_world_body(world_body)
        self.get_world().add_object(world_object)
        self.get_world().set_object_pose(world_body.name, global_pose)
        try:
            m = self.get_world().get_object(world_body.name).as_marker_msg()
            m.header.frame_id = self.map_frame
            self.publish_object_as_marker(m)
        except:
            pass
        # SUB-CASE: If it is an articulated object, open up a joint state subscriber
        # FIXME also keep track of base pose
        if world_body.joint_state_topic:
            callback = (lambda msg: self.object_js_cb(world_body.name, msg))
            self.object_js_subs[world_body.name] = \
                rospy.Subscriber(world_body.joint_state_topic, JointState, callback, queue_size=1)

    def detach_object(self, req):
        self.get_world().detach(req.body.name)
        try:
            m = self.get_world().get_object(req.body.name).as_marker_msg()
            m.header.frame_id = self.map_frame
            self.publish_object_as_marker(m)
        except:
            pass

    def attach_object(self, req):
        """
        :type req: UpdateWorldRequest
        """
        if self.get_world().has_object(req.body.name):
            p = PoseStamped()
            p.header.frame_id = self.map_frame
            p.pose = self.get_world().get_object(req.body.name).base_pose
            p = transform_pose(req.pose.header.frame_id, p)
            world_object = self.get_world().get_object(req.body.name)
            self.get_world().attach_existing_obj_to_robot(req.body.name, req.pose.header.frame_id, p.pose)
            m = world_object.as_marker_msg()
            m.header.frame_id = p.header.frame_id
            m.pose = p.pose
        else:
            world_object = WorldObject.from_world_body(req.body)
            self.get_world().robot.attach_urdf_object(world_object,
                                                      req.pose.header.frame_id,
                                                      req.pose.pose)
            logging.loginfo(u'--> attached object {} on link {}'.format(req.body.name, req.pose.header.frame_id))
            m = world_object.as_marker_msg()
            m.pose = req.pose.pose
            m.header = req.pose.header
        try:
            m.frame_locked = True
            self.publish_object_as_marker(m)
        except:
            pass

    def remove_object(self, name):
        try:
            m = self.get_world().get_object(name).as_marker_msg()
            m.action = m.DELETE
            self.publish_object_as_marker(m)
        except:
            pass
        self.get_world().remove_object(name)
        if name in self.object_js_subs:
            self.object_js_subs[name].unregister()
            del (self.object_js_subs[name])
            try:
                del (self.object_joint_states[name])
            except:
                pass

    def clear_world(self):
        self.delete_markers()
        self.get_world().soft_reset()
        for v in self.object_js_subs.values():
            v.unregister()
        self.object_js_subs = {}
        self.object_joint_states = {}

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
