import traceback
from datetime import datetime
from multiprocessing import Lock

import rospy
from geometry_msgs.msg import PoseStamped
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest, GetObjectNames,\
    GetObjectNamesResponse, GetObjectInfo, GetObjectInfoResponse, UpdateRvizMarkers, UpdateRvizMarkersResponse,\
    GetAttachedObjects, GetAttachedObjectsResponse
from py_trees import Status
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

import giskardpy.identifier as identifier
from giskardpy.data_types import JointStates
from giskardpy.exceptions import CorruptShapeException, UnknownBodyException, \
    UnsupportedOptionException, DuplicateNameException
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import transform_pose
from giskardpy.tree.tree_manager import TreeManager
from giskardpy.utils.utils import to_joint_state_position_dict, dict_to_joint_states, write_dict, logging
from giskardpy.model.world_object import WorldObject
from giskardpy.model.urdf_object import URDFObject
from rospy_message_converter.message_converter import convert_ros_message_to_dictionary

import os

class WorldUpdatePlugin(GiskardBehavior):
    # TODO reject changes if plugin not active or something
    def __init__(self, name):
        super(WorldUpdatePlugin, self).__init__(name)
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects

    def setup(self, timeout=5.0):
        # TODO make service name a parameter
        self.srv_update_world = rospy.Service(u'~update_world', UpdateWorld, self.update_world_cb)
        self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.get_object_names = rospy.Service(u'~get_object_names', GetObjectNames, self.get_object_names)
        self.get_object_info = rospy.Service(u'~get_object_info', GetObjectInfo, self.get_object_info)
        self.get_attached_objects = rospy.Service(u'~get_attached_objects', GetAttachedObjects, self.get_attached_objects)
        self.update_rviz_markers = rospy.Service(u'~update_rviz_markers', UpdateRvizMarkers, self.update_rviz_markers)
        self.dump_state_srv = rospy.Service(u'~dump_state', Trigger, self.dump_state_cb)
        return super(WorldUpdatePlugin, self).setup(timeout)

    def dump_state_cb(self, data):
        try:
            path = self.get_god_map().unsafe_get_data(identifier.data_folder)
            folder_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            folder_path = '{}{}'.format(path, folder_name)
            os.mkdir(folder_path)
            robot = self.unsafe_get_robot()
            world = self.unsafe_get_world()
            with open("{}/dump.txt".format(folder_path), u'w') as f:
                tree_manager = self.get_god_map().unsafe_get_data(identifier.tree_manager) # type: TreeManager
                joint_state_message = tree_manager.get_node(u'js1').lock.get()
                f.write("initial_robot_joint_state_dict = ")
                write_dict(to_joint_state_position_dict(joint_state_message), f)
                f.write("try:\n" +
                        "   x_joint = initial_robot_joint_state_dict[\"odom_x_joint\"]\n" +
                        "   y_joint = initial_robot_joint_state_dict[\"odom_y_joint\"]\n" +
                        "   z_joint = initial_robot_joint_state_dict[\"odom_z_joint\"]\n" +
                        "   base_pose = PoseStamped()\n" +
                        "   base_pose.header.frame_id = \"map\"\n" +
                        "   base_pose.pose.position = Point(x_joint, y_joint, 0)\n" +
                        "   base_pose.pose.orientation = Quaternion(*quaternion_about_axis(z_joint, [0, 0, 1]))\n" +
                        "   zero_pose.teleport_base(base_pose)\n" +
                        "except:\n" +
                        "   logging.loginfo(\'no x,y and z joint\')\n\n")
                f.write("zero_pose.send_and_check_joint_goal(initial_robot_joint_state_dict)\n")
                robot_base_pose = PoseStamped()
                robot_base_pose.header.frame_id = 'map'
                robot_base_pose.pose = robot.base_pose
                f.write("map_odom_transform_dict = ")
                write_dict(convert_ros_message_to_dictionary(robot_base_pose), f)
                f.write("map_odom_pose_stamped = convert_dictionary_to_ros_message(\'geometry_msgs/PoseStamped\', map_odom_transform_dict)\n")
                f.write("map_odom_transform = Transform()\n" +
                        "map_odom_transform.rotation = map_odom_pose_stamped.pose.orientation\n" +
                        "map_odom_transform.translation = map_odom_pose_stamped.pose.position\n\n")
                f.write("set_odom_map_transform = rospy.ServiceProxy('/map_odom_transform_publisher/update_map_odom_transform', UpdateTransform)\n")
                f.write("set_odom_map_transform(map_odom_transform)\n")

                original_robot = URDFObject(robot.original_urdf)
                link_names = robot.get_link_names()
                original_link_names = original_robot.get_link_names()
                attached_objects = list(set(link_names).difference(original_link_names))
                for object_name in attached_objects:
                    parent = robot.get_parent_link_of_joint(object_name)
                    pose = robot.get_fk_pose(parent, object_name)
                    world_object = robot.get_sub_tree_at_joint(object_name)
                    f.write("#attach {}\n".format(object_name))
                    with open("{}/{}.urdf".format(folder_path, object_name), u'w') as f_urdf:
                        f_urdf.write(world_object.original_urdf)

                    f.write('with open(\'{}/{}.urdf\', \"r\") as f:\n'.format(folder_path, object_name))
                    f.write("   {}_urdf = f.read()\n".format(object_name))
                    f.write("{0}_name = \"{0}\"\n".format(object_name))
                    f.write("{}_pose_stamped_dict = ".format(object_name))
                    write_dict(convert_ros_message_to_dictionary(pose), f)
                    f.write("{0}_pose_stamped = convert_dictionary_to_ros_message('geometry_msgs/PoseStamped', {0}_pose_stamped_dict)\n".format(object_name))
                    f.write(
                        "zero_pose.add_urdf(name={0}_name, urdf={0}_urdf, pose={0}_pose_stamped)\n".format(object_name))
                    f.write(
                        "zero_pose.attach_existing(name={0}_name, frame_id=\'{1}\')\n".format(object_name, parent))

                for object_name, world_object in world.get_objects().items(): # type: (str, WorldObject)
                    f.write("#add {}\n".format(object_name))
                    with open("{}/{}.urdf".format(folder_path, object_name), u'w') as f_urdf:
                        f_urdf.write(world_object.original_urdf)

                    f.write('with open(\'{}/{}.urdf\', \"r\") as f:\n'.format(folder_path,object_name))
                    f.write("   {}_urdf = f.read()\n".format(object_name))
                    f.write("{0}_name = \"{0}\"\n".format(object_name))
                    f.write("{0}_js_topic = \"{0}_js_topic\"\n".format(object_name))
                    f.write("{}_pose_dict = ".format(object_name))
                    write_dict(convert_ros_message_to_dictionary(world_object.base_pose), f)
                    f.write("{0}_pose = convert_dictionary_to_ros_message('geometry_msgs/Pose', {0}_pose_dict)\n".format(object_name))
                    f.write("{}_pose_stamped = PoseStamped()\n".format(object_name))
                    f.write("{0}_pose_stamped.pose = {0}_pose\n".format(object_name))
                    f.write("{0}_pose_stamped.header.frame_id = \"map\"\n".format(object_name))
                    f.write("zero_pose.add_urdf(name={0}_name, urdf={0}_urdf, pose={0}_pose_stamped, js_topic={0}_js_topic, set_js_topic=None)\n".format(object_name))
                    f.write("{}_joint_state = ".format(object_name))
                    write_dict(to_joint_state_position_dict((dict_to_joint_states(world_object.joint_state))), f)
                    f.write("zero_pose.set_object_joint_state({0}_name, {0}_joint_state)\n\n".format(object_name))

                last_goal = self.get_god_map().unsafe_get_data(identifier.next_move_goal)
                if last_goal:
                    f.write(u'last_goal_dict = ')
                    write_dict(convert_ros_message_to_dictionary(last_goal), f)
                    f.write("last_goal_msg = convert_dictionary_to_ros_message('giskard_msgs/MoveCmd', last_goal_dict)\n")
                    f.write("last_action_goal = MoveActionGoal()\n")
                    f.write("last_move_goal = MoveGoal()\n")
                    f.write("last_move_goal.cmd_seq = [last_goal_msg]\n")
                    f.write("last_move_goal.type = MoveGoal.PLAN_AND_EXECUTE\n")
                    f.write("last_action_goal.goal = last_move_goal\n")
                    f.write("zero_pose.send_and_check_goal(goal=last_action_goal)\n")
                else:
                    f.write(u'#no goal\n')
            logging.loginfo(u'saved dump to {}'.format(folder_path))
        except:
            logging.logerr('failed to dump state pls try again')
            res = TriggerResponse()
            res.message = 'failed to dump state pls try again'
            return TriggerResponse()
        res = TriggerResponse()
        res.success = True
        res.message = 'saved dump to {}'.format(folder_path)
        return res

    @profile
    def update(self):
        """
        updated urdfs in god map and updates pybullet object joint states
        """
        with self.lock:
            pass
            for object_name, object_joint_state in self.object_joint_states.items():
                self.get_world().get_object(object_name).joint_state = object_joint_state

        return Status.SUCCESS

    def get_object_names(self, req):
        object_names = self.get_world().get_object_names()
        res = GetObjectNamesResponse()
        res.object_names = object_names
        return res

    def get_object_info(self, req):
        res = GetObjectInfoResponse()
        res.error_codes = GetObjectInfoResponse.SUCCESS
        try:
            object = self.get_world().get_object(req.object_name)
            res.joint_state_topic = ''
            if req.object_name in self.object_js_subs.keys():
                res.joint_state_topic = self.object_js_subs[req.object_name].name
            res.pose.pose = object.base_pose
            res.pose.header.frame_id = self.get_god_map().get_data(identifier.map_frame)
            for key, value in object.joint_state.items():
                res.joint_state.name.append(key)
                res.joint_state.position.append(value.position)
                res.joint_state.velocity.append(value.velocity)
                res.joint_state.effort.append(value.effort)
        except KeyError as e:
            logging.logerr('no object with the name {} was found'.format(req.object_name))
            res.error_codes = GetObjectInfoResponse.NAME_NOT_FOUND_ERROR

        return res


    def update_rviz_markers(self, req):
        l = req.object_names
        res = UpdateRvizMarkersResponse()
        res.error_codes = UpdateRvizMarkersResponse.SUCCESS
        ma = MarkerArray()
        if len(l) == 0:
            l = self.get_world().get_object_names()
        for name in l:
            try:
                m = self.get_world().get_object(name).as_marker_msg()
                m.action = m.ADD
                m.header.frame_id = 'map'
                m.ns = u'world' + m.ns
                ma.markers.append(m)
            except TypeError as e:
                logging.logerr('failed to convert object {} to marker: {}'.format(name, str(e)))
                res.error_codes = UpdateRvizMarkersResponse.MARKER_CONVERSION_ERROR
            except KeyError as e:
                logging.logerr('could not update rviz marker for object {}: no object with that name was found'.format(name))
                res.error_codes = UpdateRvizMarkersResponse.NAME_NOT_FOUND_ERROR

        self.pub_collision_marker.publish(ma)
        return res

    def get_attached_objects(self, req):
        original_robot = URDFObject(self.get_robot().original_urdf)
        link_names = self.get_robot().get_link_names()
        original_link_names = original_robot.get_link_names()
        attached_objects = list(set(link_names).difference(original_link_names))
        attachment_points = [self.get_robot().get_parent_link_of_joint(object) for object in attached_objects]
        res = GetAttachedObjectsResponse()
        res.object_names = attached_objects
        res.attachment_points = attachment_points
        return res

    def object_js_cb(self, object_name, msg):
        """
        Callback message for ROS Subscriber on JointState to get states of articulated objects into world.
        :param object_name: Name of the object for which the Joint State message is.
        :type object_name: str
        :param msg: Current state of the articulated object that shall be set in the world.
        :type msg: JointState
        """
        self.object_joint_states[object_name] = JointStates.from_msg(msg)

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
            with self.get_god_map():
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
                    if req.body.type == req.body.MESH_BODY:
                        return UpdateWorldResponse(UpdateWorldResponse.CORRUPT_MESH_ERROR, str(e))
                    elif req.body.type == req.body.URDF_BODY:
                        return UpdateWorldResponse(UpdateWorldResponse.CORRUPT_URDF_ERROR, str(e))
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
        # assumes that parent has god map lock
        world_body = req.body
        global_pose = transform_pose(self.map_frame, req.pose).pose
        world_object = WorldObject.from_world_body(world_body)
        self.unsafe_get_world().add_object(world_object)
        self.unsafe_get_world().set_object_pose(world_body.name, global_pose)
        try:
            m = self.unsafe_get_world().get_object(world_body.name).as_marker_msg()
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
            rospy.sleep(0.1)

    def detach_object(self, req):
        # assumes that parent has god map lock
        self.unsafe_get_world().detach(req.body.name)
        try:
            m = self.unsafe_get_world().get_object(req.body.name).as_marker_msg()
            m.header.frame_id = self.map_frame
            self.publish_object_as_marker(m)
        except:
            pass

    def attach_object(self, req):
        """
        :type req: UpdateWorldRequest
        """
        # assumes that parent has god map lock
        if self.unsafe_get_world().has_object(req.body.name):
            p = PoseStamped()
            p.header.frame_id = self.map_frame
            p.pose = self.unsafe_get_world().get_object(req.body.name).base_pose
            p = transform_pose(req.pose.header.frame_id, p)
            world_object = self.unsafe_get_world().get_object(req.body.name)
            self.unsafe_get_world().attach_existing_obj_to_robot(req.body.name, req.pose.header.frame_id, p.pose)
            m = world_object.as_marker_msg()
            m.header.frame_id = p.header.frame_id
            m.pose = p.pose
        else:
            world_object = WorldObject.from_world_body(req.body)
            self.unsafe_get_world().robot.attach_urdf_object(world_object,
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
        # assumes that parent has god map lock
        try:
            m = self.unsafe_get_world().get_object(name).as_marker_msg()
            m.action = m.DELETE
            self.publish_object_as_marker(m)
        except:
            pass
        self.unsafe_get_world().remove_object(name)
        if name in self.object_js_subs:
            self.object_js_subs[name].unregister()
            del (self.object_js_subs[name])
            try:
                del (self.object_joint_states[name])
            except:
                pass

    def clear_world(self):
        # assumes that parent has god map lock
        self.delete_markers()
        self.unsafe_get_world().soft_reset()
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
